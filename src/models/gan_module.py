from typing import Dict, Literal

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from models.networks.base_nets import BaseDiscriminator, BaseGenerator
from models.networks.vgg_nets import VGGFeatureExtractor
from optim import define_criterion
from utils import net_utils


class VSRGAN(L.LightningModule):
    def __init__(
        self,
        generator: BaseGenerator,
        discriminator: BaseDiscriminator,
        *,
        crop_border_ratio: float = 0.75,
        losses: Dict,
        gen_lr: float = 5e-5,
        dis_lr: float = 5e-5,
        dis_update_policy: Literal["always", "adaptive"] = "always",
        dis_update_threshold: float = 0.4,
    ):
        super(VSRGAN, self).__init__()
        self.save_hyperparameters(ignore=["generator", "discriminator"])
        self.G = generator
        self.D = discriminator

        # pixel criterion
        self.pix_crit, self.pix_w = define_criterion(losses.get("pixel_crit"))

        # edge criterion
        self.edge_crit, self.edge_w = define_criterion(losses.get("edge_crit"))
        if self.edge_crit is not None:
            self.edge_net = net_utils.CannyDetector(
                losses["edge_crit"].get("filter_size", 3),
                losses["edge_crit"].get("std", 3),
            )

        # warping criterion
        self.warp_crit, self.warp_w = define_criterion(losses.get("warping_crit"))

        # feature criterion
        self.feat_crit, self.feat_w = define_criterion(losses.get("feature_crit"))
        if losses.get("feature_crit")["type"] == "CosineSimilarity":
            self.feat_net = VGGFeatureExtractor(
                losses["feature_crit"].get("feature_layers", [8, 17, 26, 35])
            )
        else:
            self.feat_net = None

        # flow & mask criterion
        self.flow_crit, self.flow_w = define_criterion(losses.get("flow_crit"))

        # ping-pong criterion
        self.pp_crit, self.pp_w = define_criterion(losses.get("pingpong_crit"))

        # gan criterion
        self.gan_crit, self.gan_w = define_criterion(losses.get("gan_crit"))

        self.D_upd_cnt = 0

        self.automatic_optimization = False

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optim_G = torch.optim.Adam(params=self.G.parameters(), lr=self.hparams.gen_lr)
        optim_D = torch.optim.Adam(params=self.D.parameters(), lr=self.hparams.dis_lr)

        return optim_G, optim_D

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        optim_G, optim_D = self.optimizers()

        # ------------ prepare data ------------ #
        hr_true, lr_data = batch

        n, t, c, lr_h, lr_w = lr_data.size()
        _, _, _, gt_h, gt_w = hr_true.size()

        # generate bicubic upsampled data
        bi_data = self.G.upsample_func(lr_data.view(n * t, c, lr_h, lr_w)).view(
            n, t, c, gt_h, gt_w
        )

        # augment data for pingpong criterion
        if self.pp_crit is not None:
            # i.e., (0,1,2,...,t-2,t-1) -> (0,1,2,...,t-2,t-1,t-2,...,2,1,0)
            lr_rev = lr_data.flip(1)[:, 1:, ...]
            gt_rev = hr_true.flip(1)[:, 1:, ...]
            bi_rev = bi_data.flip(1)[:, 1:, ...]

            lr_data = torch.cat([lr_data, lr_rev], dim=1)
            hr_true = torch.cat([hr_true, gt_rev], dim=1)
            bi_data = torch.cat([bi_data, bi_rev], dim=1)

        # ------------ clear optimizers ------------ #
        optim_G.zero_grad()
        optim_D.zero_grad()

        # ------------ forward G ------------ #
        net_G_output_dict = self.G(lr_data)
        hr_fake = net_G_output_dict["hr_data"]

        # ------------ forward D ------------ #
        for param in self.D.parameters():
            param.requires_grad = True

        # feed additional data
        net_D_input_dict = {
            "net_G": self.G,
            "lr_data": lr_data,
            "bi_data": bi_data,
            "use_pp_crit": self.pp_crit is not None,
            "crop_border_ratio": self.hparams.crop_border_ratio,
        }
        net_D_input_dict.update(net_G_output_dict)

        # forward real sequence (gt)
        real_pred, net_D_output_dict = self.D(hr_true, net_D_input_dict)

        # reuse internal data (e.g., lr optical flow) to reduce computations
        net_D_input_dict.update(net_D_output_dict)

        # forward fake sequence (hr)
        fake_pred, _ = self.D(hr_fake.detach(), net_D_input_dict)

        # ------------ optimize D ------------ #
        to_log, to_log_prog = {}, {}
        real_pred_D, fake_pred_D = real_pred[0], fake_pred[0]

        # select D update policy
        update_policy = self.hparams.dis_update_policy
        if update_policy == "adaptive":
            # update D adaptively
            logged_real_pred_D = torch.log(torch.sigmoid(real_pred_D) + 1e-8)
            logged_fake_pred_D = torch.log(torch.sigmoid(fake_pred_D) + 1e-8)

            distance = logged_real_pred_D.mean() - logged_fake_pred_D.mean()

            threshold = self.hparams.dis_update_threshold
            upd_D = distance.item() < threshold
            to_log["D_real_fake_distance"] = distance
        else:
            upd_D = True

        if upd_D:
            self.D_upd_cnt += 1
            loss_real_D = self.gan_crit(real_pred_D, True)
            loss_fake_D = self.gan_crit(fake_pred_D, False)
            loss_D = loss_real_D + loss_fake_D

            # update D
            self.manual_backward(loss_D)
            optim_D.step()
            to_log["D_real_loss"] = loss_real_D
            to_log["D_fake_loss"] = loss_fake_D
            to_log["D_updates_count"] = self.D_upd_cnt
        else:
            loss_D = torch.zeros(1)

        to_log_prog["D_loss"] = loss_D

        # ------------ optimize G ------------ #
        for param in self.D.parameters():
            param.requires_grad = False

        # calculate losses
        loss_G = 0

        # pixel (pix) loss
        if self.pix_crit is not None:
            loss_pix_G = self.pix_crit(hr_fake, hr_true)
            loss_G += self.pix_w * loss_pix_G
            to_log["G_pixel_loss"] = loss_pix_G

        # edge loss
        if self.edge_crit is not None:
            hr_merge = hr_fake.view(-1, c, gt_h, gt_w)
            gt_merge = hr_true.view(-1, c, gt_h, gt_w)
            loss_edge_G = self.edge_crit(
                self.edge_net(
                    hr_merge,
                    self.hparams.losses["edge_crit"].get("threshold1", 5),
                    self.hparams.losses["edge_crit"].get("threshold2", 5),
                ),
                self.edge_net(
                    gt_merge,
                    self.hparams.losses["edge_crit"].get("threshold1", 5),
                    self.hparams.losses["edge_crit"].get("threshold2", 5),
                ),
            )
            loss_G += self.edge_w * loss_edge_G
            to_log["G_edge_loss"] = loss_edge_G

        # warping (warp) loss
        if self.warp_crit is not None:
            lr_curr = net_G_output_dict["lr_curr"]
            lr_prev = net_G_output_dict["lr_prev"]
            lr_flow = net_G_output_dict["lr_flow"]
            lr_warp = net_utils.backward_warp(lr_prev, lr_flow)

            loss_warp_G = self.warp_crit(lr_warp, lr_curr)
            loss_G += self.warp_w * loss_warp_G
            to_log["G_warping_loss"] = loss_warp_G

        # feature (feat) loss
        if self.feat_crit is not None:
            hr_merge = hr_fake.view(-1, c, gt_h, gt_w)
            gt_merge = hr_true.view(-1, c, gt_h, gt_w)

            if self.feat_net is None:
                loss_feat_G = self.feat_crit(hr_merge, gt_merge.detach()).mean()
            else:
                hr_feat_lst = self.feat_net(hr_merge)
                gt_feat_lst = self.feat_net(gt_merge)
                loss_feat_G = 0
                for hr_feat, gt_feat in zip(hr_feat_lst, gt_feat_lst):
                    loss_feat_G += self.feat_crit(hr_feat, gt_feat.detach())

            loss_G += self.feat_w * loss_feat_G
            to_log_prog["G_lpip_loss"] = loss_feat_G

        # ping-pong (pp) loss
        if self.pp_crit is not None:
            tempo_extent = self.trainer.datamodule.hparams.tempo_extent
            hr_data_fw = hr_fake[:, : tempo_extent - 1, ...]  #    -------->|
            hr_data_bw = hr_fake[:, tempo_extent:, ...].flip(1)  # <--------|

            loss_pp_G = self.pp_crit(hr_data_fw, hr_data_bw)
            loss_G += self.pp_w * loss_pp_G
            to_log["G_ping_pong_loss"] = loss_pp_G

        # gan loss
        fake_pred, _ = self.D(hr_fake, net_D_input_dict)
        fake_pred_G = fake_pred[0]

        loss_gan_G = self.gan_crit(fake_pred_G, True)
        loss_G += self.gan_w * loss_gan_G
        to_log["G_gan_loss"] = loss_gan_G
        to_log_prog["G_loss"] = loss_G

        # update G
        self.manual_backward(loss_G)
        optim_G.step()

        self.log_dict(to_log_prog, prog_bar=True)
        self.log_dict(to_log, prog_bar=False)

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        hr_true, lr_data = batch

        hr_fake = self.G(lr_data)["hr_data"]

        # ssim_val = self.ssim(y_fake, y_true).mean()
        # lpips_val = self.lpips_alex(y_fake, y_true).mean()
        # self.ssim_validation.append(float(ssim_val))
        # self.lpips_validation.append(float(lpips_val))

        lr_data, hr_true, hr_fake = (
            lr_data[:, 0, :, :],
            hr_true[:, 0, :, :],
            hr_fake[:, 0, :, :],
        )

        return lr_data, hr_true, hr_fake
