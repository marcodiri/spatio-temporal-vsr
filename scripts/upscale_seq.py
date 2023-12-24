import argparse
import os

import torch
import torch.nn.functional as F
import torchvision

from models.networks.egvsr_nets import FRNet
from utils import data_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--ckpt",
    type=str,
    help="Path to lightning model checkpoint",
)
parser.add_argument(
    "-s",
    "--seq",
    type=str,
    help="Sequence to upscale (folder with frames)",
)
parser.add_argument(
    "--ext",
    type=str,
    default="png",
    help="Extension of frame images",
)

args = parser.parse_args()
generator = FRNet(scale=4).cuda()

checkpoint = torch.load(args.ckpt)
state_dict = {
    ".".join(k.split(".")[1:]): v
    for k, v in checkpoint["state_dict"].items()
    if "G." in k
}
generator.load_state_dict(state_dict)

lr_paths = sorted(data_utils.get_pics_in_subfolder(args.image, ext=args.ext))
lr_list = []
for p in lr_paths:
    lr = data_utils.load_img(p)
    lr = data_utils.transform(lr)
    lr_list.append(lr)
lr_seq = torch.stack(lr_list).cuda()

generator.eval()
hr_fake = generator.infer_sequence(lr_seq)
hr_fake = torch.clamp(hr_fake, min=-1.0, max=1.0)

hr_bic = F.interpolate(lr_seq, scale_factor=4, mode="bicubic")
hr_bic = torch.clamp(hr_bic, min=-1.0, max=1.0)

to_image = torchvision.transforms.ToPILImage()
os.makedirs("./output/fake/", exist_ok=True)
os.makedirs("./output/bic/", exist_ok=True)

frm_idx_lst = ["{:04d}.png".format(i + 1) for i in range(hr_fake.size(0))]
for i in range(hr_fake.size(0)):
    hr_f = data_utils.de_transform(hr_fake[i])
    hr_f.save(f"./output/fake/{frm_idx_lst[i]}")

    hr_b = data_utils.de_transform(hr_bic[i])
    hr_b.save(f"./output/bic/{frm_idx_lst[i]}")
