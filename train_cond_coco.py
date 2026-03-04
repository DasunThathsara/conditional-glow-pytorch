import os
import argparse
from math import log
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from transformers import CLIPTokenizer, CLIPTextModel

from coco_dataset import CocoCaptionsSimple
from model_cond import GlowCond


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_z_shapes(n_channel, input_size, n_block):
    z_shapes = []
    for _ in range(n_block - 1):
        input_size //= 2
        n_channel *= 2
        z_shapes.append((n_channel, input_size, input_size))
    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))
    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    n_pixel = image_size * image_size * 3
    loss = -log(n_bins) * n_pixel + logdet + log_p
    bpd = (-loss / (log(2) * n_pixel))
    return bpd.mean(), (log_p / (log(2) * n_pixel)).mean(), (logdet / (log(2) * n_pixel)).mean()


class CondProjector(nn.Module):
    def __init__(self, in_dim=512, cond_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def make_cond(tokenizer, text_enc, projector, captions):
    tok = tokenizer(list(captions), padding=True, truncation=True, return_tensors="pt")
    tok = {k: v.to(device) for k, v in tok.items()}
    te = text_enc(**tok).pooler_output  # [B,512]
    cond = projector(te)
    return cond


@torch.no_grad()
def sample_one(model_glow, tokenizer, text_enc, projector, prompt, z_shapes, temp, out_path):
    tok = tokenizer([prompt], padding=True, truncation=True, return_tensors="pt")
    tok = {k: v.to(device) for k, v in tok.items()}
    te = text_enc(**tok).pooler_output
    cond = projector(te)  # [1,cond_dim]

    eps_list = [torch.randn(1, c, h, w, device=device) * temp for (c, h, w) in z_shapes]
    x = model_glow.reverse(eps_list, cond, reconstruct=False).clamp(-0.5, 0.5)

    utils.save_image(x.cpu(), out_path, normalize=True, value_range=(-0.5, 0.5))


def main():
    ap = argparse.ArgumentParser("Train conditional Glow on COCO captions")
    ap.add_argument("--coco_images", type=str, required=True, help=".../coco/train2017")
    ap.add_argument("--coco_captions", type=str, required=True, help=".../coco/annotations/captions_train2017.json")

    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iter", type=int, default=200000)
    ap.add_argument("--lr", type=float, default=1e-4)

    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--n_bits", type=int, default=5)

    ap.add_argument("--n_flow", type=int, default=32)
    ap.add_argument("--n_block", type=int, default=4)
    ap.add_argument("--cond_dim", type=int, default=256)
    ap.add_argument("--affine", action="store_true")

    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=0, help="limit dataset size for quick tests (0 = full)")

    ap.add_argument("--outdir", type=str, default="runs_textglow")
    ap.add_argument("--sample_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=10000)
    ap.add_argument("--prompt_preview", type=str, default="a cat",
                    help="prompt to sample during training")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "samples"), exist_ok=True)

    # Image transforms (like your current pipeline)
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    ds = CocoCaptionsSimple(
        images_dir=args.coco_images,
        captions_json=args.coco_captions,
        transform=transform,
        max_samples=(args.max_samples if args.max_samples > 0 else None),
        seed=args.seed,
    )

    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)

    # CLIP (frozen)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_enc.eval()
    for p in text_enc.parameters():
        p.requires_grad = False

    projector = CondProjector(512, args.cond_dim).to(device)
    glow = GlowCond(3, args.n_flow, args.n_block, cond_dim=args.cond_dim, affine=args.affine).to(device)

    # Keep it simple: use one DataParallel wrapper
    model = nn.ModuleDict({"glow": glow, "proj": projector}).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    n_bins = 2.0 ** args.n_bits
    z_shapes = calc_z_shapes(3, args.img_size, args.n_block)

    data_iter = iter(dl)
    pbar = tqdm(range(args.iter), dynamic_ncols=True)




    for step in pbar:
        try:
            imgs, caps = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            imgs, caps = next(data_iter)

        imgs = imgs.to(device)

        # quantize + dequantize (same style)
        imgs = imgs * 255.0
        if args.n_bits < 8:
            imgs = torch.floor(imgs / (2 ** (8 - args.n_bits)))
        imgs = imgs / n_bins - 0.5
        imgs = imgs + torch.rand_like(imgs) / n_bins

        # cond from captions
        if isinstance(model, nn.DataParallel):
            cond = make_cond(tokenizer, text_enc, model.module["proj"], caps)
            glow_model = model.module["glow"]
        else:
            cond = make_cond(tokenizer, text_enc, model["proj"], caps)
            glow_model = model["glow"]

        # ActNorm init pass (like your i==0 trick)
        if step == 0:
            with torch.no_grad():
                _ = glow_model(imgs, cond)
            continue

        log_p, logdet, _ = glow_model(imgs, cond)
        loss, lp, ld = calc_loss(log_p, logdet, args.img_size, n_bins)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"bpd {loss.item():.4f} | logP {lp.item():.4f} | logdet {ld.item():.4f}")

        # sampling preview
        if (step + 1) % args.sample_every == 0:
            out_path = os.path.join(args.outdir, "samples", f"{step+1:06d}.png")
            if isinstance(model, nn.DataParallel):
                sample_one(model.module["glow"], tokenizer, text_enc, model.module["proj"],
                           args.prompt_preview, z_shapes, args.temp, out_path)
            else:
                sample_one(model["glow"], tokenizer, text_enc, model["proj"],
                           args.prompt_preview, z_shapes, args.temp, out_path)

        # checkpoint
        if (step + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.outdir, "checkpoints", f"model_{step+1:06d}.pt")
            torch.save(model.state_dict(), ckpt_path)

    print("Training done.")


if __name__ == "__main__":
    main()
