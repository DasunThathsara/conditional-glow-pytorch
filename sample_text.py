import os
import re
import math
import argparse
import torch
from torchvision import utils

from transformers import CLIPTokenizer, CLIPTextModel

from model_cond import GlowCond
from train_cond_coco import CondProjector, calc_z_shapes


def infer_latest_checkpoint(path: str) -> str:
    if os.path.isfile(path):
        return path
    files = os.listdir(path)
    cands = [f for f in files if re.match(r"model_\d+\.pt$", f)]
    if not cands:
        raise FileNotFoundError(f"No model_*.pt found in: {path}")
    def stepnum(s: str) -> int:
        m = re.findall(r"\d+", s)
        return int(m[-1]) if m else -1
    best = max(cands, key=stepnum)
    return os.path.join(path, best)


def load_ckpt_into(glow, proj, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")

    # The training may save keys like:
    # - "glow.blocks...." or "module.glow.blocks...."
    # - same for proj
    glow_sd = {}
    proj_sd = {}

    for k, v in sd.items():
        kk = k.replace("module.", "")
        if kk.startswith("glow."):
            glow_sd[kk.replace("glow.", "")] = v
        elif kk.startswith("proj."):
            proj_sd[kk.replace("proj.", "")] = v

    glow.load_state_dict(glow_sd, strict=False)
    proj.load_state_dict(proj_sd, strict=False)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("Sample from text-conditional Glow")
    ap.add_argument("--ckpt", type=str, required=True, help="runs_textglow/checkpoints OR model_xxxxxx.pt")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--n_sample", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.7)

    ap.add_argument("--n_flow", type=int, default=32)
    ap.add_argument("--n_block", type=int, default=4)
    ap.add_argument("--cond_dim", type=int, default=256)
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--affine", action="store_true")

    ap.add_argument("--out", type=str, default="samples_text")
    ap.add_argument("--single", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CLIP (frozen)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_enc.eval()
    for p in text_enc.parameters():
        p.requires_grad = False

    proj = CondProjector(512, args.cond_dim).to(device)
    glow = GlowCond(3, args.n_flow, args.n_block, cond_dim=args.cond_dim, affine=args.affine).to(device)
    glow.eval()
    proj.eval()

    ckpt_path = infer_latest_checkpoint(args.ckpt)
    print("Loading:", ckpt_path)
    load_ckpt_into(glow, proj, ckpt_path)

    # Prompt -> cond
    tok = tokenizer([args.prompt] * args.n_sample, padding=True, truncation=True, return_tensors="pt")
    tok = {k: v.to(device) for k, v in tok.items()}
    te = text_enc(**tok).pooler_output
    cond = proj(te)

    # eps list
    z_shapes = calc_z_shapes(3, args.img_size, args.n_block)
    eps_list = [torch.randn(args.n_sample, c, h, w, device=device) * args.temp for (c, h, w) in z_shapes]

    x = glow.reverse(eps_list, cond, reconstruct=False).clamp(-0.5, 0.5)

    if args.single:
        for i in range(args.n_sample):
            out_path = os.path.join(args.out, f"sample_{i:03d}.png")
            utils.save_image(x[i:i+1].cpu(), out_path, normalize=True, value_range=(-0.5, 0.5))
        print("Saved singles to:", args.out)
    else:
        nrow = max(1, int(math.isqrt(args.n_sample)))
        out_path = os.path.join(args.out, "samples.png")
        utils.save_image(x.cpu(), out_path, nrow=nrow, normalize=True, value_range=(-0.5, 0.5))
        print("Saved:", out_path)


if __name__ == "__main__":
    main()
