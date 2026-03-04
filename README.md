# Text-Conditional Glow (CLIP + FiLM) — COCO Captions

This project trains a **Glow** normalizing flow model to generate images **conditioned on text prompts** using:
- **CLIP text encoder** (frozen)
- **FiLM conditioning** injected into Glow **coupling layers** and **priors**

> Note: Some files you uploaded earlier in chat have expired on my side, so this README is based on the minimal codebase we discussed: `model_cond.py`, `coco_dataset.py`, `train_cond_coco.py`, `sample_text.py`.

---

## Folder Structure (Simple / Root-Level)

glow_text_prompt/
├── requirements.txt
├── model_cond.py
├── coco_dataset.py
├── train_cond_coco.py
├── sample_text.py
│
├── data/
│   └── coco2017/
│       ├── train2017/
│       │   ├── 000000000009.jpg
│       │   ├── 000000000025.jpg
│       │   └── ... (all COCO 2017 train images)
│       │
│       └── annotations/
│           └── captions_train2017.json
│
├── runs_textglow/                 # auto-created by training
│   ├── checkpoints/
│   │   ├── model_010000.pt
│   │   ├── model_020000.pt
│   │   └── ...
│   └── samples/
│       ├── 000500.png
│       ├── 001000.png
│       └── ...
│
└── samples_text/                  # created by sampling
    ├── sample_000.png             # if --single
    └── samples.png                # if grid

---

## Dataset Download (COCO 2017)

You need:
- **COCO 2017 Train images** (`train2017.zip`)
- **COCO 2017 annotations** (`annotations_trainval2017.zip`) which contains `captions_train2017.json`

Official download page:
- https://cocodataset.org/#download

Direct links (from the official page):
- 2017 Train images: https://images.cocodataset.org/zips/train2017.zip
- 2017 Train/Val annotations: https://images.cocodataset.org/annotations/annotations_trainval2017.zip

### Extract & place like this
After extraction:

- Put images in:
  `data/coco2017/train2017/`
- Put captions json in:
  `data/coco2017/annotations/captions_train2017.json`

Expected paths:

data/coco2017/train2017/000000000009.jpg  
data/coco2017/annotations/captions_train2017.json  

---

## Setup

### 1) Create venv + install dependencies

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2) Transformers / PyTorch compatibility fix (IMPORTANT)

If you see:

> Disabling PyTorch because PyTorch >= 2.4 is required but found 2.2.x  
> CLIPTextModel requires the PyTorch library but it was not found...

Pin Transformers to a version compatible with Torch 2.2.x:

```bash
pip uninstall -y transformers tokenizers
pip install --no-cache-dir "transformers==4.39.3" "tokenizers==0.15.2"
```

Verify:

```bash
python -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__)"
python -c "from transformers import CLIPTextModel, CLIPTokenizer; print('CLIP OK')"
```

---

## Train (COCO Captions → Conditional Glow)

Full training:

```bash
python train_cond_coco.py \
  --coco_images data/coco2017/train2017 \
  --coco_captions data/coco2017/annotations/captions_train2017.json \
  --n_flow 32 --n_block 4 --img_size 64 --n_bits 5 \
  --cond_dim 256 --affine \
  --batch 8 --iter 200000 \
  --outdir runs_textglow \
  --prompt_preview "a cat"
```

Quick test (smaller dataset + fewer iterations):

```bash
python train_cond_coco.py \
  --coco_images data/coco2017/train2017 \
  --coco_captions data/coco2017/annotations/captions_train2017.json \
  --max_samples 5000 \
  --iter 20000 \
  --batch 8 \
  --outdir runs_textglow_test \
  --prompt_preview "a cat"
```

Training outputs:
- checkpoints: `runs_textglow/checkpoints/model_XXXXXX.pt`
- sample previews: `runs_textglow/samples/XXXXXX.png`

---

## Generate Images (Text Prompt → Image)

Single image:

```bash
python sample_text.py \
  --ckpt runs_textglow/checkpoints \
  --prompt "white cat with a fish" \
  --n_flow 32 --n_block 4 --img_size 64 --cond_dim 256 --affine \
  --n_sample 1 --temp 0.7 \
  --single \
  --out samples_text
```

Grid of samples:

```bash
python sample_text.py \
  --ckpt runs_textglow/checkpoints \
  --prompt "a dog running on grass" \
  --n_flow 32 --n_block 4 --img_size 64 --cond_dim 256 --affine \
  --n_sample 9 --temp 0.7 \
  --out samples_text
```

---

## Notes / Expectations

- This is **true conditional training**: it learns **p(x | text)** from COCO captions.
- At **64×64**, prompt alignment is usually **coarse** (object/category level).
- Highly specific prompts like *"white cat with a fish"* may be hit-or-miss depending on the dataset distribution.

---