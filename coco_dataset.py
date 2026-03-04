import os
import random
from PIL import Image

from torch.utils.data import Dataset
from pycocotools.coco import COCO


class CocoCaptionsSimple(Dataset):
    """
    Minimal COCO captions dataset:
      - returns (image_tensor, caption_string)
      - uses captions_train2017.json

    Notes:
      - One image has multiple captions; we randomly pick one each __getitem__.
    """
    def __init__(self, images_dir, captions_json, transform=None, max_samples=None, seed=0):
        self.images_dir = images_dir
        self.coco = COCO(captions_json)
        self.transform = transform
        self.rng = random.Random(seed)

        # image ids that have captions
        self.img_ids = list(self.coco.imgs.keys())

        if max_samples is not None and max_samples > 0:
            self.rng.shuffle(self.img_ids)
            self.img_ids = self.img_ids[:max_samples]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        file_name = img_info["file_name"]
        img_path = os.path.join(self.images_dir, file_name)

        # pick a random caption for this image
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        caption = self.rng.choice(anns)["caption"] if len(anns) > 0 else ""

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, caption
