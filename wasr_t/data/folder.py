from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

class FolderDataset(torch.utils.data.Dataset):
    """Dataset wrapper for a general directory of images."""

    def __init__(self, image_dir, normalize_t=None, resize=None):
        """Creates the dataset.

        Args:
            image_dir (str): path to the image directory. Can contain arbitrary subdirectory structures.
            normalize_t (callable, optional): Transform used to normalize the images. Defaults to None.
            resize (tuple, optional): Resize the input images to this size. No resizing if `None`.
        """

        self.image_dir = Path(image_dir)
        self.images = sorted([p.relative_to(image_dir) for p in Path(image_dir).glob('**/*.jpg')])

        self.normalize_t = normalize_t
        self.resize = resize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rel_path = self.images[idx]
        img_path = self.image_dir / rel_path

        # Load as PIL image to access .size (W, H)
        img_pil = Image.open(str(img_path)).convert("RGB")
        original_size = img_pil.size  # (W, H)

        # Resize with PIL (if needed)
        if self.resize is not None:
            img_pil = img_pil.resize(self.resize, resample=Image.BILINEAR)

        # Convert to numpy and tensor
        img_np = np.array(img_pil)

        if self.normalize_t is not None:
            img_tensor = self.normalize_t(img_np)
        else:
            img_tensor = TF.to_tensor(img_np)

        features = {
            'image': img_tensor,
            'image_original': TF.to_tensor(img_pil),  # still resized, for overlay
            'original_size': torch.tensor(original_size, dtype=torch.int32)  # (W, H)
        }

        metadata = {
            'image_name': img_path.name,
            'image_path': str(rel_path)
        }

        return features, metadata

