import os
import random
import argparse
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from model import SatelliteRestormerSR, CharbonnierFFTLoss
from dataset import dataset_warning, find_images, inspect_dataset

class SatelliteSRDataset(Dataset):
    def __init__(self, folder_path, hr_size=128, scale=4, max_images_per_class=None):
        paths = find_images(folder_path)
        if max_images_per_class is not None:
            paths_by_class = {}
            for path in paths:
                paths_by_class.setdefault(path.parent, []).append(path)
            rng = random.Random(42)
            self.paths = [
                path
                for class_paths in paths_by_class.values()
                for path in rng.sample(class_paths, min(max_images_per_class, len(class_paths)))
            ]
            self.paths.sort()
        else:
            self.paths = paths
        self.crop = T.RandomCrop(hr_size)
        self.hr_size = hr_size
        self.lr_size = hr_size // scale

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if img.width < self.hr_size or img.height < self.hr_size:
            img = img.resize((self.hr_size, self.hr_size), Image.Resampling.BICUBIC)
        hr_patch = self.crop(img)
        lr_patch = hr_patch.resize((self.lr_size, self.lr_size), Image.Resampling.BICUBIC)
        return TF.to_tensor(lr_patch), TF.to_tensor(hr_patch)

def train(epochs=30, batch_size=8):
    project_root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = project_root / "data" / "train"
    report = inspect_dataset(data_path)
    warning = dataset_warning(report)
    if warning:
        raise RuntimeError(warning)

    dataset = SatelliteSRDataset(
        folder_path=data_path,
        hr_size=128,
        scale=4,
        max_images_per_class=17,
    )
    if len(dataset) == 0:
        raise RuntimeError("Place satellite images inside 'data/train/' before running train.py")
    print(f"Using {len(dataset)} balanced training images.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    model = SatelliteRestormerSR(scale=4).to(device)

    weights_path = project_root / "restormer_weights.pth"
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path} for finetuning...")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        lr = 5e-5  # Reduced learning rate for finetuning
        fft_weight = 0.15 # Increased FFT weight for sharper edges
    else:
        lr = 2e-4
        fft_weight = 0.1

    criterion = CharbonnierFFTLoss(fft_weight=fft_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for lr_imgs, hr_imgs in loader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            optimizer.zero_grad()
            preds = model(lr_imgs)
            loss = criterion(preds, hr_imgs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss / len(loader):.4f}")
        torch.save(model.state_dict(), weights_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the satellite super-resolution model.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size)
