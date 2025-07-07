# Dataset loader + char mappings
import os
import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import string
import argparse

# === CHAR DICTIONARY ===
alphabet = string.digits + string.ascii_lowercase + ".,-'/ "
char_to_idx = {c: i + 1 for i, c in enumerate(alphabet)}  # Start from 1
char_to_idx['<blank>'] = 0  # For CTC
idx_to_char = {i: c for c, i in char_to_idx.items()}

# === LABEL ENCODING ===
def encode_label(text):
    text = text.lower()
    return [char_to_idx[c] for c in text if c in char_to_idx]

# === COLLATE FUNCTION ===
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    encoded_labels = [torch.tensor(encode_label(label), dtype=torch.long) for label in labels]
    targets = torch.cat(encoded_labels)
    target_lengths = torch.tensor([len(l) for l in encoded_labels], dtype=torch.long)
    return images, targets, target_lengths, labels

# === AUGMENTED DATASET ===
class DeHaDoDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, max_samples=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.samples = []
        self.max_samples = max_samples
        self._load_samples()
        self.transform = transform or self._get_transform()

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.RandomApply([
                transforms.GaussianBlur(3),
                transforms.RandomRotation(degrees=2),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _load_samples(self):
        for img_file in self.image_files:
            img_path = os.path.join(self.image_dir, img_file)
            json_file = img_file.replace('.jpg', '.json')
            json_path = os.path.join(self.label_dir, json_file)

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    fields = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error reading JSON: {json_path}")
                print(e)
                continue

            image = cv2.imread(img_path)
            if image is None:
                print(f"Image not found: {img_path}")
                continue

            for field in fields:
                label = field["Field value"].strip()
                x1, y1, x2, y2 = field["Coordinate"]
                crop = image[y1:y2, x1:x2]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(crop)
                self.samples.append((crop_pil, label))

                if self.max_samples and len(self.samples) >= self.max_samples:
                    return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self.samples[idx]
        image_tensor = self.transform(image)
        return image_tensor, label

# === QUICK TEST ===
if __name__ == "__main__":
    image_dir = r"ADD DATASET IMAGES FILE PATH" #DATASET IMAGES 
    label_dir = r"ADD DATASET LABELS FILE PATH" #DATASET LABELS

    dataset = DeHaDoDataset(image_dir=image_dir, label_dir=label_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    for batch_idx, (images, targets, target_lengths, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}")
        print(f"Images shape: {images.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Target lengths: {target_lengths}")
        print(f"Sample labels: {labels[:3]}")
        break
