# Training script
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loader import DeHaDoDataset, collate_fn, char_to_idx
from model import CNN_BiLSTM_CTC
# ==== Configuration ====

image_dir = r"ADD DATASET IMAGES FILE PATH" #DATASET IMAGES 
label_dir = r"ADD DATASET LABELS FILE PATH" #DATASET LABELS
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

checkpoint_path = os.path.join(model_dir, "checkpoint.pth")
best_model_path = os.path.join(model_dir, "best_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==== Initialize model, optimizer, loss ====
num_classes = len(char_to_idx)
model = CNN_BiLSTM_CTC(num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None

# ==== Resume from checkpoint if exists ====
start_epoch = 1
best_loss = float('inf')

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint.get("best_loss", float('inf'))
    print(f"🔁 Resumed from epoch {checkpoint['epoch']} (best_loss={best_loss:.4f})")

# ==== Data loader ====
dataset = DeHaDoDataset(image_dir=image_dir, label_dir=label_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# ==== Training ====
num_epochs = 50

for epoch in range(start_epoch, num_epochs + 1):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, targets, target_lengths, _) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                seq_len = outputs.size(0)
                input_lengths = torch.full((images.size(0),), seq_len, dtype=torch.long).to(device)
                loss = ctc_loss(outputs, targets, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            seq_len = outputs.size(0)
            input_lengths = torch.full((images.size(0),), seq_len, dtype=torch.long).to(device)
            loss = ctc_loss(outputs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)
    print(f"📘 Epoch {epoch} finished — Avg Loss: {avg_loss:.4f}")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Best model saved: {best_model_path} (loss: {best_loss:.4f})")

    # Save checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss
    }, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")
