import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os

# --- KONFIGURASI ---
BATCH_SIZE = 32
EPOCHS = 15 # Ditambah sedikit agar stabil
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "data/antispoof_dataset"
MODEL_SAVE_PATH = "models/antispoofing_model.pth"

# 1. Transformasi Data (Tetap menggunakan augmentasi untuk train)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    # MENAMBAHKAN VARIASI KECERAHAN SECARA ACAK
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), 
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Load & Split Dataset
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)

# Hitung jumlah untuk split 80/20
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Menggunakan Device: {DEVICE}")
print(f"Total Data: {len(full_dataset)}")
print(f"Data Latihan: {train_size} | Data Validasi: {val_size}")

# 3. Load Model MobileNetV3-Small
model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_ftrs, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4. Loop Training & Validation
print("\n--- Memulai Latihan ---")
for epoch in range(EPOCHS):
    # FASE LATIHAN
    model.train()
    train_loss, train_correct = 0.0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_correct += predicted.eq(labels).sum().item()

    # FASE VALIDASI (Ujian)
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()

    train_acc = 100 * train_correct / train_size
    val_acc = 100 * val_correct / val_size
    
    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"  Train Loss: {train_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss/len(val_loader):.4f} | Acc: {val_acc:.2f}%")

# 5. Simpan Model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nModel Baru Disimpan: {MODEL_SAVE_PATH}")