import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

# --- KONFIGURASI ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD_FACE = 0.7
MODEL_SPOOF_PATH = "models/antispoofing_model.pth"

# 1. Load Model Anti-Spoofing (MobileNetV3)
def load_spoof_model(path):
    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

spoof_model = load_spoof_model(MODEL_SPOOF_PATH)

# Transformasi untuk Anti-Spoofing
spoof_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Load Model FaceNet
mtcnn = MTCNN(keep_all=True, device=DEVICE)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
face_db = torch.load("database/face_db.pt")

# 3. Inisialisasi Kamera
cap = cv2.VideoCapture(1)

print("Sistem Keamanan Aktif. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret: break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    boxes, _ = mtcnn.detect(img_pil)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            face = img_pil.crop((x1, y1, x2, y2))
            
            # --- TAHAP 1: CEK ANTI-SPOOFING ---
            face_spoof = spoof_transform(face).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = spoof_model(face_spoof)
                # Index 0 = Real, Index 1 = Spoof (berdasarkan folder dataset kita)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, pred = torch.max(probabilities, 1)
                is_real = (pred.item() == 0) and (confidence.item() > 0.90)

            if is_real:
                # --- TAHAP 2: CEK IDENTITAS (Jika Lulus Anti-Spoof) ---
                face_tensor = mtcnn(face)
                if face_tensor is not None:
                    if face_tensor.ndim == 3:
                        face_tensor = face_tensor.unsqueeze(0)
                    with torch.no_grad():
                        input_emb = facenet(face_tensor.to(DEVICE)).cpu()

                    best_match = "Unknown"
                    min_dist = 99.0
                    for nim, reg_emb in face_db.items():
                        dist = torch.dist(input_emb, reg_emb).item()
                        if dist < min_dist:
                            min_dist = dist
                            best_match = nim

                    if min_dist < THRESHOLD_FACE:
                        label = f"AUTHORIZED: {best_match}"
                        color = (0, 255, 0) # Hijau
                    else:
                        label = "UNAUTHORIZED USER"
                        color = (0, 0, 255) # Merah
            else:
                # --- JIKA TERDETEKSI FOTO/LAYAR ---
                label = "SPOOFING DETECTED!"
                color = (0, 165, 255) # Oranye (Peringatan)

            # Gambar Box dan Label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Biometric Security - TA Juan', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()