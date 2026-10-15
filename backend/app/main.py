import base64
import io
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from torchvision import models, transforms
from facenet_pytorch import MTCNN, InceptionResnetV1

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD_FACE = 0.7
MODEL_SPOOF_PATH = "models/antispoofing_model.pth"
DATABASE_PATH = "database/face_db.pt"

app = FastAPI()

# 1. Konfigurasi CORS agar React bisa akses
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Inisialisasi Model (Hanya sekali saat server jalan)
print(f"Loading models on {DEVICE}...")

# Anti-Spoofing Model
def get_spoof_model():
    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(MODEL_SPOOF_PATH, map_location=DEVICE))
    return model.eval().to(DEVICE)

spoof_model = get_spoof_model()
spoof_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Face Recognition Models
# Tambahkan threshold agar hanya wajah yang "sangat jelas" yang terdeteksi
# Threshold diturunkan agar lebih mudah mendeteksi wajah di cahaya redup
mtcnn = MTCNN(keep_all=False, device=DEVICE, thresholds=[0.5, 0.6, 0.6])
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
face_db = torch.load(DATABASE_PATH)

# 3. Model Request (Schema)
class ImageRequest(BaseModel):
    image: str # Ini akan menerima string Base64 dari React

# 4. API Endpoint
@app.post("/verify")
async def verify_face(request: ImageRequest):
    try:
        header, encoded = request.image.split(",", 1)
        data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(data)).convert("RGB")

        # Deteksi Wajah
        boxes, _ = mtcnn.detect(image)
        
        # Hitung rata-rata kecerahan gambar
        grayscale = image.convert("L")
        stat = np.array(grayscale).mean()
        
        if stat < 30: # Jika rata-rata pixel di bawah 30 (sangat gelap)
            return {"status": "too_dark", "name": "Terlalu Gelap!", "is_real": False}

        # JIKA TIDAK ADA WAJAH
        if boxes is None:
            return {
                "status": "no_face", 
                "is_real": False, 
                "name": "-", 
                "box": []
            }
        
        # Ambil kotak pertama
        x1, y1, x2, y2 = boxes[0].astype(int)
        face_crop = image.crop((x1, y1, x2, y2))

        # C. Cek Anti-Spoofing (MobileNetV3)
        # Gunakan face_crop agar anti-spoofing fokus pada tekstur wajah/layar
        face_spoof = spoof_transform(face_crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = spoof_model(face_spoof)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            print(f"DEBUG - Skor Real vs Spoof: {probs[0].tolist()}")
            conf, pred = torch.max(probs, 1)
            
        # LOGIKA BARU (Mapping yang benar: 0=Real, 1=Spoof):
        val_real = probs[0][0].item()  # Probabilitas Real (Index 0)
        val_spoof = probs[0][1].item() # Probabilitas Spoof (Index 1)

        if val_real > 0.85:
            is_real = True
            status_msg = "success"
        elif val_spoof > 0.85:
            is_real = False
            status_msg = "spoof"
        else:
            # Jika AI bingung (karena gelap atau noise), jangan tuduh spoofing!
            return {
                "status": "uncertain", 
                "is_real": False, 
                "name": "Pencahayaan Kurang", 
                "box": [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            }

        # Cek Identitas (Hanya jika Real)
        face_tensor = mtcnn(image) # Gunakan image asli karena MTCNN lebih akurat dengan konteks
        if face_tensor is None:
             return {
                "status": "error", 
                "message": "Wajah tidak dapat diproses untuk rekognisi.", 
                "box": [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            }
        
        with torch.no_grad():
            input_emb = facenet(face_tensor.unsqueeze(0).to(DEVICE)).cpu()

        best_match = "Unknown"
        min_dist = 99.0
        for nim, reg_emb in face_db.items():
            dist = torch.dist(input_emb, reg_emb).item()
            if dist < min_dist:
                min_dist = dist
                best_match = nim

        name_result = best_match if min_dist < THRESHOLD_FACE else "Unknown"

        return {
            "status": "success",
            "is_real": True,
            "name": name_result,
            "box": [int(x1), int(y1), int(x2-x1), int(y2-y1)] # [x, y, width, height]
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)