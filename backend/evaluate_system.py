import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import numpy as np

# --- KONFIGURASI ---
THRESHOLD = 0.7  # Ambang batas jarak (Semakin kecil semakin ketat)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Model & Database
mtcnn = MTCNN(image_size=160, margin=0, device=DEVICE)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
face_db = torch.load("database/face_db.pt") 

def get_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    face_tensor = mtcnn(img)
    if face_tensor is not None:
        with torch.no_grad():
            return facenet(face_tensor.unsqueeze(0).to(DEVICE))
    return None

def calculate_distance(emb1, emb2):
    return torch.dist(emb1.to(DEVICE), emb2.to(DEVICE)).item()

def run_test(folder_path, expected_auth=True):
    correct = 0
    total = 0
    distances = []
    
    print(f"Menguji folder: {folder_path}...")
    
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".png")):
            total += 1
            emb_input = get_embedding(os.path.join(folder_path, filename))
            
            if emb_input is None:
                continue
            
            # Bandingkan dengan semua data terdaftar (Mencari jarak terkecil)
            min_dist = min([calculate_distance(emb_input, emb_reg) for emb_reg in face_db.values()])
            distances.append(min_dist)
            
            is_authorized = min_dist < THRESHOLD
            if is_authorized == expected_auth:
                correct += 1
                
    return correct, total, distances

# --- EKSEKUSI PENGUJIAN ---
# 1. Tes Wajah Anda (Authorized)
tp, total_me, dist_me = run_test("data/test_me/", expected_auth=True)

# 2. Tes Orang Asing (Unauthorized)
tn, total_un, dist_un = run_test("data/test_unknown/", expected_auth=False)

# --- PERHITUNGAN METRIK ---
accuracy = (tp + tn) / (total_me + total_un) * 100
far = ((total_un - tn) / total_un) * 100  # False Acceptance
frr = ((total_me - tp) / total_me) * 100  # False Rejection

print("\n" + "="*40)
print("HASIL EVALUASI SISTEM BIOMETRIK")
print("="*40)
print(f"Threshold yang digunakan : {THRESHOLD}")
print(f"Total Sampel Uji         : {total_me + total_un}")
print("-" * 40)
print(f"Akurasi Total            : {accuracy:.2f}%")
print(f"FAR (Salah Terima)       : {far:.2f}%")
print(f"FRR (Salah Tolak)        : {frr:.2f}%")
print(f"Rata-rata Jarak (Anda)   : {np.mean(dist_me):.4f}")
print(f"Rata-rata Jarak (Asing)  : {np.mean(dist_un):.4f}")
print("="*40)