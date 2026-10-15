import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import os

# Inisialisasi Device GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Model
mtcnn = MTCNN(image_size=160, margin=0, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def register_students(data_path="data/registered/", db_path="database/face_db.pt"):
    face_db = {}
    
    print(f"Memulai registrasi wajah dari {data_path}...")
    
    for filename in os.listdir(data_path):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            nim = os.path.splitext(filename)[0] # Ambil NIM dari nama file
            img_path = os.path.join(data_path, filename)
            
            img = Image.open(img_path).convert('RGB')
            
            # 1. Deteksi & Crop Wajah
            face_tensor = mtcnn(img)
            
            if face_tensor is not None:
                # 2. Ekstraksi Embedding
                with torch.no_grad():
                    embedding = facenet(face_tensor.unsqueeze(0).to(device))
                
                face_db[nim] = embedding.cpu() # Simpan embedding berdasarkan NIM
                print(f"Berhasil meregistrasi NIM: {nim}")
            else:
                print(f"Gagal mendeteksi wajah pada file: {filename}")

    # Simpan database ke file lokal
    if not os.path.exists("database"): os.makedirs("database")
    torch.save(face_db, db_path)
    print(f"Pendaftaran selesai. {len(face_db)} wajah tersimpan di {db_path}.")

if __name__ == "__main__":
    register_students()