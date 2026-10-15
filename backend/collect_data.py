import cv2
import os
import torch
from facenet_pytorch import MTCNN
from PIL import Image

# --- KONFIGURASI ---
label = "spoof"  # Ubah ke "real" untuk wajah asli, "spoof" untuk foto HP
save_path = f"data/antispoof_dataset/{label}"
max_samples = 250 

if not os.path.exists(save_path): os.makedirs(save_path)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=DEVICE)

cap = cv2.VideoCapture(1) # Gunakan 1 jika webcam eksternal, 0 jika internal
count = 0

print(f"Memulai pengambilan data crop wajah untuk {label}...")
print("Pastikan wajah/HP masuk ke kamera dengan jelas.")

while count < max_samples:
    ret, frame = cap.read()
    if not ret: break
    
    # Deteksi dan crop wajah
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    boxes, _ = mtcnn.detect(image_pil)
    
    # Hanya simpan Murni CROP WAJAH
    if boxes is not None:
        x1, y1, x2, y2 = boxes[0].astype(int)
        
        # Tambah margin sedikit supaya tidak terlalu ketat
        w, h = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - 0.1 * w))
        y1 = max(0, int(y1 - 0.1 * h))
        x2 = min(image_pil.width, int(x2 + 0.1 * w))
        y2 = min(image_pil.height, int(y2 + 0.1 * h))
        
        face_crop = image_pil.crop((x1, y1, x2, y2))
        
        # Convert kembali ke BGR untuk disimpan OpenCV (atau simpan PIL langsung)
        file_name = f"{save_path}/{label}_{count}.jpg"
        face_crop.save(file_name)
        
        count += 1
        
        # Tampilkan kotak di UI koleksi data
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.putText(frame, f"Captured: {count}/{max_samples}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Data Collection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print(f"Selesai! {count} crop wajah tersimpan di {save_path}")