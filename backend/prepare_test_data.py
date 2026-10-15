import os
import random
import shutil

# --- PENGATURAN PATH ---
# Pastikan folder ini ada di dalam folder 'backend'
source_dir = "lfw-deepfunneled" 
target_dir = os.path.join("data", "test_unknown")

print(f"DEBUG: Mencari folder sumber di: {os.path.abspath(source_dir)}")

if not os.path.exists(source_dir):
    print(f"KESALAHAN: Folder '{source_dir}' TIDAK DITEMUKAN!")
    print("Pastikan folder LFW hasil ekstrak ada di dalam folder 'backend'.")
    exit()

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    print(f"DEBUG: Membuat folder tujuan di {target_dir}")

# Ambil semua folder orang
all_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
print(f"DEBUG: Menemukan {len(all_folders)} folder orang di dalam '{source_dir}'.")

if len(all_folders) == 0:
    print("KESALAHAN: Tidak ada folder orang ditemukan. Cek isi folder sumber Anda.")
    exit()

random.shuffle(all_folders)
selected_folders = all_folders[:100]

count = 0
for i, folder_name in enumerate(selected_folders):
    folder_path = os.path.join(source_dir, folder_name)
    photos = [p for p in os.listdir(folder_path) if p.endswith(('.jpg', '.jpeg', '.png'))]
    
    if photos:
        old_path = os.path.join(folder_path, photos[0])
        new_name = f"unknown_{i+1:03d}.jpg"
        new_path = os.path.join(target_dir, new_name)
        
        shutil.copy(old_path, new_path)
        count += 1

print(f"DEBUG: Berhasil menyalin {count} foto ke {target_dir}")
print("="*30)
if count > 0:
    print("PROSES BERHASIL!")
else:
    print("PROSES GAGAL: Tidak ada foto yang disalin.")