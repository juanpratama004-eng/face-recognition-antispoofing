# 🛡️ Sistem Verifikasi Biometrik Wajah dengan Anti-Spoofing

Repositori ini berisi implementasi kode untuk Tugas Akhir (Skripsi): **"Development of a Student Facial Biometric Verification System Based on Transfer Learning and Anti-Spoofing"**. 

Sistem ini merupakan aplikasi *full-stack* yang mengintegrasikan pengenalan identitas wajah (*Face Recognition*) dengan deteksi keaslian objek (*Liveness Detection / Anti-Spoofing*) secara *real-time* untuk mencegah penipuan menggunakan foto cetak maupun layar gawai.

## ✨ Fitur Utama
*   **Deteksi Wajah Akurat**: Melokalisasi wajah secara langsung menggunakan **MTCNN**.
*   **Liveness Detection (Anti-Spoofing)**: Mencegah serangan *spoofing* (layar HP/foto) menggunakan model klasifikasi biner berbasis **MobileNetV3-Small** (Akurasi Uji: 99.00%).
*   **Verifikasi Identitas**: Mencocokkan wajah mahasiswa dengan basis data lokal menggunakan **FaceNet (InceptionResnetV1)** berdasarkan Jarak Euclidean (*Threshold* 0.7, Akurasi: 100%).
*   **Real-time Web UI**: Antarmuka interaktif yang merender *bounding box* dan status kehadiran secara dinamis menggunakan **React.js**.
*   **Decoupled Architecture**: Pemisahan beban kerja antara antarmuka klien (Vite/React) dan *backend* komputasi AI (FastAPI/PyTorch).

## 🛠️ Teknologi yang Digunakan
*   **Bahasa Pemrograman**: Python 3.10, JavaScript (ES6+)
*   **Kecerdasan Buatan**: PyTorch, Torchvision, OpenCV, Facenet-PyTorch
*   **Backend API**: FastAPI, Uvicorn
*   **Frontend UI**: React.js, Vite, Axios, React-Webcam
*   **Akselerasi Perangkat Keras**: NVIDIA CUDA (Diuji pada RTX 3050)

## 📂 Struktur Proyek
Proyek ini dibagi menjadi dua direktori utama:
*   `/backend`: Berisi logika AI, pemuatan model `.pth` / `.pt`, skrip *training*, manajemen *dataset*, dan server FastAPI.
*   `/frontend`: Berisi antarmuka pengguna berbasis React untuk menangkap *stream* dari webcam dan berkomunikasi dengan server melalui format *Base64*.

## 🚀 Panduan Instalasi dan Penggunaan

### Persyaratan Sistem
*   Python 3.10+
*   Node.js v16+ & npm
*   (Opsional namun disarankan) GPU NVIDIA yang mendukung CUDA untuk inferensi *real-time*.

### 1. Kloning Repositori
```bash
git clone https://github.com/juanpratama004-eng/face-recognition-antispoofing.git
cd face-recognition-antispoofing
```

### 2. Pengaturan Backend (Server AI)
Buka terminal baru dan navigasikan ke folder `backend`.
```bash
cd backend

# Buat dan aktifkan Virtual Environment
python -m venv venv
# Untuk Windows:
.\venv\Scripts\activate
# Untuk Linux/Mac:
source venv/bin/activate

# Instalasi dependensi
pip install -r requirements.txt
```
> **Catatan:** Pastikan model `antispoofing_model.pth` dan database wajah `face_db.pt` sudah diletakkan di dalam folder `/backend/models` dan `/backend/database`.

**Menjalankan Backend:**
```bash
python app/main.py
```
Server backend akan berjalan di `http://0.0.0.0:8000`.

### 3. Pengaturan Frontend (Antarmuka Web)
Buka terminal terpisah, lalu navigasikan ke folder `frontend`.
```bash
cd frontend

# Instalasi dependensi Node.js
npm install
```

**Menjalankan Frontend:**
```bash
npm run dev
```
Aplikasi web akan terbuka di peramban pada alamat `http://localhost:5173`. 

## 🧪 Skrip Evaluasi dan Pelatihan (Backend)
Repositori ini juga menyertakan skrip pendukung di dalam folder `backend` untuk kebutuhan pengujian akademik:
*   `train_antispoof.py`: Untuk melatih ulang model deteksi *liveness* dengan teknik augmentasi `ColorJitter` dan `RandomRotation`.
*   `evaluate_system.py`: Untuk menguji akurasi keseluruhan sistem (menghitung persentase FAR dan FRR).
*   `register_faces.py`: Untuk mendaftarkan *embedding* wajah mahasiswa baru ke dalam *database*.

## 👨‍💻 Penulis
**Juan Pratama**  
Program Studi Informatika  
Universitas Pembangunan Jaya

---
*Dibuat untuk memenuhi persyaratan Tugas Akhir Program Sarjana (S1).*