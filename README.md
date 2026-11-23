# 📰 NewsCheck

Aplikasi web sederhana untuk memprediksi apakah sebuah **judul berita berbahasa Inggris** termasuk **FAKE** atau **REAL** menggunakan model Machine Learning (TF-IDF + Linear SVM).

Proyek UAS Machine Learning — topik *Text Classification: Fake vs Real News*.

> ⚠️ NewsCheck **tidak memverifikasi fakta**. Hasilnya hanya prediksi berdasarkan pola kata pada data latih, sehingga bisa salah. Tetap cek kebenaran berita dari sumber terpercaya.

## Fitur

- Form untuk memasukkan judul berita berbahasa Inggris, dengan tombol **Check News**.
- Hasil prediksi **FAKE** atau **REAL** beserta penjelasan singkat.
- Validasi input:
  - judul kosong ditolak;
  - judul yang kata-katanya tidak dikenali model sama sekali ditolak;
  - judul yang kurang dari setengah kata pentingnya dikenali model (misalnya judul berbahasa Indonesia) tetap diprediksi, tetapi diberi peringatan bahwa hasilnya kurang bisa diandalkan.
- Informasi model: metode, akurasi, cara kerja klasifikasi, dan batasan model.

## Teknologi

Python · Streamlit · scikit-learn (TF-IDF + LinearSVC) · Joblib

## Struktur proyek

```text
NewsCheck-UAS-ML/
├── .streamlit/
│   └── config.toml              # Tema tampilan web
├── model/
│   └── newscheck_pipeline.joblib  # Model yang sudah dilatih
├── notebook/
│   └── training_model.ipynb     # Proses pelatihan & evaluasi model
├── tests/
│   └── test_app.py              # Pengujian fungsi antarmuka web
├── app.py                       # Aplikasi web Streamlit
├── predict.py                   # Fungsi predict_news() untuk prediksi
└── requirements.txt             # Daftar library
```

## Cara menjalankan

Sudah dites dengan Python 3.14 di Windows.

1. Clone repository lalu masuk ke foldernya:

   ```bash
   git clone https://github.com/upiman25-ctrl/NewsCheck-UAS-ML.git
   cd NewsCheck-UAS-ML
   ```

2. Buat dan aktifkan virtual environment:

   ```powershell
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   ```bash
   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install library:

   ```bash
   pip install -r requirements.txt
   ```

4. Jalankan aplikasi:

   ```bash
   streamlit run app.py
   ```

   Browser akan terbuka otomatis. Jika tidak, buka **http://localhost:8501**. Tekan `Ctrl + C` di terminal untuk menghentikan aplikasi.

## Cara menguji

Pengujian fungsi antarmuka dijalankan tanpa membuka browser menggunakan Streamlit AppTest:

```bash
python -m unittest discover tests -v
```

Yang diuji (14 tes):

- halaman terbuka tanpa error, berisi satu input judul dan tombol **Check News**;
- validasi input kosong dan judul yang tidak dikenali model;
- peringatan untuk judul yang sebagian besar katanya tidak dikenali (misalnya judul berbahasa Indonesia);
- hasil FAKE/REAL di web sama dengan keluaran `predict_news()`;
- peringatan "bukan verifikasi fakta" selalu muncul bersama hasil;
- informasi model (akurasi 77,60%) tampil;
- input berisi HTML tidak merusak halaman;
- pesan error yang jelas jika model gagal dimuat atau prediksi gagal.

## Informasi model

| Item | Keterangan |
|---|---|
| Input | Judul (headline) berita berbahasa Inggris |
| Ekstraksi fitur | TF-IDF (huruf kecil, stop words bahasa Inggris dibuang, maks. 10.000 fitur) |
| Klasifikasi | Linear Support Vector Machine (LinearSVC) |
| Dataset | 2.831 judul unik setelah preprocessing |
| Pembagian data | 80% latih (2.264) · 20% uji (567), *stratified*, `random_state=42` |
| Label | `0` = FAKE · `1` = REAL |

Hasil evaluasi pada data uji:

| Metrik | Nilai |
|---|---|
| Accuracy | 77,60% |
| Precision (weighted) | 77,52% |
| Recall (weighted) | 77,60% |
| F1-Score (weighted) | 77,47% |

Contoh memakai model langsung dari Python:

```python
from predict import predict_news

print(predict_news("Senate passes bill to fund infrastructure projects"))  # REAL / FAKE
```

## Batasan

- Model dirancang untuk **judul berita berbahasa Inggris** dan hanya membaca judul, bukan isi berita.
- Model hanya mengenali pola kata dari data latih, sehingga **tidak bisa memastikan** sebuah berita benar atau salah.
- Dengan akurasi 77,60%, sekitar 22% prediksi pada data uji masih salah.

## Pembagian tugas

| Anggota | Peran | Tugas |
|---|---|---|
| Luthfi | Machine Learning | Menyiapkan dan preprocessing dataset, melatih model SVM dengan TF-IDF, mengevaluasi performa model, menyiapkan model untuk diintegrasikan ke website. |
| Rifqi | Website & UI/UX | Membuat website menggunakan Streamlit, mendesain halaman input dan hasil prediksi, mengintegrasikan model Machine Learning ke website, memastikan website berfungsi dengan baik. |
