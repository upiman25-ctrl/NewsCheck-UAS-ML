
import joblib
from pathlib import Path

model_path = Path(__file__).resolve().parent / "model" / "newscheck_pipeline.joblib"

model = joblib.load(model_path)

def predict_news(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Judul berita tidak boleh kosong.")

    hasil = int(model.predict([text.strip()])[0])

    if hasil == 0:
        return "FAKE"
    return "REAL"


if __name__ == "__main__":
    judul = input("Masukkan judul berita: ")
    print("Hasil prediksi:", predict_news(judul))
