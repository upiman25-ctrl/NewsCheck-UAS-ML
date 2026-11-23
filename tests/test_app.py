"""
Pengujian fungsi antarmuka (UI) NewsCheck menggunakan Streamlit AppTest.

Setiap tes menjalankan app.py tanpa membuka browser, mengisi form judul berita,
menekan tombol "Check News", lalu memeriksa apa yang tampil di halaman.

Cara menjalankan (dari folder utama proyek):
    python -m unittest discover tests -v
"""

import logging
import sys
import unittest
from pathlib import Path
from unittest import mock

from streamlit.testing.v1 import AppTest

# Sembunyikan peringatan "missing ScriptRunContext". Peringatan ini wajar
# saat Streamlit diuji tanpa browser dan tidak menandakan error.
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").addFilter(
    lambda record: "missing ScriptRunContext" not in record.getMessage()
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from predict import predict_news  # noqa: E402

APP = str(ROOT / "app.py")

# Judul bahasa Inggris untuk menguji hasil FAKE dan REAL.
JUDUL_INGGRIS = [
    "Senate passes bill to fund infrastructure projects",
    "Federal Reserve raises interest rates by a quarter point",
    "Obama secretly signs order to ban guns nationwide",
    "Hillary Clinton caught on tape admitting election fraud",
]


# Statistik sumber palsu untuk pengujian, supaya hasil tes tidak bergantung
# pada isi model/domain_stats.json yang asli.
STATISTIK_CONTOH = {
    "jumlah_berita": 1080,
    "domain": {
        "reuters.com": {"fake": 0, "real": 1000},
        "beforeitsnews.com": {"fake": 60, "real": 0},
        "cnn.com": {"fake": 2, "real": 18},
    },
}


def buka_app(statistik=None):
    """Jalankan app.py seperti saat pertama kali dibuka di browser.

    statistik=None mensimulasikan model/domain_stats.json yang belum dibuat.
    """
    with mock.patch("source_stats.muat_statistik", return_value=statistik):
        at = AppTest.from_file(APP, default_timeout=60)
        at.run()
    return at


def cek_berita(judul, link=None, statistik=None):
    """Isi form dengan judul (dan link jika ada), lalu tekan tombol Check News."""
    with mock.patch("source_stats.muat_statistik", return_value=statistik):
        at = AppTest.from_file(APP, default_timeout=60)
        at.run()
        at.text_input(key="judul").input(judul)
        if link is not None:
            at.text_input(key="link").input(link)
        at.button[0].click().run()
    return at


def isi_html(at):
    return " ".join(el.proto.body for el in at.get("html"))


def hasil_tampil(at):
    """Ambil label hasil (FAKE/REAL) dari kotak hasil, atau None jika tidak ada."""
    html = isi_html(at)
    if "nc-result fake" in html:
        return "FAKE"
    if "nc-result real" in html:
        return "REAL"
    return None


def semua_teks(elemen):
    return " ".join(el.value for el in elemen)


class TestHalamanUtama(unittest.TestCase):
    def setUp(self):
        self.at = buka_app()

    def test_halaman_terbuka_tanpa_error(self):
        self.assertFalse(self.at.exception)
        self.assertIn("nc-title", isi_html(self.at))

    def test_form_berisi_satu_input_dan_tombol_check_news(self):
        self.assertEqual(len(self.at.text_input), 1)
        self.assertEqual(self.at.button[0].label, "Check News")

    def test_input_dibatasi_300_karakter(self):
        self.assertEqual(self.at.text_input(key="judul").max_chars, 300)

    def test_belum_ada_hasil_sebelum_tombol_ditekan(self):
        self.assertIsNone(hasil_tampil(self.at))

    def test_informasi_model_ditampilkan(self):
        nilai_metric = [m.value for m in self.at.metric]
        self.assertIn("77,60%", nilai_metric)
        self.assertIn("TF-IDF + Linear Support Vector Machine", semua_teks(self.at.markdown))


class TestValidasiInput(unittest.TestCase):
    def test_judul_kosong_ditolak(self):
        for judul in ["", "   "]:
            with self.subTest(judul=judul):
                at = cek_berita(judul)
                self.assertFalse(at.exception)
                self.assertIn("tidak boleh kosong", semua_teks(at.warning))
                self.assertIsNone(hasil_tampil(at))

    def test_kata_tidak_dikenali_ditolak(self):
        at = cek_berita("zzqx qwrtp")
        self.assertIn("tidak mengenali kata-kata", semua_teks(at.warning))
        self.assertIsNone(hasil_tampil(at))
        self.assertEqual(len(at.info), 0)

    def test_judul_bahasa_indonesia_diberi_peringatan(self):
        judul = "Polisi tangkap pelaku penipuan online di Surabaya"
        at = cek_berita(judul)
        self.assertEqual(hasil_tampil(at), predict_news(judul))
        self.assertIn("kurang bisa diandalkan", semua_teks(at.warning))


class TestHasilPrediksi(unittest.TestCase):
    def test_hasil_sama_dengan_predict_news(self):
        for judul in JUDUL_INGGRIS:
            with self.subTest(judul=judul):
                at = cek_berita(judul)
                self.assertFalse(at.exception)
                self.assertEqual(hasil_tampil(at), predict_news(judul))
                self.assertEqual(len(at.warning), 0)
                self.assertEqual(len(at.error), 0)

    def test_tampilan_fake_dan_real_sama_sama_teruji(self):
        label = {predict_news(judul) for judul in JUDUL_INGGRIS}
        self.assertEqual(label, {"FAKE", "REAL"})

    def test_peringatan_bukan_verifikasi_fakta_selalu_muncul(self):
        at = cek_berita(JUDUL_INGGRIS[0])
        self.assertIn("bukan verifikasi fakta", semua_teks(at.info))

    def test_input_berisi_html_tidak_merusak_halaman(self):
        at = cek_berita("<script>alert(1)</script> breaking news")
        html = isi_html(at)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>alert", html)


# Judul kotak catatan sumber (berbeda dari teks "Catatan sumber" di bagian batasan).
PENANDA_CATATAN = "🔗 Catatan sumber"


class TestCatatanSumber(unittest.TestCase):
    JUDUL = JUDUL_INGGRIS[0]

    def catatan(self, at):
        return semua_teks(at.markdown)

    def test_kolom_link_tidak_tampil_jika_statistik_belum_ada(self):
        at = buka_app(statistik=None)
        self.assertEqual(len(at.text_input), 1)
        self.assertNotIn(PENANDA_CATATAN, semua_teks(at.markdown))

    def test_kolom_link_tampil_jika_statistik_ada(self):
        at = buka_app(statistik=STATISTIK_CONTOH)
        self.assertEqual(len(at.text_input), 2)
        self.assertEqual(at.text_input(key="link").max_chars, 500)
        self.assertIn("tidak dipakai model", semua_teks(at.markdown))

    def test_domain_dikenali(self):
        at = cek_berita(self.JUDUL, "https://www.reuters.com/world/abc", STATISTIK_CONTOH)
        self.assertFalse(at.exception)
        teks = self.catatan(at)
        self.assertIn(PENANDA_CATATAN, teks)
        self.assertIn("**1.000 berita** dari `reuters.com`", teks)
        self.assertIn("**0 berlabel FAKE** (0%)", teks)
        self.assertIn("**1.000 berlabel REAL** (100%)", teks)
        self.assertIn("tidak dipakai model", semua_teks(at.caption))

    def test_subdomain_cocok_ke_domain_induk(self):
        # Link tanpa https:// dan dengan subdomain tetap dikenali.
        at = cek_berita(self.JUDUL, "edition.cnn.com/2017/01/01/x.html", STATISTIK_CONTOH)
        self.assertIn("**20 berita** dari `cnn.com`", self.catatan(at))

    def test_domain_tidak_ada_di_dataset(self):
        at = cek_berita(self.JUDUL, "https://example.org/berita", STATISTIK_CONTOH)
        self.assertIn("`example.org` tidak ada di dataset", self.catatan(at))

    def test_link_tidak_valid_diberi_peringatan(self):
        at = cek_berita(self.JUDUL, "bukan link", STATISTIK_CONTOH)
        self.assertIn("Link berita tidak bisa dibaca", semua_teks(at.warning))
        self.assertEqual(hasil_tampil(at), predict_news(self.JUDUL))

    def test_link_kosong_tidak_menampilkan_catatan(self):
        at = cek_berita(self.JUDUL, "   ", STATISTIK_CONTOH)
        self.assertNotIn(PENANDA_CATATAN, self.catatan(at))
        self.assertEqual(len(at.warning), 0)

    def test_link_tidak_mengubah_prediksi(self):
        # Prediksi harus tetap sama walaupun link dari situs yang di dataset
        # semuanya FAKE atau semuanya REAL.
        for judul in JUDUL_INGGRIS:
            for link in ["https://beforeitsnews.com/x", "https://www.reuters.com/x"]:
                with self.subTest(judul=judul, link=link):
                    at = cek_berita(judul, link, STATISTIK_CONTOH)
                    self.assertEqual(hasil_tampil(at), predict_news(judul))

    def test_catatan_sumber_hanya_muncul_bersama_hasil(self):
        at = cek_berita("", "https://www.reuters.com/x", STATISTIK_CONTOH)
        self.assertIn("tidak boleh kosong", semua_teks(at.warning))
        self.assertNotIn(PENANDA_CATATAN, self.catatan(at))


class TestPesanError(unittest.TestCase):
    def test_pesan_jelas_jika_model_gagal_dimuat(self):
        # Simulasikan predict.py / file model gagal di-import.
        modul_asli = sys.modules.get("predict")
        sys.modules["predict"] = None
        try:
            at = buka_app()
        finally:
            sys.modules["predict"] = modul_asli
        self.assertFalse(at.exception)
        self.assertIn("Model gagal dimuat", semua_teks(at.error))
        self.assertEqual(len(at.text_input), 0)

    def test_pesan_jelas_jika_prediksi_gagal(self):
        # Simulasikan predict_news() melempar error saat dipanggil.
        with mock.patch("predict.predict_news", side_effect=RuntimeError("simulasi")):
            at = cek_berita(JUDUL_INGGRIS[0])
        self.assertFalse(at.exception)
        self.assertIn("Terjadi kesalahan saat memproses prediksi", semua_teks(at.error))
        self.assertIsNone(hasil_tampil(at))


if __name__ == "__main__":
    unittest.main()
