"""
Pengujian source_stats.py (statistik sumber berita untuk "Catatan sumber").

Tes ini memakai file CSV kecil buatan sendiri, jadi tidak butuh data/data.csv.

Cara menjalankan (dari folder utama proyek):
    python -m unittest discover tests -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import source_stats  # noqa: E402
from source_stats import ambil_domain, cari_domain, hitung_statistik, muat_statistik  # noqa: E402


class TestAmbilDomain(unittest.TestCase):
    def test_link_valid(self):
        kasus = {
            "https://www.reuters.com/world/abc": "reuters.com",
            "http://beforeitsnews.com/x.html?id=1": "beforeitsnews.com",
            "HTTPS://WWW.BBC.CO.UK/news": "bbc.co.uk",
            "edition.cnn.com/2017/01/01/x.html": "edition.cnn.com",
            "  www.nytimes.com  ": "nytimes.com",
            "//example.org/a": "example.org",
            "https://user:pass@example.com:8080/a": "example.com",
            "https://reuters.com./a": "reuters.com",
        }
        for link, domain in kasus.items():
            with self.subTest(link=link):
                self.assertEqual(ambil_domain(link), domain)

    def test_link_tidak_valid(self):
        for link in ["", "   ", "bukan link", "asdf", "http://", "https://[::1", None, 123]:
            with self.subTest(link=link):
                self.assertIsNone(ambil_domain(link))

    def test_www_saja_tidak_dibuang_jika_jadi_tidak_valid(self):
        self.assertEqual(ambil_domain("www.com"), "www.com")


class TestCariDomain(unittest.TestCase):
    DATA = {"cnn.com": {"fake": 1, "real": 9}, "bbc.co.uk": {"fake": 0, "real": 5}}

    def test_domain_persis(self):
        self.assertEqual(cari_domain("cnn.com", self.DATA), ("cnn.com", self.DATA["cnn.com"]))

    def test_mundur_ke_domain_induk(self):
        self.assertEqual(cari_domain("edition.cnn.com", self.DATA)[0], "cnn.com")
        self.assertEqual(cari_domain("news.bbc.co.uk", self.DATA)[0], "bbc.co.uk")

    def test_tidak_ditemukan(self):
        self.assertIsNone(cari_domain("example.org", self.DATA))
        # Domain induk saja tidak boleh cocok ke subdomain.
        self.assertIsNone(cari_domain("co.uk", self.DATA))


class TestHitungStatistik(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def tulis_csv(self, baris):
        path = self.folder / "data.csv"
        isi = "URLs,Headline,Body,Label\n" + "\n".join(baris) + "\n"
        path.write_text(isi, encoding="utf-8")
        return path

    def test_hitung_per_domain(self):
        path = self.tulis_csv([
            "https://www.reuters.com/a,Judul A,\"Isi, dengan koma\",1",
            "https://reuters.com/b,Judul B,Isi,1",
            "http://beforeitsnews.com/c,Judul C,Isi,0",
            "http://beforeitsnews.com/c,Judul C,Isi,0",  # duplikat -> dihitung sekali
            "http://edition.cnn.com/d,Judul D,Isi,0",
            "http://edition.cnn.com/e,Judul E,Isi,1",
            "http://bentrok.com/f,Judul F,Isi,0",
            "http://bentrok.com/f,Judul F,Isi,1",  # label bentrok -> dibuang
            "bukan link,Judul G,Isi,1",  # tidak terbaca -> dibuang
            "https://valid.com/h,Judul H,Isi,2",  # label tidak valid -> dibuang
        ])
        statistik, ringkasan = hitung_statistik(path)

        self.assertEqual(statistik["domain"], {
            "reuters.com": {"fake": 0, "real": 2},
            "beforeitsnews.com": {"fake": 1, "real": 0},
            "edition.cnn.com": {"fake": 1, "real": 1},
        })
        # Urutan: berita terbanyak dulu, lalu nama domain.
        self.assertEqual(list(statistik["domain"]), ["edition.cnn.com", "reuters.com", "beforeitsnews.com"])
        self.assertEqual(statistik["jumlah_berita"], 5)
        self.assertEqual(ringkasan["jumlah_baris"], 10)
        self.assertEqual(ringkasan["label_tidak_valid"], 1)
        self.assertEqual(ringkasan["link_label_bentrok"], 1)
        self.assertEqual(ringkasan["link_duplikat"], 1)
        self.assertEqual(ringkasan["link_tidak_terbaca"], 1)
        self.assertEqual(ringkasan["jumlah_domain"], 3)
        # reuters.com (2) + beforeitsnews.com (1) semuanya berlabel sama.
        self.assertEqual(ringkasan["berita_dari_domain_satu_label"], 3)

    def test_simpan_lalu_muat_kembali(self):
        path = self.tulis_csv(["https://www.reuters.com/a,Judul A,Isi,1"])
        statistik, _ = hitung_statistik(path)
        file_json = self.folder / "domain_stats.json"
        source_stats.simpan_statistik(statistik, file_json)
        self.assertEqual(muat_statistik(file_json), statistik)


class TestMuatStatistik(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "domain_stats.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_file_belum_ada(self):
        self.assertIsNone(muat_statistik(self.path))

    def test_file_rusak_atau_salah_format(self):
        isi_salah = [
            "bukan json",
            json.dumps([]),
            json.dumps({}),
            json.dumps({"domain": {}}),
            json.dumps({"domain": {"a.com": {"fake": 1}}}),
            json.dumps({"domain": {"a.com": {"fake": -1, "real": 0}}}),
            json.dumps({"domain": {"a.com": {"fake": "1", "real": 0}}}),
            json.dumps({"domain": {"a.com": {"fake": True, "real": 0}}}),
        ]
        for isi in isi_salah:
            with self.subTest(isi=isi):
                self.path.write_text(isi, encoding="utf-8")
                self.assertIsNone(muat_statistik(self.path))

    def test_file_asli_valid_jika_sudah_dibuat(self):
        if not source_stats.FILE_STATISTIK.exists():
            self.skipTest("model/domain_stats.json belum dibuat (jalankan python source_stats.py)")
        self.assertIsNotNone(muat_statistik(), "model/domain_stats.json rusak atau salah format")


if __name__ == "__main__":
    unittest.main()
