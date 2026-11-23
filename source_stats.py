"""
Statistik sumber (domain situs) berita untuk fitur "Catatan sumber" di NewsCheck.

File ini TIDAK mengubah model. Model tetap hanya membaca judul berita.
Statistik di sini hanya hitungan berapa berita FAKE dan REAL yang berasal
dari tiap situs di dataset, lalu ditampilkan di web sebagai catatan tambahan.

Membuat / memperbarui file statistik (butuh data/data.csv):
    python source_stats.py

Hasilnya disimpan ke model/domain_stats.json dan dibaca oleh app.py.
"""

import json
import re
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
FILE_DATASET = ROOT / "data" / "data.csv"
FILE_STATISTIK = ROOT / "model" / "domain_stats.json"

# Domain yang valid: huruf kecil/angka/tanda hubung, minimal satu titik,
# dan bagian terakhir (misalnya "com") diawali huruf.
POLA_DOMAIN = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z][a-z0-9-]*$")
POLA_SKEMA = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


def ambil_domain(link):
    """Ambil nama domain dari sebuah link, atau None jika link tidak bisa dibaca.

    Contoh:
        "https://www.reuters.com/world/abc"  -> "reuters.com"
        "edition.cnn.com/2017/01/01/x.html"  -> "edition.cnn.com"
        "bukan link"                         -> None
    """
    if not isinstance(link, str):
        return None
    link = link.strip()
    if not link:
        return None
    if not POLA_SKEMA.match(link):
        link = "http://" + link.lstrip("/")
    try:
        host = urlsplit(link).hostname
    except ValueError:
        return None
    if not host:
        return None
    try:
        # Ubah domain non-Latin (misalnya .рф) ke bentuk ASCII.
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return None
    host = host.lower().rstrip(".")
    if host.startswith("www.") and "." in host[4:]:
        host = host[4:]
    if not POLA_DOMAIN.match(host):
        return None
    return host


def cari_domain(domain, data_domain):
    """Cari domain di statistik, mundur ke domain induk jika tidak ada.

    Misalnya "edition.cnn.com" dicari sebagai "edition.cnn.com" dulu,
    lalu "cnn.com". Mengembalikan (domain_yang_cocok, {"fake": n, "real": n})
    atau None jika tidak ditemukan.
    """
    bagian = domain.split(".")
    for i in range(len(bagian) - 1):
        kandidat = ".".join(bagian[i:])
        if kandidat in data_domain:
            return kandidat, data_domain[kandidat]
    return None


def muat_statistik(path=FILE_STATISTIK):
    """Baca file statistik. Mengembalikan None jika file belum ada atau rusak."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        statistik = json.loads(path.read_text(encoding="utf-8"))
        data_domain = statistik["domain"]
        if not isinstance(data_domain, dict) or not data_domain:
            return None
        for jumlah in data_domain.values():
            for kunci in ("fake", "real"):
                nilai = jumlah[kunci]
                if not isinstance(nilai, int) or isinstance(nilai, bool) or nilai < 0:
                    return None
        return statistik
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def hitung_statistik(path_csv=FILE_DATASET):
    """Hitung jumlah berita FAKE (label 0) dan REAL (label 1) per domain.

    Setiap link hanya dihitung sekali. Link yang sama tetapi labelnya
    berbeda dibuang karena tidak jelas labelnya.
    Mengembalikan (statistik, ringkasan).
    """
    import pandas as pd

    df = pd.read_csv(path_csv, usecols=["URLs", "Label"])
    jumlah_baris = len(df)

    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df[df["Label"].isin([0, 1])].copy()
    label_tidak_valid = jumlah_baris - len(df)

    df["URLs"] = df["URLs"].astype(str).str.strip()
    label_per_link = df.groupby("URLs")["Label"].nunique()
    link_bentrok = set(label_per_link[label_per_link > 1].index)
    df = df[~df["URLs"].isin(link_bentrok)]
    sebelum_dedup = len(df)
    df = df.drop_duplicates(subset=["URLs"])
    link_duplikat = sebelum_dedup - len(df)

    df["domain"] = df["URLs"].map(ambil_domain)
    link_tidak_terbaca = int(df["domain"].isna().sum())
    df = df.dropna(subset=["domain"])

    tabel = (
        df.groupby("domain")["Label"]
        .agg(fake=lambda s: int((s == 0).sum()), real=lambda s: int((s == 1).sum()))
        .reset_index()
    )
    tabel["total"] = tabel["fake"] + tabel["real"]
    tabel = tabel.sort_values(["total", "domain"], ascending=[False, True])

    data_domain = {
        baris.domain: {"fake": int(baris.fake), "real": int(baris.real)}
        for baris in tabel.itertuples()
    }
    jumlah_berita = int(tabel["total"].sum())
    satu_label = tabel[(tabel["fake"] == 0) | (tabel["real"] == 0)]

    statistik = {
        "keterangan": (
            "Jumlah berita FAKE (label 0) dan REAL (label 1) per domain di "
            "data/data.csv, tiap link dihitung sekali. Dibuat oleh "
            "source_stats.py. Tidak dipakai oleh model."
        ),
        "jumlah_berita": jumlah_berita,
        "domain": data_domain,
    }
    ringkasan = {
        "jumlah_baris": jumlah_baris,
        "label_tidak_valid": label_tidak_valid,
        "link_label_bentrok": len(link_bentrok),
        "link_duplikat": link_duplikat,
        "link_tidak_terbaca": link_tidak_terbaca,
        "jumlah_berita": jumlah_berita,
        "jumlah_domain": len(data_domain),
        "berita_dari_domain_satu_label": int(satu_label["total"].sum()),
    }
    return statistik, ringkasan


def simpan_statistik(statistik, path=FILE_STATISTIK):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(statistik, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main():
    if not FILE_DATASET.exists():
        print(f"Dataset tidak ditemukan: {FILE_DATASET}")
        print("Taruh file data.csv (kolom URLs, Headline, Body, Label) di folder data/.")
        raise SystemExit(1)

    statistik, r = hitung_statistik(FILE_DATASET)
    simpan_statistik(statistik)

    print(f"Statistik sumber disimpan ke {FILE_STATISTIK}")
    print()
    print(f"Baris di dataset          : {r['jumlah_baris']}")
    print(f"Label tidak valid (dibuang): {r['label_tidak_valid']}")
    print(f"Link dengan label bentrok : {r['link_label_bentrok']} (dibuang)")
    print(f"Link duplikat             : {r['link_duplikat']} (dihitung sekali)")
    print(f"Link tidak terbaca        : {r['link_tidak_terbaca']} (dibuang)")
    print(f"Berita yang dihitung      : {r['jumlah_berita']}")
    print(f"Jumlah domain             : {r['jumlah_domain']}")
    print()
    print("15 domain dengan berita terbanyak:")
    print(f"  {'Domain':<35}{'FAKE':>7}{'REAL':>7}")
    for domain, jumlah in list(statistik["domain"].items())[:15]:
        print(f"  {domain:<35}{jumlah['fake']:>7}{jumlah['real']:>7}")

    if r["jumlah_berita"]:
        persen = 100 * r["berita_dari_domain_satu_label"] / r["jumlah_berita"]
        print()
        print(
            f"{persen:.1f}% berita berasal dari situs yang SEMUA beritanya "
            "berlabel sama (hanya FAKE atau hanya REAL)."
        )
        if persen >= 80:
            print(
                "Artinya di dataset ini sumber hampir selalu menentukan label. "
                "Jika sumber dijadikan fitur model, model hanya akan menghafal "
                "nama situs (data leakage), jadi sumber cukup ditampilkan "
                "sebagai catatan."
            )


if __name__ == "__main__":
    main()
