"""
NewsCheck — aplikasi web Streamlit untuk memprediksi apakah sebuah judul
berita berbahasa Inggris termasuk FAKE atau REAL.

Semua proses prediksi memakai predict_news() dari predict.py
(model TF-IDF + Linear SVM). File ini hanya mengatur tampilan.

Cara menjalankan:
    streamlit run app.py
"""

import html

import streamlit as st

# ---------------------------------------------------------------
# 1. Konfigurasi halaman (harus jadi perintah Streamlit pertama)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="NewsCheck — Fake vs Real News",
    page_icon="📰",
    layout="centered",
)

# ---------------------------------------------------------------
# 2. Gaya tampilan (CSS)
# ---------------------------------------------------------------
st.html(
    """
    <style>
    .nc-hero { text-align: center; padding: 0.5rem 0 1.25rem; }
    .nc-logo { font-size: 3rem; line-height: 1; }
    .nc-title {
        font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em;
        margin: 0.35rem 0 0.4rem;
    }
    .nc-title span { color: #2563eb; }
    .nc-tagline {
        font-size: 1.05rem; opacity: 0.75; max-width: 34rem; margin: 0 auto;
    }

    .nc-result {
        border: 1px solid; border-radius: 14px;
        padding: 1.25rem 1.5rem; margin: 0.5rem 0 0.75rem;
    }
    .nc-result.fake { background: rgba(220, 38, 38, 0.07); border-color: rgba(220, 38, 38, 0.45); }
    .nc-result.real { background: rgba(22, 163, 74, 0.07); border-color: rgba(22, 163, 74, 0.45); }
    .nc-result-label {
        font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.7;
    }
    .nc-verdict { font-size: 2.2rem; font-weight: 800; line-height: 1.2; margin: 0.15rem 0 0.6rem; }
    .nc-result.fake .nc-verdict { color: #dc2626; }
    .nc-result.real .nc-verdict { color: #15803d; }
    .nc-headline { font-style: italic; opacity: 0.85; margin-bottom: 0.6rem; overflow-wrap: anywhere; }
    .nc-desc { font-size: 0.95rem; }

    .nc-footer { text-align: center; font-size: 0.85rem; opacity: 0.6; padding-top: 0.5rem; }

    /* Teks bantuan bawaan "Press Enter to submit form · 0/300" menutupi judul
       yang panjang (terutama di HP), jadi disembunyikan. Batas 300 karakter
       tetap berlaku dan dijelaskan di tooltip input. */
    [data-testid="InputInstructions"] { display: none; }
    </style>
    """
)

# ---------------------------------------------------------------
# 3. Muat fungsi prediksi dari predict.py
#    (model .joblib ikut dimuat saat predict.py di-import)
# ---------------------------------------------------------------
try:
    import predict
    from predict import predict_news
except Exception as e:
    st.error(
        "Model gagal dimuat. Pastikan file `model/newscheck_pipeline.joblib` "
        "ada dan semua library di `requirements.txt` sudah ter-install."
    )
    st.caption(f"Detail teknis: {e}")
    st.stop()

# Statistik sumber (model/domain_stats.json) bersifat opsional. Jika file belum
# dibuat dengan `python source_stats.py`, kolom link tidak ditampilkan dan
# aplikasi berjalan seperti biasa.
try:
    import source_stats

    STATISTIK_SUMBER = source_stats.muat_statistik()
except Exception:
    STATISTIK_SUMBER = None


# Jika kurang dari 50% kata penting dikenali model, hasil tetap ditampilkan
# tetapi diberi peringatan bahwa prediksinya kurang bisa diandalkan.
BATAS_KATA_DIKENALI = 0.5


def cek_kosakata(teks):
    """Hitung berapa kata penting pada judul yang ada di kosakata TF-IDF model.

    "Kata penting" = kata yang tersisa setelah TF-IDF membuang stop words
    (the, is, of, ...). Mengembalikan tuple (jumlah_dikenali, jumlah_kata).

    Jika tidak ada kata yang dikenali, vektor TF-IDF berisi nol semua dan
    model tetap mengeluarkan label tanpa dasar apa pun, jadi kasus ini ditolak.
    Fungsi ini hanya membaca model yang sudah dimuat predict.py, tanpa mengubahnya.
    """
    try:
        tfidf = predict.model.named_steps["tfidf"]
        kata = set(tfidf.build_analyzer()(teks))
        dikenali = sum(1 for k in kata if k in tfidf.vocabulary_)
        return dikenali, len(kata)
    except Exception:
        # Jika struktur model berubah, lewati pengecekan agar aplikasi tetap jalan.
        return None


def format_angka(n):
    """1234 -> "1.234" (format angka Indonesia)."""
    return f"{n:,}".replace(",", ".")


def tampilkan_catatan_sumber(link):
    """Tampilkan hitungan berita FAKE/REAL dari situs link tersebut di dataset.

    Catatan ini terpisah dari prediksi: model tetap hanya membaca judul.
    """
    domain = source_stats.ambil_domain(link)
    if domain is None:
        st.warning(
            "Link berita tidak bisa dibaca, jadi catatan sumber tidak ditampilkan. "
            "Contoh link yang benar: *https://www.reuters.com/world/...*",
            icon="⚠️",
        )
        return

    cocok = source_stats.cari_domain(domain, STATISTIK_SUMBER["domain"])
    with st.container(border=True):
        st.markdown("**🔗 Catatan sumber**")
        if cocok is None:
            st.markdown(
                f"Situs `{domain}` tidak ada di dataset NewsCheck, jadi belum ada "
                "catatan untuk sumber ini."
            )
        else:
            domain_cocok, jumlah = cocok
            fake, real = jumlah["fake"], jumlah["real"]
            total = fake + real
            st.markdown(
                f"Di dataset NewsCheck ada **{format_angka(total)} berita** dari "
                f"`{domain_cocok}`: **{format_angka(fake)} berlabel FAKE** "
                f"({round(100 * fake / total)}%) dan **{format_angka(real)} berlabel "
                f"REAL** ({round(100 * real / total)}%)."
            )
        st.caption(
            "Catatan ini hanya hitungan dari dataset, bukan penilaian apakah situs "
            "tersebut terpercaya, dan tidak dipakai model untuk prediksi di atas."
        )

# ---------------------------------------------------------------
# 4. Halaman utama: judul & deskripsi
# ---------------------------------------------------------------
st.html(
    """
    <div class="nc-hero">
        <div class="nc-logo">📰</div>
        <div class="nc-title">News<span>Check</span></div>
        <p class="nc-tagline">
            Masukkan judul berita berbahasa Inggris, lalu model Machine Learning
            akan memprediksi apakah judul tersebut termasuk <b>FAKE</b> atau <b>REAL</b>.
        </p>
    </div>
    """
)

# ---------------------------------------------------------------
# 5. Form klasifikasi berita
# ---------------------------------------------------------------
with st.form("form_cek_berita"):
    headline = st.text_input(
        "Judul berita (bahasa Inggris)",
        key="judul",
        placeholder="Contoh: Senate passes bill to fund infrastructure projects",
        help="Masukkan judul (headline) saja, bukan isi berita. Maksimal 300 "
        "karakter. Tekan Enter atau klik Check News untuk memeriksa.",
        max_chars=300,
    )
    link = ""
    if STATISTIK_SUMBER:
        link = st.text_input(
            "Link berita (opsional)",
            key="link",
            placeholder="Contoh: https://www.reuters.com/world/...",
            help="Jika diisi, NewsCheck menampilkan berapa berita FAKE dan REAL "
            "dari situs ini di dataset. Link tidak memengaruhi prediksi model.",
            max_chars=500,
        )
    tombol_cek = st.form_submit_button(
        "Check News", type="primary", icon="🔍", width="stretch"
    )

# ---------------------------------------------------------------
# 6. Hasil prediksi
# ---------------------------------------------------------------
TAMPILAN_HASIL = {
    "FAKE": {
        "kelas": "fake",
        "judul": "🚨 FAKE",
        "penjelasan": "Pola kata pada judul ini lebih mirip dengan judul berita "
        "<b>palsu (fake)</b> di data latih model.",
    },
    "REAL": {
        "kelas": "real",
        "judul": "✅ REAL",
        "penjelasan": "Pola kata pada judul ini lebih mirip dengan judul berita "
        "<b>asli (real)</b> di data latih model.",
    },
}

if tombol_cek:
    headline = (headline or "").strip()
    kosakata = cek_kosakata(headline) if headline else None

    if not headline:
        st.warning("Judul berita tidak boleh kosong. Silakan isi dulu.", icon="⚠️")
    elif kosakata is not None and kosakata[0] == 0:
        st.warning(
            "Model tidak mengenali kata-kata pada judul ini, jadi tidak bisa "
            "memberi prediksi. Gunakan judul berita berbahasa Inggris, misalnya: "
            "*Senate passes bill to fund infrastructure projects*.",
            icon="⚠️",
        )
    else:
        try:
            hasil = predict_news(headline)
        except Exception as e:
            st.error("Terjadi kesalahan saat memproses prediksi. Silakan coba lagi.")
            st.caption(f"Detail teknis: {e}")
        else:
            tampilan = TAMPILAN_HASIL[hasil]
            st.html(
                f"""
                <div class="nc-result {tampilan['kelas']}">
                    <div class="nc-result-label">Hasil prediksi</div>
                    <div class="nc-verdict">{tampilan['judul']}</div>
                    <div class="nc-headline">“{html.escape(headline)}”</div>
                    <div class="nc-desc">{tampilan['penjelasan']}</div>
                </div>
                """
            )
            if kosakata is not None:
                dikenali, jumlah_kata = kosakata
                if dikenali / jumlah_kata < BATAS_KATA_DIKENALI:
                    st.warning(
                        f"Model hanya mengenali **{dikenali} dari {jumlah_kata} kata "
                        "penting** pada judul ini, jadi hasil prediksi di atas "
                        "**kurang bisa diandalkan**. Ini biasa terjadi jika judul "
                        "tidak berbahasa Inggris atau memakai banyak kata yang "
                        "tidak ada di data latih.",
                        icon="⚠️",
                    )
            if STATISTIK_SUMBER and (link or "").strip():
                tampilkan_catatan_sumber(link)
            st.info(
                "Hasil ini adalah **prediksi Machine Learning** berdasarkan pola "
                "dataset pelatihan, **bukan verifikasi fakta**. Model bisa salah, "
                "jadi tetap cek kebenaran berita dari sumber terpercaya.",
                icon="ℹ️",
            )

# ---------------------------------------------------------------
# 7. Informasi model
# ---------------------------------------------------------------
st.divider()
st.subheader("🧠 Informasi Model")
st.write(
    "NewsCheck memakai model **TF-IDF + Linear Support Vector Machine (Linear SVM)** "
    "yang dilatih dengan judul berita berbahasa Inggris."
)

kolom_1, kolom_2, kolom_3 = st.columns(3)
kolom_1.metric("Akurasi pengujian", "77,60%", border=True)
kolom_2.metric("Judul data latih", "2.264", border=True)
kolom_3.metric("Judul data uji", "567", border=True)

st.markdown("**Cara kerja klasifikasi**")
langkah_1, langkah_2, langkah_3 = st.columns(3)
with langkah_1.container(border=True):
    st.markdown("**1. Bersihkan teks**")
    st.caption(
        "Judul diubah ke huruf kecil, lalu kata umum bahasa Inggris "
        "(*stop words* seperti “the” dan “is”) dibuang."
    )
with langkah_2.container(border=True):
    st.markdown("**2. TF-IDF**")
    st.caption(
        "Setiap kata diubah menjadi angka (bobot). Kata yang khas pada sebuah "
        "judul mendapat bobot lebih tinggi daripada kata yang umum."
    )
with langkah_3.container(border=True):
    st.markdown("**3. Linear SVM**")
    st.caption(
        "Model mencari batas pemisah antara pola judul FAKE dan REAL dari data "
        "latih, lalu menentukan judul baru berada di sisi mana."
    )

BATASAN_SUMBER = (
    """
- Catatan sumber (jika link diisi) hanya hitungan berita per situs di dataset,
  bukan penilaian kredibilitas situs, dan **tidak dipakai model** untuk prediksi."""
    if STATISTIK_SUMBER
    else ""
)

with st.expander("📊 Detail evaluasi & batasan model"):
    st.markdown(
        """
| Metrik | Nilai |
|---|---|
| Accuracy | 77,60% |
| Precision (weighted) | 77,52% |
| Recall (weighted) | 77,60% |
| F1-Score (weighted) | 77,47% |

**Dataset:** 2.831 judul unik setelah preprocessing (pembersihan teks dan
penghapusan judul duplikat), dibagi 80% data latih (2.264) dan 20% data uji
(567) secara *stratified*.

**Batasan:**
- Model dirancang untuk **judul berita berbahasa Inggris**.
- Model hanya membaca **judul**, bukan isi berita.
- Jika tidak ada satu pun kata pada judul yang dikenali model, aplikasi tidak
  menampilkan prediksi.
- Jika model mengenali kurang dari setengah kata penting pada judul (misalnya
  judul berbahasa Indonesia), prediksi tetap ditampilkan disertai peringatan
  bahwa hasilnya kurang bisa diandalkan.
- Model **tidak memverifikasi fakta**. Model hanya mengenali pola kata dari
  data latih, sehingga hasilnya bisa salah."""
        + BATASAN_SUMBER
    )

# ---------------------------------------------------------------
# 8. Footer
# ---------------------------------------------------------------
st.html('<div class="nc-footer">NewsCheck · Proyek UAS Machine Learning</div>')
