import streamlit as st
## import library
import pandas as pd
import numpy as np
import pickle

# df = pd.read_excel('F:\KEDE\Documents\Kuliah\SKRIPSI\projek\Dataset Magang - Data Triage dan ASMED IGD.xlsx')

# Muat model
filename = './model_logistic_regression.sav'
loaded_model = pickle.load(open(filename, 'rb'))
filename1 = './minmax_scalerr.sav'
scaler = pickle.load(open(filename1, 'rb'))


st.title("Aplikasi Sederhana Untuk Menentukan Tindak Lanjut Akhir Pasien IGD")

# Menu di sidebar
with st.sidebar:
    st.header("Menu Navigasi")
    page = st.radio("Pilih Halaman:", ["Halaman Utama", "Prediksi", "History"])

if 'History' not in st.session_state:
    st.session_state.History = []

if 'show_next' not in st.session_state:
    st.session_state.show_next = False

# Konten berdasarkan pilihan
if page == "Halaman Utama":
    # Title
    st.subheader("Halaman Utama")
    st.write("""
        Sebuah Aplikasi Sederhana berbasis Python menggunakan library Streamlit untuk menentukan tindak lanjut akhir pasien IGD, Rawat Inap/Tidak berdasarkan
        inputan variabel.
        """)

    # Company Values Section
    st.subheader("Tujuan Aplikasi ini")
    st.write("""
        - Memprediksi Tindak Lnajut Akhir Pasien IGD, Rawat Inap/Tidak.
        - Membantu Pengambilan Keputusan.
        - Membantu Memperoleh hasil Prediktif berdasarkan Model yang Dilatih.
        """)

    # Services Offered
    st.subheader("Menu Navigasi")
    st.write("""
        - **Halaman Utama:** Memuat Informasi terkait Aplikasi dan Pembuat Aplikasi.
        - **Prediksi:** Halaman untuk melakukan prediksi.
        - **History:** Lampiran dan Informasi Lokasi Penelitian.
        """)

    # Contact Information
    st.subheader("Hubungi Kami")
    st.write("""
        Email: avillaalif13@gmail.com  
        Phone: +62 87804675210  
        Address: Cibodas, Kota Tangerang, Banten.
        """)

    # Footer
    st.markdown("""
        ---
        ### Follow Us:
        [LinkedIn](https://www.linkedin.com/in/muhammad-avilla-701ba6324?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) | [Twitter](https://www.instagram.com/avillalif?igsh=czVzcnJxY2QzZ2M0)
        """)


elif page == "Prediksi":
    st.write("Masukkan data untuk prediksi.")
    nama = st.text_input("Nama")
    # kategori_umur = st.selectbox("Kategori Umur", ["15 - 25 Tahun", "45 - 64 Tahun", "1 - 4 Tahun", "26 - 44 Tahun",">= 65 Tahun","5 - 14 Tahun","29 Hari - < 1 Tahun", "0 - 6 Hari","7 - 28 Hari"])
    # jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])

    # sistolik = st.number_input("Sistolik (mmHg)", value=120)
    # diastolik = st.number_input("Diastolik (mmHg)", value=80)
    # pernapasan = st.number_input("Frekuensi Pernapasan (rpm)", value=20)
    # suhu = st.number_input("Suhu Tubuh (°C)", value=36.5)
    # nadi = st.number_input("Nadi (bpm)", value=75)
    # saturasi_oksigen = st.number_input("Saturasi Oksigen (%)", value=98)
    # skala_nyeri = st.number_input("Skala Nyeri (0-10)", value=0)

    # # Input GCS (Eye, Motorik, Verbal)
    # eye = st.number_input("GCS - Eye", min_value=1, max_value=4, value=4)
    # motorik = st.number_input("GCS - Motorik", min_value=1, max_value=6, value=6)
    # verbal = st.number_input("GCS - Verbal", min_value=1, max_value=5, value=5)
    col1, col2 = st.columns(2)

    with col1:
        kategori_umur = st.selectbox("Kategori Umur", ["15 - 25 Tahun", "45 - 64 Tahun", "1 - 4 Tahun", "26 - 44 Tahun",">= 65 Tahun","5 - 14 Tahun","29 Hari - < 1 Tahun", "0 - 6 Hari","7 - 28 Hari"])
        sistolik = st.number_input("Sistolik (mmHg)", min_value=0, max_value=300, value=120)
        pernapasan = st.number_input("Frekuensi Pernapasan (rpm)", min_value=0, max_value=100, value=20)
        nadi = st.number_input("Nadi (bpm)", min_value=0, max_value=200, value=75)
        skala_nyeri = st.number_input("Skala Nyeri (0-10)", min_value=0, max_value=10, value=0)
        eye = st.number_input("GCS - Eye", min_value=1, max_value=4, value=4)

    with col2:
        jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        diastolik = st.number_input("Diastolik (mmHg)", min_value=0, max_value=200, value=80)
        suhu= st.number_input("Suhu Tubuh (°C)", min_value=30.0, max_value=45.0, value=36.5)
        saturasi_oksigen = st.number_input("Saturasi Oksigen (%)", min_value=0, max_value=100, value=98)
        motorik = st.number_input("GCS - Motorik", min_value=1, max_value=6, value=6)
        verbal = st.number_input("GCS - Verbal", min_value=1, max_value=5, value=5)

    # Mapping Kategori menjadi numerik
    kategori_umur_mapping = {"0 - 6 Hari": 0, "7 - 28 Hari": 1, "29 Hari - < 1 Tahun": 2, "1 - 4 Tahun": 3, "5 - 14 Tahun": 4, "15 - 25 Tahun": 5, "45 - 64 Tahun": 7, "26 - 44 Tahun": 6, ">= 65 Tahun": 8, }
    jenis_kelamin_mapping = {"Laki-laki": 1, "Perempuan": 0}

    # Konversi input ke numerik
    # input_data = np.array([
    #     kategori_umur_mapping[kategori_umur],
    #     jenis_kelamin_mapping[jenis_kelamin],
    #     sistolik, diastolik, pernapasan, suhu, nadi,
    #     saturasi_oksigen, skala_nyeri, eye, motorik, verbal
    # ]).reshape(1, -1)

    # # features_to_scale = input_data[:, 1:]

    # scaled_input = scaler.transform(input_data)

    # # Definisikan nama kolom sesuai dengan model Anda
    # feature_names = ['KATEGORI UMUR', 'JENIS KELAMIN', 'sistolik', 'diastolik',
    #                 'pernapasan', 'suhu', 'nadi', 'saturasi_oksigen', 
    #                 'skala_nyeri', 'eye', 'motorik', 'verbal']

    # # Konversi input menjadi DataFrame
    # input_df = pd.DataFrame(scaled_input, columns=feature_names)

    # Prediksi menggunakan model
    if st.button("Prediksi"):
        # Mempersiapkan data untuk model
        input_data = [
            kategori_umur_mapping[kategori_umur],
            jenis_kelamin_mapping[jenis_kelamin],
            sistolik, diastolik, pernapasan, suhu,
            nadi, saturasi_oksigen, skala_nyeri,
            eye, motorik, verbal
        ]

        # Konversi input menjadi array dan scale
        input_array = np.array(input_data).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        # Konversi input menjadi DataFrame
        feature_names = ["KATEGORI UMUR", "JENIS KELAMIN", "sistolik", "diastolik", "pernapasan", "suhu", "nadi", "saturasi_oksigen", "skala_nyeri", "eye", "motorik", "verbal"]
        input_df = pd.DataFrame(scaled_input, columns=feature_names)

        # Melakukan prediksi menggunakan model
        prediction = loaded_model.predict(input_df)

        hasil_prediksi = "Pasien Berpotensi Rawat Inap" if prediction[0] == 1 else "Pasien Tidak Memerlukan Rawat Inap"

        # Simpan data ke History
        st.session_state.History.append({
            "Nama": nama,
            "Kategori Umur": kategori_umur,
            "Jenis Kelamin": jenis_kelamin,
            "Sistolik": sistolik,
            "Diastolik": diastolik,
            "Frekuensi Pernapasan": pernapasan,
            "Suhu Tubuh": suhu,
            "Nadi": nadi,
            "Saturasi Oksigen": saturasi_oksigen,
            "Skala Nyeri": skala_nyeri,
            "GCS - Eye": eye,
            "GCS - Motorik": motorik,
            "GCS - Verbal": verbal,
            "Prediksi": hasil_prediksi
        })

        st.write("### Hasil Prediksi:")
        if prediction[0] == 1:
            st.success("Pasien Berpotensi Rawat Inap")
        else:
            st.info("Pasien Tidak Memerlukan Rawat Inap")

        st.session_state.show_next = True

    if st.session_state.show_next:
        if st.button("Next"):
            st.session_state.show_next = False
            st.rerun()
            
else:
    st.write("History Inputan.")
    st.write("### Riwayat Prediksi")
    if not st.session_state.History:
        st.info("Belum ada data yang diprediksi.")
    else:
        for i, record in enumerate(st.session_state.History[::-1], start=1):
            label = f"{i}. {record.get('Nama', 'Tanpa Nama')}"
            
            with st.expander(f"{i}. {record.get('Nama', 'Tanpa Nama')}"):
                for key, value in record.items():
                    st.write(f"**{key}:** {value}")

    # st.image("F:\\KEDE\\Documents\\Kuliah\\SKRIPSI\\projek\\foto igd.jpg", caption="Our Company Building",  use_container_width=True)

