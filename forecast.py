import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import seaborn as sns
from datetime import timedelta  # Import timedelta dari modul datetime
import plotly.express as px
# import matplotlib.pyplot as plt
# Menambahkan judul aplikasi
st.title('Aplikasi Time-Series Forecasting (Prediksi) dengan Machine Learning')

# Menambahkan sidebar
st.sidebar.header('Pilih Algoritma')

# Menambahkan dropdown menu di sidebar
option = st.sidebar.selectbox(
    'Pilih Algoritma',
    ['ARIMA', 'XGBOOST']
)

# Menampilkan pilihan di halaman utama
st.write(f'Anda memilih {option}')

# Menampilkan penjelasan algoritma
if option == 'ARIMA':
    st.write('ARIMA adalah Algoritma yang digunakan untuk analisis deret waktu, menggabungkan Autoregressive (AR), Integrated (I), dan Moving Average (MA) untuk memprediksi nilai masa depan berdasarkan data historis.')

elif option == 'XGBOOST':
    st.write('XGBOOST adalah algoritma gradient boosting untuk di machine learning yang bisa digunakan untuk memprediksi nilai masa depan berdasarkan data historis.')

else:
    st.write('Naive Bayes adalah algoritma klasifikasi berbasis probabilitas yang berdasarkan pada Teorema Bayes dengan asumsi independensi antara fitur-fitur.')

st.subheader('Upload Dataset')

# Fitur upload dataset
uploaded_file = st.file_uploader("Upload file dataset (CSV)", type=['csv'])

if uploaded_file is not None:
    try:
        # Membaca dataset dari file yang di-upload
        df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        st.write('Preview Dataset:')

        # Menampilkan dataset dengan scroll
        st.dataframe(df.head(10), height=300)
        st.write(f'Jumlah Baris: {df.shape[0]}')
        st.write(f'Jumlah Kolom: {df.shape[1]}')

        if st.checkbox('Tampilkan deskripsi dataset Awal'):
            st.subheader('Deskripsi Dataset')
            st.write(df.describe())

        # Display missing values information
        st.write('Jumlah Nilai Hilang:')
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0])

        st.subheader('Pemilihan Variabel')
        # Menampilkan dropdown untuk memilih variabel fitur dan filter
        columns = df.columns.tolist()
        waktu = st.selectbox('Pilih Kolom Waktu', options=columns)
        resample_option = st.selectbox(
        'Pilih Frekuensi Prediksi',
        options=['D', 'M', 'Y', 'W'],  # D: Harian, M: Bulanan, Y: Tahunan, W: Mingguan
        format_func=lambda x: {'D': 'Harian', 'M': 'Bulanan', 'Y': 'Tahunan', 'W': 'Mingguan'}[x])
        # filter = st.selectbox('Pilih Prediksi Objek berdasaran Kolom', options=[""] + columns, format_func=lambda x: "Pilih Kolom" if x == "" else x)
        target = st.selectbox('Pilih Kolom Target', options=columns)
        filter = st.selectbox('Pilih Prediksi Objek berdasarkan Kolom', options=[""] + columns, format_func=lambda x: "Pilih Kolom" if x == "" else x)

        if filter:  # Pastikan filter tidak kosong
            unique_values = df[filter].unique()
            selected_values = st.multiselect(f'Pilih Nilai Unik dari {filter}', options=unique_values)

            if selected_values:
                df = df[df[filter].isin(selected_values)]
            else:
                st.warning(f'Tidak ada nilai unik yang dipilih untuk {filter}. Menampilkan dataset asli tanpa filter.')
        else:
            st.warning('Kolom Prediksi Objek (filter) tidak dipilih. Menampilkan dataset asli tanpa filter.')
        # else:
        #     st.warning('Kolom Prediksi Objek (filter) tidak dipilih. Menampilkan dataset asli tanpa filter.')

        n_periods = st.number_input(
            'Berapa lama prediksi berdasarkan frekuensi prediksi',
            min_value=1,  # Nilai minimum
            max_value=365,  # Nilai maksimum
            value=30,  # Nilai default
            step=1  # Langkah perubahan nilai
        )

        df = df[[waktu, target]]
        df[target] = pd.to_numeric(df[target], errors='coerce').astype('Int64')  # Menggunakan Int64 untuk mendukung nilai NaN
        df = df.set_index(waktu)
        df.index = pd.to_datetime(df.index)
        df = df.resample(resample_option).sum()

        # Pembersihan dan Transformasi Data
        st.subheader('Pembersihan dan Transformasi Data')

        # Menangani nilai hilang
        if st.checkbox('Hapus Baris dengan Nilai Hilang'):
            missing_count_before = df.isnull().sum().sum()
            df = df.dropna()
            missing_count_after = df.isnull().sum().sum()
            st.write(f'Jumlah nilai hilang yang dihapus: {missing_count_before - missing_count_after}')

        if st.checkbox('Isi Nilai Hilang dengan Rata-Rata'):
            # Tampilkan informasi nilai hilang sebelum pengisian
            missing_before = df.isnull().sum()
            if missing_before.sum() > 0:
                st.write("Jumlah nilai hilang per kolom sebelum pengisian:")
                st.write(missing_before[missing_before > 0])

                # Pisahkan kolom numerik dan kategorikal
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
                categorical_columns = df.select_dtypes(include=['object']).columns

                # Isi nilai hilang untuk kolom numerik dengan mean
                if len(numeric_columns) > 0:
                    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
                    st.write("Kolom numerik diisi dengan nilai rata-rata")

                # Isi nilai hilang untuk kolom kategorikal dengan mode (nilai yang paling sering muncul)
                if len(categorical_columns) > 0:
                    for col in categorical_columns:
                        df[col] = df[col].fillna(df[col].mode()[0])
                    st.write("Kolom kategorikal diisi dengan nilai yang paling sering muncul")

                # Tampilkan informasi nilai hilang setelah pengisian
                missing_after = df.isnull().sum()
                if missing_after.sum() > 0:
                    st.write("Jumlah nilai hilang per kolom setelah pengisian:")
                    st.write(missing_after[missing_after > 0])
                else:
                    st.write("Semua nilai hilang telah diisi!")

        st.write('Dataset Setelah Pembersihan dan Transformasi:')
        st.dataframe(df, height=300)
        st.write(f'Jumlah Baris: {df.shape[0]}')
        st.write(f'Jumlah Kolom: {df.shape[1]}')

        start_date = df.index.min().date()
        end_date = df.index.max().date()

        # Slider untuk memilih rentang tanggal
        date_range = st.slider(
            'Pilih Rentang Tanggal',
            min_value=start_date,
            max_value=end_date,
            value=(start_date, end_date),  # Nilai default adalah seluruh rentang tanggal
            format="YYYY-MM-DD"
        )

        # Filter DataFrame berdasarkan rentang tanggal yang dipilih
        df = df.loc[date_range[0]:date_range[1]]

        # Memilih proporsi data latih dan uji
        # test_size = st.slider('Pilih Proporsi Data Uji (%)', min_value=10, max_value=90, value=20)

        resample_mapping = {
            'D': 7,  # Harian
            'W': 4,  # Mingguan
            'M': 12, # Bulanan
            'Y': 5   # Tahunan
        }

        # Mendapatkan nilai resample_factor berdasarkan resample_option
        resample_factor = resample_mapping.get(resample_option, None)

        if st.checkbox('Tampilkan Grafik Awal'):
            st.subheader('Grafik')
            fig = px.line(
                df,
                x=df.index,  # Kolom waktu sebagai sumbu X
                y=target,  # Kolom filter sebagai sumbu Y
                title='Grafik Penjualan Obat',
                labels={waktu: 'Waktu', filter: 'Penjualan'},
                template='plotly_white'
            )
            # Menampilkan grafik di Streamlit
            st.plotly_chart(fig)

        if st.button('Prediksi'):
            if target and waktu:
                # model.summary()
                end_date = df.index.max()
                end_date = pd.Timestamp(end_date)
                # st.write(f"end_date: {end_date}, type: {type(end_date)}")
                # st.write(f"resample_option: {resample_option}")

                # Menyiapkan model sesuai algoritma yang dipilih
                if option == 'ARIMA':
                    model = pm.auto_arima(df, start_p=1, start_q=1,
                            test='adf',
                            max_p=3, max_q=3, m=resample_factor,
                            start_P=0, seasonal=True,
                            d=None, D=1, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)
                    
                    fitted, confint = model.predict(n_periods=n_periods, return_conf_int=True)
                    index_of_fc = pd.date_range(df.index[-1], periods = n_periods, freq=resample_option)

                    # make series for plotting purpose
                    fitted_series = pd.Series(fitted, index=index_of_fc)
                    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
                    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

                    if resample_option == 'D':  # Harian
                        start_date = end_date + timedelta(days=1)
                    elif resample_option == 'W':  # Mingguan
                        start_date = end_date + timedelta(weeks=1)
                    elif resample_option == 'M':  # Bulanan
                        start_date = end_date + pd.DateOffset(months=1)
                    elif resample_option == 'Y':  # Tahunan
                        start_date = end_date + pd.DateOffset(years=1)
                    else:
                        st.error("Frekuensi waktu tidak valid.")
                        start_date = None

                    # st.write(f"start_date: {start_date}, type: {type(start_date)}")
                    # Pastikan start_date valid sebelum melanjutkan
                    if start_date:
                        # Buat DataFrame baru untuk prediksi
                        date_range = pd.date_range(start=start_date, periods=n_periods, freq=resample_option)
                        df_fitted = pd.DataFrame({target: fitted}, index=date_range)

                        # Gabungkan DataFrame asli dengan DataFrame prediksi
                        dfbaru = pd.concat([df, df_fitted])

                        # Tampilkan DataFrame gabungan
                        st.write('Dataset Gabungan (Asli + Prediksi):')
                        st.dataframe(dfbaru, height=300)

                        df_before = dfbaru[dfbaru.index <= end_date]
                        df_after = dfbaru[dfbaru.index > end_date]

                        # st.dataframe(df_before)
                        # st.dataframe(df_after)
                        df_before.index = pd.to_datetime(df_before.index)
                        df_after.index = pd.to_datetime(df_after.index)


                        figa = px.line(
                            title='Prediksi Penjualan Obat',
                            labels={waktu: 'Waktu', filter: 'Penjualan'},
                            template='plotly_white'
                        )

                        # Tambahkan trace untuk data sebelum end_date
                        figa.add_scatter(
                            # df_before,
                            x=df_before.index,
                            y=df_before[target],
                            mode='lines',
                            line=dict(color='blue'),
                            name='Sebelum End Date'
                        )

                        # Tambahkan trace untuk data sesudah end_date
                        figa.add_scatter(
                            # df_after,
                            x=df_after.index,
                            y=target[target],
                            mode='lines',
                            line=dict(color='green'),
                            name='Sesudah End Date'
                        )

                        # Tambahkan garis vertikal pada end_date
                        figa.add_vline(
                            x=end_date,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="End Date",
                            annotation_position="top right"
                        )

                        # Tampilkan grafik di Streamlit
                        st.plotly_chart(figa)

                # elif option == 'XGBOOST':
                #     model = DecisionTreeClassifier()
                # else:
                #     model = GaussianNB()

                # # Memprediksi hasil
                # y_pred = model.predict(X_test)
                # accuracy = accuracy_score(y_test, y_pred)
                # report = classification_report(y_test, y_pred, output_dict=True)
                # conf_matrix = confusion_matrix(y_test, y_pred)

                # st.write(f'Akurasi Model: {accuracy:.2f}')
                # st.subheader('Classification Report')
                # st.text(classification_report(y_test, y_pred))

                # st.subheader('Confusion Matrix')
                # fig, ax = plt.subplots(figsize=(10, 7))
                # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                # plt.xlabel('Predicted Labels')
                # plt.ylabel('True Labels')
                # st.pyplot(fig)

                # # Menampilkan beberapa contoh prediksi
                # st.subheader('Contoh Prediksi')
                # examples = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
                # st.write(examples.head(10))

        #     else:
        #         st.error('Pilih kolom filter dan fitur dengan benar.')

    except pd.errors.EmptyDataError:
        st.error('File kosong atau format tidak valid.')
    except Exception as e:
        st.error(f'Error: {e}')

# Add after st.dataframe(df, height=300) line