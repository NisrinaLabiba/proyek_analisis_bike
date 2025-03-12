import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Load datasets
day_df = pd.read_csv("day.csv")
hour_df = pd.read_csv("hour.csv")

day_df['dteday'] = pd.to_datetime(day_df['dteday'])
day_df.set_index('dteday', inplace=True)

# Mapping musim ke label yang lebih mudah dipahami
season_labels = {1: "Semi", 2: "Panas", 3: "Gugur", 4: "Dingin"}
day_df['season_label'] = day_df['season'].map(season_labels)

# Mapping hari kerja
workingday_labels = {0: "Akhir Pekan", 1: "Hari Kerja"}
day_df['workingday_label'] = day_df['workingday'].map(workingday_labels)

st.title("📊 Dashboard Analisis Peminjaman Sepeda")

# Sidebar navigation
menu = st.sidebar.selectbox(
    "Pilih Analisis", 
    ["Tren Musiman", "Hari Kerja vs Akhir Pekan", "Pola Peminjaman Jam Kerja", "Clustering Pengguna", "Prediksi ARIMA"]
)

# Sidebar Filtering
st.sidebar.subheader("🔍 Filter Data")

# Filter Berdasarkan Tanggal
start_date = st.sidebar.date_input("Tanggal Mulai", day_df.index.min())
end_date = st.sidebar.date_input("Tanggal Akhir", day_df.index.max())

if start_date > end_date:
    st.sidebar.error("Tanggal akhir harus lebih besar atau sama dengan tanggal mulai!")

# Pilihan musim dengan opsi "Semua Musim"
season_options = ["Semua Musim"] + list(season_labels.values())
selected_season = st.sidebar.selectbox("Pilih Musim", season_options)

# Pilihan hari dengan opsi "Semua Hari"
day_options = ["Semua Hari", "Hari Kerja", "Akhir Pekan"]
selected_day = st.sidebar.selectbox("Pilih Hari", day_options)

# Filter dataset berdasarkan tanggal, musim, dan hari kerja
filtered_day_df = day_df.loc[start_date:end_date]

if selected_season != "Semua Musim":
    filtered_day_df = filtered_day_df[filtered_day_df['season_label'] == selected_season]

if selected_day != "Semua Hari":
    filtered_day_df = filtered_day_df[filtered_day_df['workingday_label'] == selected_day]

if menu == "Tren Musiman":
    st.subheader("🚲 Tren Peminjaman Sepeda Berdasarkan Musim")
    plt.figure(figsize=(8,5))
    sn.barplot(x=filtered_day_df['season_label'], y=filtered_day_df['cnt'], ci=None, palette='coolwarm')
    plt.xlabel("Musim")
    plt.ylabel("Jumlah Peminjaman")
    plt.title("Rata-rata Peminjaman Sepeda per Musim")
    st.pyplot(plt)

elif menu == "Hari Kerja vs Akhir Pekan":
    st.subheader("📅 Perbandingan Peminjaman Sepeda: Hari Kerja vs Akhir Pekan")
    plt.figure(figsize=(8,5))
    sn.barplot(x=filtered_day_df['workingday_label'], y=filtered_day_df['cnt'], ci=None, palette='coolwarm')
    plt.xlabel("Kategori Hari")
    plt.ylabel("Jumlah Peminjaman")
    plt.title("Peminjaman Sepeda pada Hari Kerja vs Akhir Pekan")
    st.pyplot(plt)

elif menu == "Pola Peminjaman Jam Kerja":
    st.subheader("⏰ Pola Peminjaman Sepeda pada Jam Kerja vs Non-Kerja")
    plt.figure(figsize=(10,5))
    sn.barplot(x=hour_df['hr'], y=hour_df['cnt'], ci=None, palette='coolwarm')
    plt.xlabel("Jam")
    plt.ylabel("Jumlah Peminjaman")
    plt.title("Distribusi Peminjaman Sepeda per Jam")
    st.pyplot(plt)

elif menu == "Clustering Pengguna":
    st.subheader("🔍 Segmentasi Pengguna Sepeda dengan Clustering")
    features = hour_df[['hr', 'cnt', 'workingday', 'season']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    hour_df['cluster'] = kmeans.fit_predict(features_scaled)
    
    plt.figure(figsize=(10,5))
    sn.scatterplot(x=hour_df['hr'], y=hour_df['cnt'], hue=hour_df['cluster'], palette='coolwarm')
    plt.xlabel("Jam")
    plt.ylabel("Jumlah Peminjaman")
    plt.title("Clustering Pengguna Berdasarkan Jam dan Peminjaman")
    plt.legend(title="Cluster")
    st.pyplot(plt)

elif menu == "Prediksi ARIMA":
    st.subheader("📈 Prediksi Peminjaman Sepeda dengan ARIMA")
    result = seasonal_decompose(filtered_day_df['cnt'], model='additive', period=30)
    fig = result.plot()
    st.pyplot(fig)
    
    train = filtered_day_df['cnt'][:-30]
    test = filtered_day_df['cnt'][-30:]
    model = ARIMA(train, order=(5,1,2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    
    plt.figure(figsize=(10,5))
    plt.plot(filtered_day_df.index, filtered_day_df['cnt'], label='Actual')
    plt.plot(test.index, forecast, label='Predicted', color='red')
    plt.xlabel("Tanggal")
    plt.ylabel("Jumlah Peminjaman")
    plt.title("Prediksi Peminjaman Sepeda dengan ARIMA")
    plt.legend()
    st.pyplot(plt)
