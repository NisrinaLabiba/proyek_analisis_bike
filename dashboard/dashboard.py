import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

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

# Sidebar Filtering
st.sidebar.subheader("🔍 Filter Data")

default_start = day_df.index.min()
default_end = day_df.index.max()
start_date = st.sidebar.date_input("Tanggal Mulai", default_start)
end_date = st.sidebar.date_input("Tanggal Akhir", default_end)

if start_date > end_date:
    st.sidebar.error("Tanggal akhir harus lebih besar atau sama dengan tanggal mulai!")

# Pilihan musim dengan opsi "Semua Musim"
season_options = ["Semua Musim"] + list(season_labels.values())
selected_season = st.sidebar.selectbox("Pilih Musim", season_options)

# Pilihan hari dengan opsi "Semua Hari"
day_options = ["Semua Hari"] + list(workingday_labels.values())
selected_day = st.sidebar.selectbox("Pilih Hari", day_options)

# Filter dataset berdasarkan tanggal, musim, dan hari kerja
filtered_day_df = day_df.loc[start_date:end_date]

if selected_season != "Semua Musim":
    filtered_day_df = filtered_day_df[filtered_day_df['season_label'] == selected_season]

if selected_day != "Semua Hari":
    filtered_day_df = filtered_day_df[filtered_day_df['workingday_label'] == selected_day]

# Debugging untuk memastikan data tidak kosong
st.write("### Contoh Data Setelah Filter")
st.write(filtered_day_df.head())

# Menampilkan semua visualisasi secara bersamaan
st.subheader("🚲 Analisis Peminjaman Sepeda")

# Tren Musiman
st.write("### Tren Musiman")
plt.figure(figsize=(8,5))
sn.barplot(x=filtered_day_df['season_label'], y=filtered_day_df['cnt'], ci=None, palette='Blues')
plt.xlabel("Musim")
plt.ylabel("Jumlah Peminjaman")
plt.title("Rata-rata Peminjaman Sepeda per Musim")
st.pyplot(plt)

# Hari Kerja vs Akhir Pekan
st.write("### Hari Kerja vs Akhir Pekan")
plt.figure(figsize=(8,5))
sn.barplot(x=filtered_day_df['workingday_label'], y=filtered_day_df['cnt'], ci=None, palette='Blues')
plt.xlabel("Kategori Hari")
plt.ylabel("Jumlah Peminjaman")
plt.title("Peminjaman Sepeda pada Hari Kerja vs Akhir Pekan")
st.pyplot(plt)

# Pola Peminjaman Jam Kerja
st.write("### Pola Peminjaman Jam Kerja")
plt.figure(figsize=(10,5))
sn.barplot(x=hour_df['hr'], y=hour_df['cnt'], ci=None, palette='Blues')
plt.xlabel("Jam")
plt.ylabel("Jumlah Peminjaman")
plt.title("Distribusi Peminjaman Sepeda per Jam")
st.pyplot(plt)

# Clustering Pengguna dengan Hierarchical Clustering
st.write("### Segmentasi Pengguna Sepeda dengan Hierarchical Clustering")
rfm = hour_df.groupby('instant').agg(
    Recency=('instant', lambda x: x.max() - x.min()),
    Frequency=('instant', 'count'),
    Monetary=('cnt', 'sum')
).reset_index()
st.write(rfm.head())

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
linked = linkage(rfm_scaled, method='ward')

plt.figure(figsize=(10,5))
dendrogram(linked)
plt.title("Dendrogram untuk Hierarchical Clustering")
plt.xlabel("Pengguna")
plt.ylabel("Jarak")
st.pyplot(plt)

# Prediksi ARIMA
st.write("### Prediksi Peminjaman Sepeda dengan ARIMA")
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
