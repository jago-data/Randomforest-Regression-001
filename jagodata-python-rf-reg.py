#!/usr/bin/env python
# coding: utf-8

# Kode di bawah ini mengimpor berbagai library yang diperlukan untuk analisis data dan pemodelan. `pandas` digunakan untuk manipulasi data, `matplotlib.pyplot` untuk visualisasi data, dan modul dari `sklearn` untuk pembagian data (`train_test_split`), pencarian parameter model terbaik (`GridSearchCV`), dan evaluasi model (`RandomForestRegressor`, `mean_absolute_error`, `mean_squared_error`, `mean_absolute_percentage_error`). `StandardScaler` dari `sklearn.preprocessing` digunakan untuk menstandarkan fitur. `numpy` digunakan untuk operasi numerik, `time` untuk melacak waktu eksekusi, dan `warnings` untuk memfilter peringatan selama eksekusi kode.

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import warnings as ws

ws.filterwarnings("ignore")

# Kode berikut membaca data dari file CSV bernama "DATA TA FIXS MARET APRIL.csv" menggunakan `pandas` dan kemudian menampilkan lima baris pertama dari data tersebut. Berikut adalah penjelasan langkah demi langkah:
# 
# 1. `df = pd.read_csv("DATA TA FIXS MARET APRIL.csv")`: Menggunakan fungsi `read_csv` dari `pandas` untuk membaca file CSV dan menyimpannya dalam variabel `df` sebagai DataFrame.
# 
# 2. `df.head()`: Menampilkan lima baris pertama dari DataFrame `df` untuk memberikan gambaran tentang isi dan struktur data.

# In[2]:


df = pd.read_csv("DATA TA FIXS MARET APRIL.csv")
df.head()

# Fungsi `dataframe_summary` di bawah ini  memberikan ringkasan statistik dan informasi penting lainnya dari sebuah DataFrame. Fungsi ini menerima sebuah DataFrame sebagai parameter dan menghasilkan dictionary yang berisi deskripsi statistik (menggunakan `describe` dari `pandas`), jumlah nilai yang hilang di setiap kolom (`isnull().sum()`), tipe data dari masing-masing kolom (`dtypes`), dan bentuk (jumlah baris dan kolom) dari DataFrame tersebut (`shape`). Fungsi ini membantu pengguna memahami struktur dan kualitas data yang mereka miliki.

# In[3]:


def dataframe_summary(df):
    """
    Menyediakan ringkasan statistik, pemeriksaan nilai yang hilang, tipe data, dan bentuk dari dataframe.
    Parameter:
    df (pd.DataFrame): Dataframe yang akan dianalisis.
    Mengembalikan:
    dict: Dictionary yang berisi deskripsi statistik, jumlah nilai yang hilang, tipe data, dan bentuk dataframe.
    """
    summary = {}
    # Deskripsi statistik
    summary["description"] = df.describe(include="all")
    # Memeriksa nilai yang hilang
    summary["missing_values"] = df.isnull().sum()
    # Tipe data
    summary["data_types"] = df.dtypes
    # Bentuk dataframe
    summary["shape"] = df.shape
    return summary

# In[4]:


summary = dataframe_summary(df)

print("Deskripsi Statistik:\n", summary["description"])
print("\nNilai yang Hilang:\n", summary["missing_values"])
print("\nTipe Data:\n", summary["data_types"])
print("\nBentuk Dataframe:\n", summary["shape"])

# Kode di bawah ini mengubah nama kolom DataFrame dan memformat kolom tanggal dan waktu. Kode ini pertama-tama mengubah nama kolom DataFrame `df` menjadi ["DATE_TIME", "VOLT", "AMPERE", "POWER", "IRRADIANCE"] untuk memberikan nama yang lebih deskriptif dan sesuai dengan data yang diwakilinya. Selanjutnya, kolom "DATE_TIME" diformat sebagai tipe datetime menggunakan `pd.to_datetime` dengan format "%d/%m/%Y %H:%M". Ini memastikan bahwa nilai dalam kolom "DATE_TIME" dikenali sebagai objek datetime, yang memudahkan manipulasi dan analisis waktu lebih lanjut.

# In[5]:


df.columns = ["DATE_TIME", "VOLT", "AMPERE", "POWER", "IRRADIANCE"]
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format="%d/%m/%Y %H:%M")

# Fungsi `preprocess_data` yang diberikan bertujuan untuk memproses DataFrame dengan menambahkan kolom-kolom fitur lag untuk kolom "IRRADIANCE".  Fungsi `preprocess_data` mengambil DataFrame `df` sebagai input bersama dengan parameter opsional `lag_start` dan `lag_end`, yang defaultnya diatur pada 13 dan 39. Fungsi ini melakukan perulangan untuk menambahkan kolom-kolom baru ke DataFrame yang mewakili nilai "IRRADIANCE" dengan lag yang bervariasi dari `lag_start` hingga `lag_end`. Setiap kolom baru diberi nama sesuai dengan lagnya, menggunakan fungsi `shift` dari pandas untuk memindahkan nilai "IRRADIANCE" ke atas sebanyak lag yang diberikan. Selanjutnya, fungsi ini mengubah nama kolom-kolom tersebut agar sesuai dengan urutan yang diinginkan dengan mengganti nama menggunakan `rename` dari pandas, yang memastikan kolom-kolom memiliki label yang konsisten dan informatif. Akhirnya, baris-baris yang mengandung nilai yang hilang (NaN) dihapus dengan `dropna` sehingga DataFrame yang dihasilkan siap untuk analisis lebih lanjut.

# In[6]:


def preprocess_data(df, lag_start=13, lag_end=39):
    # Loop untuk menambahkan kolom fitur lag untuk Irradiance ke dataframe
    for lag in range(lag_start, lag_end):
        df[f"IRRADIANCE_LAG_{lag}"] = df["IRRADIANCE"].shift(lag)

    df = df.rename(
        columns={
            f"IRRADIANCE_LAG_{lag}": f"IRRADIANCE_LAG_{lag - (lag_start-1)}"
            for lag in range(lag_start, lag_end + 1)
        }
    )
    df = df.dropna()
    return df

# In[7]:


df.head()

# In[8]:


df = preprocess_data(df, lag_start=13, lag_end=39)

# Kode di bawah ini melakukan beberapa langkah pra-pemrosesan data untuk persiapan model prediksi:
# 
# 1. **Memisahkan Fitur dan Target**:
#    - `X = df.drop(columns=["DATE_TIME", "VOLT", "AMPERE", "POWER", "IRRADIANCE"])`: Memisahkan fitur-fitur dari DataFrame `df`, kecuali kolom-kolom yang menjadi target prediksi ("DATE_TIME", "VOLT", "AMPERE", "POWER", "IRRADIANCE").
#    - `y = df[["VOLT", "AMPERE", "POWER"]]`: Menentukan target prediksi yang terdiri dari kolom "VOLT", "AMPERE", dan "POWER".
# 
# 2. **Membagi Data untuk Pelatihan dan Pengujian**:
#    - `train_test_split`: Memisahkan data menjadi data pelatihan dan pengujian. Data pelatihan (80%) digunakan untuk melatih model, sedangkan data pengujian (20%) digunakan untuk menguji kinerja model.
#    - `test_size=0.2, random_state=42`: Menentukan ukuran data pengujian sebesar 20% dari total data, dengan `random_state=42` untuk memastikan hasil pembagian data dapat direproduksi secara deterministik.
# 
# 3. **Standarisasi Fitur-Fitur Numerik**:
#    - `StandardScaler()`: Inisialisasi objek scaler untuk mentransformasi fitur-fitur numerik dengan cara menghilangkan rata-rata dan menskalakan ke varians unit.
#    - `X_train_scaled = scaler.fit_transform(X_train)`: Melakukan proses fitting dan transformasi pada data pelatihan.
#    - `X_test_scaled = scaler.transform(X_test)`: Menggunakan parameter yang sama dari data pelatihan untuk mentransformasi data pengujian, memastikan bahwa skala yang sama diterapkan pada kedua set data.
# 
# Langkah-langkah ini penting untuk mempersiapkan data sebelum dilatih dengan model machine learning, memastikan konsistensi dalam skala dan pembagian data untuk evaluasi model yang akurat.

# In[9]:


# Memisahkan fitur dan target
X = df.drop(columns=["DATE_TIME", "VOLT", "AMPERE", "POWER", "IRRADIANCE"])
y = df[["VOLT", "AMPERE", "POWER"]]

# Membagi data menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standarisasi fitur-fitur numerik
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Kode di bawah ini melakukan proses pelatihan, prediksi, dan evaluasi kinerja model RandomForestRegressor untuk setiap target dalam `y.columns` (yaitu "VOLT", "AMPERE", dan "POWER"). Berikut adalah penjelasan singkatnya:
# 
# 1. **Inisialisasi Variabel**:
#    - `models = {}`: Dictionary untuk menyimpan model-model yang dilatih.
#    - `predictions = {}`: Dictionary untuk menyimpan hasil prediksi dari setiap model.
#    - `mae = {}`, `rmse = {}`, `mse = {}`: Dictionary untuk menyimpan nilai Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), dan Mean Squared Error (MSE) untuk setiap target.
# 
# 2. **Iterasi untuk Setiap Target**:
#    - `for target in y.columns:`: Melakukan iterasi untuk setiap kolom target dalam `y`, yaitu "VOLT", "AMPERE", dan "POWER".
# 
# 3. **Pembuatan Model dan Pelatihan**:
#    - `model = RandomForestRegressor(random_state=42)`: Membuat model RandomForestRegressor dengan pengaturan random state untuk hasil yang konsisten.
#    - `model.fit(X_train_scaled, y_train[target])`: Melatih model menggunakan data pelatihan untuk target saat ini.
# 
# 4. **Prediksi dan Perhitungan Metrik Evaluasi**:
#    - `predictions[target] = model.predict(X_test_scaled)`: Melakukan prediksi menggunakan data pengujian.
#    - `mse[target]`, `mae[target]`, `rmse[target]`: Menghitung MSE, MAE, dan RMSE antara nilai aktual (`y_test[target]`) dan prediksi (`predictions[target]`).
#    
# 5. **Output Evaluasi Metrik**:
#    - `print(f"Metrics for {target}:")`: Menampilkan header untuk setiap target.
#    - `print(f"MSE: {mse[target]}")`: Menampilkan nilai MSE untuk target saat ini.
#    - `print(f"MAE: {mae[target]}")`: Menampilkan nilai MAE untuk target saat ini.
#    - `print(f"RMSE: {rmse[target]}")`: Menampilkan nilai RMSE untuk target saat ini.
#    - `print("")`: Menampilkan baris kosong untuk pemisah antara setiap target.
# 
# Langkah-langkah ini memberikan evaluasi yang komprehensif terhadap kinerja model untuk setiap target yang diprediksi, membantu dalam pemilihan dan penyetelan model yang lebih baik.

# In[10]:


models = {}
predictions = {}
mae = {}
rmse = {}
mse = {}

for target in y.columns:
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_scaled, y_train[target])
    models[target] = model

    predictions[target] = model.predict(X_test_scaled)
    mse[target] = mean_squared_error(y_test[target], predictions[target])
    mae[target] = mean_absolute_error(y_test[target], predictions[target])
    rmse[target] = np.sqrt(mean_squared_error(y_test[target], predictions[target]))

    print(f"Metrics for {target}:")
    print(f"MSE: {mse[target]}")  
    print(f"MAE: {mae[target]}")
    print(f"RMSE: {rmse[target]}")
    print("")

# Kode di bawah ini melakukan proses penyetelan hiperparameter menggunakan GridSearchCV untuk model RandomForestRegressor untuk setiap target dalam `y.columns` (yaitu "VOLT", "AMPERE", dan "POWER"). Berikut adalah penjelasan singkat dari kode tersebut:
# 
# 1. **Inisialisasi Variabel**:
#    - `start = time.time()`: Memulai penghitungan waktu eksekusi kode.
# 
# 2. **Grid Search untuk Penyetelan Hiperparameter**:
#    - `param_grid`: Daftar hiperparameter yang akan diuji menggunakan GridSearchCV untuk model RandomForestRegressor.
#    - `best_models = {}`, `predictions = {}`, `mae = {}`, `rmse = {}`, `mse = {}`: Dictionary untuk menyimpan model terbaik, hasil prediksi, dan metrik evaluasi (MAE, RMSE, MSE) untuk setiap target.
# 
# 3. **Iterasi untuk Setiap Target**:
#    - `for target in y.columns:`: Melakukan iterasi untuk setiap kolom target dalam `y`, yaitu "VOLT", "AMPERE", dan "POWER".
# 
# 4. **GridSearchCV dan Pelatihan Model**:
#    - `model = RandomForestRegressor(random_state=42)`: Membuat model RandomForestRegressor dengan random state untuk hasil yang konsisten.
#    - `grid_search = GridSearchCV(...)`: Membuat objek GridSearchCV dengan model, param_grid, skor yang dievaluasi (scoring), jumlah lipatan silang (cv), dan pengaturan lainnya.
#    - `grid_search.fit(X_train_scaled, y_train[target])`: Melatih GridSearchCV pada data pelatihan untuk mencari hiperparameter terbaik.
# 
# 5. **Model Terbaik dan Evaluasi**:
#    - `best_params_ = grid_search.best_params_`: Mengambil hiperparameter terbaik dari hasil GridSearchCV.
#    - `best_model = RandomForestRegressor(**best_params_, random_state=42)`: Membuat model RandomForestRegressor dengan menggunakan hiperparameter terbaik.
#    - `best_model.fit(X_train_scaled, y_train[target])`: Melatih model terbaik pada data pelatihan.
#    - `best_models[target] = best_model`: Menyimpan model terbaik dalam dictionary `best_models`.
#    - Menggunakan model terbaik untuk membuat prediksi pada data pengujian dan menghitung MSE, MAE, dan RMSE untuk evaluasi model.
# 
# 6. **Output Evaluasi Metrik**:
#    - Menampilkan hasil metrik evaluasi (MSE, MAE, RMSE) untuk setiap target.
#    - `print(f"Time Consume : {time.time() - start} s")`: Menampilkan waktu total yang dibutuhkan untuk eksekusi keseluruhan kode setelah iterasi selesai.
# 
# Langkah-langkah ini membantu dalam menemukan hiperparameter terbaik untuk model RandomForestRegressor dan mengevaluasi kinerjanya untuk setiap target yang diprediksi, meningkatkan akurasi dan performa prediksi secara keseluruhan.

# In[11]:


start = time.time()

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_features": ["auto", "sqrt", "log2"],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}
best_models = {}
predictions = {}
mae = {}
rmse = {}
mse = {}

for target in y.columns:
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        verbose=0,
        n_jobs=-1,
    )
    grid_search.fit(X_train_scaled, y_train[target])
    best_params_ = grid_search.best_params_
    print(f"Best Params for RandomForest Model, Target Var : {target}")
    print(best_params_)
    best_model = RandomForestRegressor(**best_params_, random_state=42)
    best_model.fit(X_train_scaled, y_train[target])
    best_models[target] = best_model
    predictions[target] = best_model.predict(X_test_scaled)
    mse[target] = mean_squared_error(y_test[target], predictions[target])
    mae[target] = mean_absolute_error(y_test[target], predictions[target])
    rmse[target] = np.sqrt(mean_squared_error(y_test[target], predictions[target]))

    print(f"Metrics for {target}:")
    print(f"MSE: {mse[target]}")  # Print MSE
    print(f"MAE: {mae[target]}")
    print(f"RMSE: {rmse[target]}")
    print("")
print(f"Time Consume : {time.time() - start} s")

# Selanjutnya adalah melakukan preprocess ke data baru seperti langkah-langkah yang di atas dan menggunakan model yang telah dilatih untuk memprediksi Volt, Ampere & Power untuk beberapa periode ke depan.

# In[12]:


new_df1 = pd.read_csv("DATA TA FIXS MARET APRIL.csv")
new_df2 = pd.read_csv("DATA TA FIXS APRIL MEI.csv")

# In[13]:


new_df2.head()

# In[14]:


new_df1.columns = ["DATE_TIME", "VOLT", "AMPERE", "POWER", "IRRADIANCE"]
new_df1["DATE_TIME"] = pd.to_datetime(new_df1["DATE_TIME"], format="%d/%m/%Y %H:%M")
new_df2.columns = ["DATE_TIME", "VOLT", "AMPERE", "POWER", "IRRADIANCE"]
new_df2["DATE_TIME"] = pd.to_datetime(new_df2["DATE_TIME"], format="%d/%m/%Y %H:%M")

# In[15]:


new_df = pd.concat([new_df1, new_df2], axis=0).reset_index(drop=True)

# In[16]:


new_df.head()

# In[17]:


new_df.tail()

# In[18]:


new_df = preprocess_data(new_df, lag_start=13, lag_end=39)

# In[19]:


X_new = new_df.drop(columns=["DATE_TIME", "VOLT", "AMPERE", "POWER", "IRRADIANCE"])
y_new = new_df[["VOLT", "AMPERE", "POWER"]]

# In[20]:


X_new_scaled = scaler.transform(X_new)

# In[21]:


# Initialize dictionaries to store metrics
mae = {}
rmse = {}
mse = {}  # Initialize mse dictionary

# Calculate metrics for each target variable
for target in y_new.columns:
    predictions[target] = best_models[target].predict(X_new_scaled)
    mse[target] = mean_squared_error(y_new[target], predictions[target])
    mae[target] = mean_absolute_error(y_new[target], predictions[target])
    rmse[target] = np.sqrt(mean_squared_error(y_new[target], predictions[target]))
    print(f"Metrics for {target}:")
    print(f"MSE: {mse[target]}")  # Print MSE
    print(f"MAE: {mae[target]}")
    print(f"RMSE: {rmse[target]}")
    print("")

# Kode di bawah ini menghasilkan DataFrame `result` yang berisi kolom-kolom "DATE_TIME", "VOLT", "AMPERE", dan "POWER" dari `new_df`, serta menambahkan kolom-kolom prediksi untuk "VOLT", "AMPERE", dan "POWER" yang disimpan dalam dictionary `predictions`. DataFrame `result` kemudian disaring untuk rentang tanggal antara "2024-04-12" dan "2024-05-11" dan direset ulang indeksnya. Berikut adalah penjelasan singkat dari kode tersebut:
# 
# 1. **Seleksi dan Penambahan Kolom**:
#    - `result = new_df[["DATE_TIME", "VOLT", "AMPERE", "POWER"]]`: Memilih kolom "DATE_TIME", "VOLT", "AMPERE", dan "POWER" dari DataFrame `new_df` dan menyimpannya dalam DataFrame `result`.
#    - `result["VOLT_pred"] = predictions["VOLT"]`: Menambahkan kolom "VOLT_pred" ke `result` yang berisi prediksi untuk kolom "VOLT" dari dictionary `predictions`.
#    - `result["AMPERE_pred"] = predictions["AMPERE"]`: Menambahkan kolom "AMPERE_pred" ke `result` yang berisi prediksi untuk kolom "AMPERE" dari dictionary `predictions`.
#    - `result["POWER_pred"] = predictions["POWER"]`: Menambahkan kolom "POWER_pred" ke `result` yang berisi prediksi untuk kolom "POWER" dari dictionary `predictions`.
# 
# 2. **Pemfilteran Berdasarkan Tanggal**:
#    - `result = result[(result["DATE_TIME"] >= "2024-04-12") & (result["DATE_TIME"] <= "2024-05-11")]`: Memfilter baris-baris dalam `result` di mana nilai "DATE_TIME" berada dalam rentang tanggal antara "2024-04-12" dan "2024-05-11".
#    - `reset_index(drop=True)`: Mengatur ulang indeks DataFrame setelah penyaringan untuk memastikan indeks dimulai dari 0.
# 
# 3. **Output DataFrame Hasil**:
#    - `result.head()`: Menampilkan lima baris pertama dari DataFrame `result` setelah proses pemfilteran dan penambahan kolom prediksi.
# 
# Langkah-langkah ini membantu dalam mempersiapkan data yang sesuai untuk pembuatan plot atau analisis lebih lanjut, dengan fokus pada rentang waktu yang spesifik dan menggabungkan data aktual dan prediksi untuk memudahkan perbandingan dan evaluasi.

# In[22]:


result = new_df[["DATE_TIME", "VOLT", "AMPERE", "POWER"]]
result["VOLT_pred"] = predictions["VOLT"]
result["AMPERE_pred"] = predictions["AMPERE"]
result["POWER_pred"] = predictions["POWER"]
result = result[
    (result["DATE_TIME"] >= "2024-04-12") & (result["DATE_TIME"] <= "2024-05-11")
].reset_index(drop=True)
result.head()

# Kode di bawah ini digunakan untuk membuat plot yang membandingkan nilai aktual dan nilai yang diprediksi untuk variabel "VOLT", "AMPERE", dan "POWER" dalam DataFrame `result`. Berikut adalah penjelasan singkat dari kode tersebut:
# 
# 1. **Mengatur Ukuran Plot**:
#    - `plt.figure(figsize=(14, 12))`: Mengatur ukuran gambar plot menjadi 14x12 inci untuk memastikan visualisasi yang jelas.
# 
# 2. **Plotting Variabel VOLT**:
#    - `plt.subplot(3, 1, 1)`: Membuat subplot pertama dalam grid 3x1 untuk variabel "VOLT".
#    - `plt.plot(result["DATE_TIME"], result["VOLT"], marker="o", linestyle="-", color="b", label="Actual")`: Memplot nilai aktual "VOLT" dari `result` dengan garis solid biru.
#    - `plt.plot(result["DATE_TIME"], result["VOLT_pred"], marker="o", linestyle="--", color="r", label="Predicted")`: Memplot nilai yang diprediksi "VOLT" dari `result` dengan garis putus-putus merah.
#    - Menambahkan judul, label sumbu, dan legenda untuk subplot ini.
# 
# 3. **Plotting Variabel AMPERE**:
#    - `plt.subplot(3, 1, 2)`: Membuat subplot kedua dalam grid 3x1 untuk variabel "AMPERE".
#    - Plot nilai aktual dan nilai yang diprediksi untuk "AMPERE" dengan cara yang sama seperti variabel "VOLT".
# 
# 4. **Plotting Variabel POWER**:
#    - `plt.subplot(3, 1, 3)`: Membuat subplot ketiga dalam grid 3x1 untuk variabel "POWER".
#    - Plot nilai aktual dan nilai yang diprediksi untuk "POWER" dengan cara yang sama seperti variabel "VOLT" dan "AMPERE".
# 
# 5. **Penyesuaian Layout dan Menyimpan Gambar**:
#    - `plt.tight_layout()`: Mengatur layout plot secara otomatis untuk memastikan tidak ada tumpang tindih antar subplot.
#    - `plt.savefig("Actual Vs Prediction Plot.png")`: Menyimpan plot sebagai file gambar dengan nama "Actual Vs Prediction Plot.png".
#    - `plt.show()`: Menampilkan plot secara langsung di jupyter notebook atau aplikasi yang digunakan.
# 
# Plot ini membantu dalam memvisualisasikan seberapa baik model prediksi memprediksi nilai aktual untuk masing-masing variabel target (VOLT, AMPERE, POWER), serta membandingkan pola dan tren antara nilai aktual dan nilai prediksi pada waktu tertentu.

# In[23]:


# Plotting
plt.figure(figsize=(14, 12))

# Plotting VOLT
plt.subplot(3, 1, 1)
plt.plot(
    result["DATE_TIME"],
    result["VOLT"],
    marker="o",
    linestyle="-",
    color="b",
    label="Actual",
)
plt.plot(
    result["DATE_TIME"],
    result["VOLT_pred"],
    marker="o",
    linestyle="--",
    color="r",
    label="Predicted",
)
plt.title("Actual vs Predicted (Volt)")
plt.xlabel("Date Time")
plt.ylabel("Volt")
plt.legend()

# Plotting AMPERE
plt.subplot(3, 1, 2)
plt.plot(
    result["DATE_TIME"],
    result["AMPERE"],
    marker="o",
    linestyle="-",
    color="b",
    label="Actual",
)
plt.plot(
    result["DATE_TIME"],
    result["AMPERE_pred"],
    marker="o",
    linestyle="--",
    color="r",
    label="Predicted",
)
plt.title("Actual vs Predicted (Ampere)")
plt.xlabel("Date Time")
plt.ylabel("Ampere")
plt.legend()

# Plotting POWER
plt.subplot(3, 1, 3)
plt.plot(
    result["DATE_TIME"],
    result["POWER"],
    marker="o",
    linestyle="-",
    color="b",
    label="Actual",
)
plt.plot(
    result["DATE_TIME"],
    result["POWER_pred"],
    marker="o",
    linestyle="--",
    color="r",
    label="Predicted",
)
plt.title("Actual vs Predicted (Power)")
plt.xlabel("Date Time")
plt.ylabel("Power")
plt.legend()

plt.tight_layout()
plt.savefig("Actual Vs Prediction Plot.png")
plt.show()
