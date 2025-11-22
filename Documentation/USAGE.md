# 🌾 PANDUAN PENGGUNAAN - ProjekPID
## Pipeline Data untuk Prediksi Hasil Panen Padi Sumatera

---

## 📋 DAFTAR ISI

1. [Pendahuluan](#pendahuluan)
2. [Instalasi](#instalasi)
3. [Struktur Proyek](#struktur-proyek)
4. [Menjalankan Pipeline](#menjalankan-pipeline)
5. [Dashboard Interaktif](#dashboard-interaktif)
6. [Interpretasi Hasil](#interpretasi-hasil)
7. [Troubleshooting](#troubleshooting)

---

## 📖 PENDAHULUAN

### Problem Statement
Produktivitas pertanian bergantung pada kondisi cuaca dan tanah. Diperlukan pipeline data untuk memprediksi hasil panen per daerah.

### Tujuan Proyek
Mengintegrasikan data pertanian dan cuaca untuk prediksi hasil panen menggunakan regresi sederhana dan advanced machine learning models.

### Dataset
- **Sumber**: Kaggle Dataset - Data Tanaman Padi Sumatera
- **Format**: CSV
- **Periode**: 1993-2020
- **Variabel**: Provinsi, Tahun, Produksi, Luas Panen, Curah Hujan, Kelembapan, Suhu Rata-rata

---

## 🔧 INSTALASI

### 1. Persyaratan Sistem
- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- 2GB RAM minimum
- 1GB free disk space

### 2. Clone Repository (jika menggunakan Git)
```bash
git clone <repository-url>
cd ProjekPID
```

### 3. Membuat Virtual Environment (Recommended)
```bash
# Membuat virtual environment
python -m venv venv

# Aktivasi virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Setup Environment Variables (Optional)
```bash
cp .env.example .env
# Edit .env sesuai kebutuhan
```

---

## 📁 STRUKTUR PROYEK

```
ProjekPID/
├── data/
│   ├── raw/                          # Data mentah dari CSV
│   ├── processed/                    # Data yang sudah diproses
│   └── predictions/                  # Hasil prediksi
├── src/
│   ├── extract.py                    # Script ekstraksi data
│   ├── transform.py                  # Script transformasi data
│   ├── load.py                       # Script loading ke database
│   ├── train_model.py                # Script training model
│   └── predict.py                    # Script prediksi
├── models/                           # Model ML yang sudah ditraining
├── dashboard/
│   └── app.py                        # Dashboard Streamlit
├── config/
│   └── config.yaml                   # File konfigurasi
├── logs/                             # Log files
├── notebooks/                        # Jupyter notebooks (optional)
├── requirements.txt                  # Python dependencies
├── .env.example                      # Template environment variables
└── USAGE.md                          # File ini
```

---

## 🚀 MENJALANKAN PIPELINE

### Metode 1: Menjalankan Pipeline Lengkap Otomatis

Jalankan script pipeline lengkap:

```bash
python run_pipeline.py
```

Script ini akan menjalankan semua tahapan secara berurutan.

---

### Metode 2: Menjalankan Tahapan Manual (Step-by-Step)

#### **Tahap 1: Extract (Ekstraksi Data)**

Ekstraksi data dari CSV dan validasi.

```bash
cd /workspaces/ProjekPID
python src/extract.py
``

**Output:**
- `data/processed/extracted_data.pkl` - Data mentah dalam format pickle
- Log validasi data di terminal

**Apa yang dilakukan:**
- Membaca file CSV
- Validasi integritas data
- Cek missing values dan duplikasi
- Menampilkan ringkasan data

---

#### **Tahap 2: Transform (Transformasi Data)**

Cleaning, feature engineering, dan transformasi data.

```bash
python src/transform.py
```

**Output:**
- `data/processed/transformed_data.pkl` - Data yang sudah ditransformasi
- `data/processed/transformed_data.csv` - Versi CSV

**Apa yang dilakukan:**
- **Data Cleaning:**
  - Menangani missing values dengan imputation (median/mean)
  - Deteksi dan handling outliers menggunakan IQR method
  - Menghapus duplikasi
  
- **Feature Engineering:**
  - Produktivitas = Produksi / Luas Panen
  - Lag features (Produksi_Lag1, Produksi_Lag2, Produktivitas_Lag1)
  - Anomali curah hujan per provinsi
  - Comfort Index (kombinasi kelembapan dan suhu)
  - Kategori curah hujan (Rendah, Sedang, Tinggi, Sangat Tinggi)
  
- **Encoding:**
  - Label Encoding untuk Provinsi
  - One-Hot Encoding untuk kategori curah hujan

---

#### **Tahap 3: Load (Pemuatan ke Database)**

Menyimpan data ke SQL Database dengan schema star schema.

```bash
python src/load.py
```

**Output:**
- `data/processed/data_warehouse.db` - SQLite database

**Apa yang dilakukan:**
- Membuat schema star schema dengan:
  - **Tabel Dimensi:**
    - `dim_provinsi` - Data provinsi
    - `dim_waktu` - Data waktu/tahun
  - **Tabel Fakta:**
    - `fakta_produksi` - Fakta produksi dan cuaca
- Load data ke setiap tabel
- Verifikasi integritas data

**Contoh Query Database:**
```python << 'EOF'
import pandas as pd
import sqlite3

# Koneksi ke database
conn = sqlite3.connect('data/processed/data_warehouse.db')

# Query contoh
query = """
SELECT 
    p.nama_provinsi,
    w.tahun,
    f.produksi,
    f.produktivitas
FROM fakta_produksi f
JOIN dim_provinsi p ON f.provinsi_id = p.provinsi_id
JOIN dim_waktu w ON f.waktu_id = w.waktu_id
WHERE w.tahun >= 2015
ORDER BY w.tahun DESC, f.produksi DESC
LIMIT 10
"""
df = pd.read_sql(query, conn)
print(df)
```
conn.close()
---

#### **Tahap 4: Train Model (Training Machine Learning)**

Melatih multiple regression models dan evaluasi performa.

```bash
python src/train_model.py
```

**Output:**
- `models/linear_regression.pkl` - Model Linear Regression
- `models/random_forest.pkl` - Model Random Forest
- `models/gradient_boosting.pkl` - Model Gradient Boosting
- `models/xgboost.pkl` - Model XGBoost
- `models/best_model.pkl` - Model terbaik berdasarkan R² score
- `models/evaluation_results.json` - Hasil evaluasi semua model
- `models/metadata.json` - Metadata model
- `models/feature_names.json` - Nama fitur yang digunakan
- `models/model_comparison.png` - Visualisasi perbandingan model
- `models/prediction_vs_actual.png` - Plot prediksi vs aktual
- `models/feature_importance.png` - Chart feature importance

**Apa yang dilakukan:**
- Training 4 model machine learning:
  1. **Linear Regression** - Model baseline sederhana
  2. **Random Forest** - Ensemble learning dengan decision trees
  3. **Gradient Boosting** - Boosting algorithm yang powerful
  4. **XGBoost** - Extreme Gradient Boosting (state-of-the-art)

- **Data Split Strategy:**
  - Time-series split: 80% training, 20% testing
  - Data terbaru digunakan untuk testing (realistic scenario)

- **Cross-Validation:**
  - 5-fold time-series cross-validation
  - Memastikan stabilitas model

- **Evaluasi Metrics:**
  - **R² Score** - Proporsi varians yang dijelaskan model (0-1, lebih tinggi lebih baik)
  - **RMSE** - Root Mean Square Error (semakin rendah semakin baik)
  - **MAE** - Mean Absolute Error (semakin rendah semakin baik)
  - **MAPE** - Mean Absolute Percentage Error dalam % (semakin rendah semakin baik)

**Interpretasi Hasil:**
```
Contoh Output:

Random Forest:
  R² Score: 0.8765      → Model menjelaskan 87.65% varians data
  RMSE: 245,123.45     → Error rata-rata 245k ton
  MAE: 189,234.56      → Absolute error rata-rata 189k ton
  MAPE: 12.34%         → Error persentase rata-rata 12.34%
  CV R² Score: 0.8523  → Konsistensi model baik
```

---

#### **Tahap 5: Predict (Prediksi Masa Depan)**

Melakukan prediksi batch untuk tahun-tahun mendatang.

```bash
python src/predict.py
```

**Output:**
- `data/predictions/future_predictions.csv` - Hasil prediksi per provinsi per tahun
- `data/predictions/prediction_summary.json` - Ringkasan prediksi

**Apa yang dilakukan:**
- Load model terbaik dari training
- Generate features untuk tahun-tahun mendatang (default: 5 tahun ke depan)
- Menggunakan:
  - Lag features dari data terakhir
  - Rata-rata historis untuk variabel cuaca
  - Trend historis per provinsi
- Menyimpan prediksi dalam format CSV dan JSON

**Format Output CSV:**
```csv
Provinsi,Tahun,Produksi_Prediksi,Produktivitas_Prediksi,Luas Panen,...
Aceh,2021,1950000.00,6.15,317000.00,...
Aceh,2022,2000000.00,6.31,317000.00,...
```

---

### Tahap 6: Visualisasi Dashboard

#### Menjalankan Dashboard Streamlit

```bash
streamlit run dashboard/app.py
```

Dashboard akan terbuka di browser pada `http://localhost:8501`

**Fitur Dashboard:**

1. **📊 Overview**
   - Ringkasan statistik data
   - Total data points, provinsi, rentang tahun
   - Total produksi

2. **📈 Time Series Analysis**
   - Trend produksi per provinsi dari 1993-2020
   - Interactive line charts
   - Filter provinsi

3. **🔥 Correlation Analysis**
   - Heatmap korelasi antar variabel
   - Identifikasi faktor paling berpengaruh

4. **🌦️ Weather Impact**
   - Scatter plots hubungan cuaca dengan produksi
   - Pilih variabel cuaca (Curah Hujan, Kelembapan, Suhu)
   - Trendline analysis

5. **📊 Province Comparison**
   - Bar chart perbandingan produksi antar provinsi
   - Ranking produktivitas

6. **🗺️ Geographic Visualization**
   - Visualisasi produktivitas per provinsi
   - Color-coded productivity map

7. **🤖 Model Performance**
   - Tabel perbandingan performa semua model
   - Best model highlighted
   - Prediction vs Actual plots
   - Feature importance charts

8. **🔮 Future Predictions**
   - Prediksi produksi 5 tahun ke depan
   - Time series prediction charts
   - Tabel detail prediksi per provinsi

---

## 📊 INTERPRETASI HASIL

### 1. Memahami Metrik Evaluasi

#### R² Score (Coefficient of Determination)
- **Range**: 0 hingga 1 (bisa negatif untuk model yang sangat buruk)
- **Interpretasi**:
  - R² = 1.0 → Model sempurna (100% varians dijelaskan)
  - R² = 0.85 → Model menjelaskan 85% varians (sangat baik)
  - R² = 0.70 → Model menjelaskan 70% varians (baik)
  - R² < 0.50 → Model kurang baik
- **Contoh**: R² = 0.87 berarti model dapat menjelaskan 87% variasi dalam produksi padi

#### RMSE (Root Mean Square Error)
- **Unit**: Sama dengan unit target (ton untuk produksi)
- **Interpretasi**: Error rata-rata dalam prediksi
- **Contoh**: RMSE = 250,000 ton berarti prediksi rata-rata meleset 250k ton

#### MAE (Mean Absolute Error)
- **Unit**: Sama dengan unit target
- **Interpretasi**: Rata-rata absolute error (lebih robust terhadap outlier)
- **Contoh**: MAE = 180,000 ton berarti rata-rata absolute error 180k ton

#### MAPE (Mean Absolute Percentage Error)
- **Unit**: Persentase (%)
- **Interpretasi**:
  - MAPE < 10% → Prediksi sangat akurat
  - MAPE 10-20% → Prediksi baik
  - MAPE 20-50% → Prediksi cukup
  - MAPE > 50% → Prediksi kurang baik
- **Contoh**: MAPE = 12% berarti error rata-rata 12% dari nilai aktual

### 2. Feature Importance

Feature importance menunjukkan variabel mana yang paling berpengaruh terhadap prediksi.

**Contoh Interpretasi:**
```
Top Features:
1. Produksi_Lag1 (0.35)      → Produksi tahun lalu paling berpengaruh
2. Luas Panen (0.22)         → Luas panen sangat penting
3. Produktivitas_Lag1 (0.15) → Produktivitas historis berpengaruh
4. Curah Hujan (0.12)        → Cuaca (hujan) mempengaruhi produksi
5. Kelembapan (0.08)         → Kelembapan juga berperan
```

### 3. Analisis Trend

Dari dashboard time series, Anda dapat mengidentifikasi:
- **Trend Naik**: Produksi meningkat → Kebijakan pertanian efektif
- **Trend Turun**: Produksi menurun → Perlu intervensi
- **Seasonality**: Pola berulang → Terkait dengan musim tanam
- **Anomali**: Lonjakan/penurunan drastis → Event khusus (bencana, dll)

### 4. Prediksi Masa Depan

**Cara Membaca Hasil Prediksi:**

```
Provinsi: Aceh
Tahun 2021: 1,950,000 ton (Prediksi)
Tahun 2020: 1,861,567 ton (Aktual)
Growth: +4.7%
```

**Catatan Penting:**
- Prediksi didasarkan pada asumsi kondisi cuaca rata-rata historis
- Perubahan iklim ekstrem dapat mempengaruhi akurasi
- Kebijakan baru atau teknologi pertanian baru tidak tercakup
- Gunakan prediksi sebagai **guideline**, bukan keputusan mutlak

---

## 🛠️ TROUBLESHOOTING

### Error: "FileNotFoundError: data/raw/..."
**Solusi**: Pastikan file CSV ada di folder `data/raw/`
```bash
ls data/raw/
# Jika tidak ada, copy file CSV ke folder tersebut
```

### Error: "ModuleNotFoundError: No module named..."
**Solusi**: Install dependencies yang kurang
```bash
pip install -r requirements.txt
```

### Error: "Memory Error" saat training
**Solusi**: 
- Reduce `n_estimators` di config.yaml
- Gunakan sample data lebih kecil untuk testing

### Dashboard tidak muncul
**Solusi**:
```bash
# Pastikan streamlit terinstall
pip install streamlit

# Cek port yang digunakan
streamlit run dashboard/app.py --server.port 8502
```

### Model accuracy rendah
**Solusi**:
- Cek kualitas data (missing values, outliers)
- Tambahkan more features
- Tune hyperparameters
- Gunakan more training data

---

## 📈 TIPS UNTUK LAPORAN

### 1. Struktur Laporan yang Disarankan

```
BAB 1: PENDAHULUAN
- Latar Belakang
- Rumusan Masalah
- Tujuan Penelitian

BAB 2: TINJAUAN PUSTAKA
- Machine Learning untuk Prediksi Pertanian
- Metode Regresi
- Feature Engineering

BAB 3: METODOLOGI
3.1 Dataset
    - Sumber data
    - Deskripsi variabel
    - Statistik deskriptif

3.2 Arsitektur Pipeline
    - Extract: [Screenshot dari log extract.py]
    - Transform: [Jelaskan feature engineering yang dilakukan]
    - Load: [Diagram star schema]

3.3 Model Machine Learning
    - Linear Regression
    - Random Forest
    - Gradient Boosting
    - XGBoost

3.4 Evaluasi Model
    - Metrik: R², RMSE, MAE, MAPE
    - Cross-validation strategy

BAB 4: HASIL DAN PEMBAHASAN
4.1 Eksplorasi Data
    - [Screenshot dashboard: Time Series, Correlation]
    
4.2 Feature Engineering
    - [Tabel: Features yang dibuat]
    - [Screenshot: Feature Importance]

4.3 Performa Model
    - [Tabel: Model Comparison]
    - [Screenshot: model_comparison.png]
    - [Screenshot: prediction_vs_actual.png]
    - Interpretasi hasil

4.4 Prediksi Masa Depan
    - [Tabel: Future Predictions]
    - [Chart: Trend Prediction]
    - Analisis trend

BAB 5: KESIMPULAN DAN SARAN
- Kesimpulan
- Keterbatasan
- Saran pengembangan
```

### 2. Screenshot yang Perlu Diambil

1. **Output Terminal:**
   - Ekstraksi data (ringkasan dan validasi)
   - Transformasi (statistik before/after)
   - Training (metrik evaluasi)
   - Prediksi (summary)

2. **Visualisasi dari Dashboard:**
   - Time series plot (2-3 provinsi)
   - Correlation heatmap
   - Weather impact scatter plot
   - Province comparison bar chart
   - Model comparison chart
   - Prediction vs Actual plot
   - Feature importance chart
   - Future predictions chart

3. **Code Snippets:**
   - Contoh feature engineering
   - Contoh model training
   - SQL query contoh

### 3. Tabel yang Perlu Dibuat

**Tabel 1: Statistik Deskriptif Dataset**
```
| Variabel          | Min      | Max        | Mean      | Std Dev  |
|-------------------|----------|------------|-----------|----------|
| Produksi (ton)    | 500,000  | 3,700,000  | 2,100,000 | 680,000  |
| Luas Panen (ha)   | 200,000  | 850,000    | 520,000   | 145,000  |
| Produktivitas     | 3.5      | 7.2        | 5.1       | 0.8      |
| Curah Hujan (mm)  | 222      | 3,595      | 1,850     | 620      |
| Kelembapan (%)    | 68       | 90         | 80        | 5        |
| Suhu (°C)         | 25       | 29         | 27        | 1        |
```

**Tabel 2: Feature Engineering Summary**
```
| Feature                 | Deskripsi                                    | Formula                          |
|-------------------------|----------------------------------------------|----------------------------------|
| Produktivitas           | Produksi per hektar                          | Produksi / Luas Panen            |
| Produksi_Lag1           | Produksi tahun sebelumnya                    | shift(1)                         |
| Curah_Hujan_Anomaly     | Deviasi dari rata-rata provinsi              | Value - Mean(Provinsi)           |
| Comfort_Index           | Indeks kenyamanan tanaman                    | (Kelembapan/100) * (30-|T-27|)   |
```

**Tabel 3: Perbandingan Model**
```
| Model              | R²     | RMSE       | MAE        | MAPE   | CV R²   |
|--------------------|--------|------------|------------|--------|---------|
| Linear Regression  | 0.7234 | 345,123    | 278,456    | 15.67% | 0.7012  |
| Random Forest      | 0.8765 | 245,123    | 189,234    | 12.34% | 0.8523  |
| Gradient Boosting  | 0.8654 | 258,789    | 198,765    | 12.89% | 0.8445  |
| XGBoost            | 0.8823 | 239,456    | 185,678    | 11.98% | 0.8678  |
```

### 4. Penjelasan untuk Pembahasan

**Contoh Pembahasan Feature Importance:**
```
Dari hasil analisis feature importance, ditemukan bahwa produksi tahun 
sebelumnya (Produksi_Lag1) memiliki kontribusi terbesar (35%) terhadap 
prediksi. Hal ini menunjukkan bahwa produksi padi memiliki pola temporal 
yang kuat, di mana hasil panen tahun sebelumnya sangat mempengaruhi 
tahun berikutnya. 

Faktor kedua terpenting adalah Luas Panen (22%), yang logis karena 
semakin luas area tanam, semakin tinggi produksi total. Faktor cuaca 
seperti Curah Hujan (12%) dan Kelembapan (8%) juga berpengaruh signifikan, 
mengkonfirmasi bahwa kondisi iklim berperan penting dalam produktivitas 
pertanian padi.
```

**Contoh Pembahasan Model Performance:**
```
Model XGBoost menunjukkan performa terbaik dengan R² score 0.8823, 
yang berarti model dapat menjelaskan 88.23% variasi dalam produksi padi. 
RMSE sebesar 239,456 ton menunjukkan bahwa prediksi rata-rata memiliki 
error sekitar 239k ton, atau sekitar 11.98% dari nilai aktual (MAPE). 

Hasil cross-validation (CV R² = 0.8678) yang konsisten dengan test score 
menunjukkan bahwa model stabil dan tidak mengalami overfitting. 
Model ini dapat digunakan dengan confidence tinggi untuk prediksi 
produksi padi di masa depan.
```

---

## 📞 BANTUAN TAMBAHAN

Jika mengalami kesulitan:
1. Periksa log file di `logs/pipeline.log`
2. Pastikan semua dependencies terinstall
3. Verifikasi struktur folder sesuai dengan dokumentasi
4. Jalankan pipeline step-by-step untuk identifikasi masalah

---

## 📝 CATATAN AKHIR

- **Backup Data**: Selalu backup data asli sebelum menjalankan pipeline
- **Version Control**: Gunakan Git untuk track changes
- **Documentation**: Update dokumentasi jika ada perubahan
- **Testing**: Test pipeline dengan sample data kecil terlebih dahulu

---

**Selamat menggunakan ProjekPID!** 🌾🚀

Untuk pertanyaan lebih lanjut, hubungi tim pengembang atau buka issue di repository.

**Last Updated**: November 2025
