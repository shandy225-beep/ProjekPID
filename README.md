# 🌾 ProjekPID - Pipeline Data untuk Prediksi Hasil Panen Padi

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Deskripsi Proyek

**ProjekPID** adalah proyek arsitektur data end-to-end untuk memprediksi hasil panen padi di Sumatera menggunakan data historis produksi dan kondisi cuaca. Proyek ini mengimplementasikan complete data pipeline dari ekstraksi hingga deployment dashboard interaktif.

### Problem Statement
Produktivitas pertanian bergantung pada kondisi cuaca dan tanah. Diperlukan pipeline data untuk memprediksi hasil panen per daerah.

### Tujuan
Mengintegrasikan data pertanian dan cuaca untuk prediksi hasil panen menggunakan machine learning regression models.

## ✨ Fitur Utama

- ✅ **Complete ETL Pipeline** (Extract, Transform, Load)
- ✅ **Data Cleaning & Feature Engineering**
- ✅ **Star Schema Data Warehouse** (SQLite/BigQuery)
- ✅ **Multiple ML Models** (Linear Regression, Random Forest, Gradient Boosting, XGBoost)
- ✅ **Time-Series Cross-Validation**
- ✅ **Batch Prediction** untuk masa depan
- ✅ **Interactive Dashboard** dengan Streamlit
- ✅ **Comprehensive Logging & Error Handling**

## 📊 Dataset

- **Sumber**: Kaggle - Data Tanaman Padi Sumatera
- **Periode**: 1993-2020
- **Variabel**: Provinsi, Tahun, Produksi, Luas Panen, Curah Hujan, Kelembapan, Suhu Rata-rata
- **Total Data Points**: 226 records

## 🏗️ Arsitektur Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌─────────┐     ┌──────────┐     ┌──────────┐
│   Extract   │ --> │  Transform   │ --> │  Load   │ --> │  Train   │ --> │ Predict  │
│   (CSV)     │     │  (Feature    │     │  (DB)   │     │  (ML)    │     │ (Batch)  │
│             │     │  Engineering)│     │         │     │          │     │          │
└─────────────┘     └──────────────┘     └─────────┘     └──────────┘     └──────────┘
                                                                                  │
                                                                                  ▼
                                                                          ┌──────────────┐
                                                                          │  Dashboard   │
                                                                          │  (Streamlit) │
                                                                          └──────────────┘
```

## 🚀 Quick Start

### 1. Instalasi

```bash
# Clone repository
git clone <repository-url>
cd ProjekPID

# Install dependencies
pip install -r requirements.txt
```

### 2. Jalankan Pipeline Lengkap

```bash
python run_pipeline.py
```

### 3. Buka Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard akan terbuka di `http://localhost:8501`

## 📁 Struktur Proyek

```
ProjekPID/
├── data/
│   ├── raw/                    # Data CSV mentah
│   ├── processed/              # Data yang sudah diproses
│   └── predictions/            # Hasil prediksi
├── src/
│   ├── extract.py              # Ekstraksi data
│   ├── transform.py            # Transformasi & feature engineering
│   ├── load.py                 # Loading ke database
│   ├── train_model.py          # Training ML models
│   └── predict.py              # Batch prediction
├── models/                     # Model ML & visualizations
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── config/
│   └── config.yaml             # Konfigurasi
├── logs/                       # Log files
├── run_pipeline.py             # Main pipeline runner
├── requirements.txt            # Dependencies
└── USAGE.md                    # Dokumentasi lengkap
```

## 🔄 Tahapan Pipeline

### 1. Extract (Ekstraksi Data)
- Membaca data dari CSV
- Validasi integritas data
- Deteksi missing values dan duplikasi

**Menjalankan:**
```bash
python src/extract.py
```

### 2. Transform (Transformasi Data)
- **Data Cleaning**: Handle missing values, outliers, duplikasi
- **Feature Engineering**:
  - Produktivitas = Produksi / Luas Panen
  - Lag features (Produksi_Lag1, Produksi_Lag2)
  - Weather features (Anomali, Comfort Index)
  - Categorical encoding
- **Normalization**: StandardScaler

**Menjalankan:**
```bash
python src/transform.py
```

### 3. Load (Pemuatan ke Database)
- Star Schema Design:
  - Tabel Dimensi: `dim_provinsi`, `dim_waktu`
  - Tabel Fakta: `fakta_produksi`
- SQLite Database

**Menjalankan:**
```bash
python src/load.py
```

### 4. Train Model (Training ML)
- **Models**:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost
- **Evaluation Metrics**: R², RMSE, MAE, MAPE
- **Cross-Validation**: 5-fold time-series CV

**Menjalankan:**
```bash
python src/train_model.py
```

### 5. Predict (Prediksi Masa Depan)
- Batch prediction untuk 5 tahun ke depan
- Menggunakan best model dari training
- Output: CSV & JSON summary

**Menjalankan:**
```bash
python src/predict.py
```

## 📊 Dashboard Features

1. **📊 Overview** - Statistik ringkas dataset
2. **📈 Time Series Analysis** - Trend produksi per provinsi
3. **🔥 Correlation Analysis** - Heatmap korelasi variabel
4. **🌦️ Weather Impact** - Scatter plots hubungan cuaca vs produksi
5. **📊 Province Comparison** - Perbandingan antar provinsi
6. **🗺️ Geographic Visualization** - Peta produktivitas
7. **🤖 Model Performance** - Evaluasi & perbandingan model
8. **🔮 Future Predictions** - Prediksi 5 tahun ke depan

## 📈 Hasil & Performa

### Model Performance (Example)

| Model              | R² Score | RMSE    | MAE     | MAPE   |
|--------------------|----------|---------|---------|--------|
| Linear Regression  | 0.7234   | 345,123 | 278,456 | 15.67% |
| Random Forest      | 0.8765   | 245,123 | 189,234 | 12.34% |
| Gradient Boosting  | 0.8654   | 258,789 | 198,765 | 12.89% |
| **XGBoost** ⭐     | 0.8823   | 239,456 | 185,678 | 11.98% |

### Top Feature Importance

1. **Produksi_Lag1** (35%) - Produksi tahun sebelumnya
2. **Luas Panen** (22%) - Area tanam
3. **Produktivitas_Lag1** (15%) - Produktivitas historis
4. **Curah Hujan** (12%) - Kondisi cuaca
5. **Kelembapan** (8%) - Kondisi iklim

## 🛠️ Teknologi yang Digunakan

- **Python 3.8+**
- **Data Processing**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn, XGBoost
- **Database**: SQLAlchemy, SQLite
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard**: Streamlit
- **Others**: joblib, pyyaml

## 📚 Dokumentasi Lengkap

Lihat [USAGE.md](USAGE.md) untuk:
- Panduan instalasi detail
- Cara menjalankan setiap tahapan
- Interpretasi hasil
- Troubleshooting
- Tips untuk membuat laporan

## 🔧 Konfigurasi

Edit `config/config.yaml` untuk mengubah:
- Path data
- Hyperparameters model
- Database settings
- Jumlah tahun prediksi

## 📝 Logging

Semua aktivitas pipeline dicatat dalam `logs/pipeline.log` untuk debugging dan monitoring.

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan:
1. Fork repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request


## 👥 Authors

- **ProjekPID Team**

## 🙏 Acknowledgments

- Kaggle untuk dataset
- Komunitas open source untuk libraries yang digunakan
- Stakeholder yang memberikan feedback

---

**⭐ Jika proyek ini membantu, berikan star di repository!**

**Last Updated**: November 2025
