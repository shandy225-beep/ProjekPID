# 🚀 QUICK START GUIDE

Panduan cepat untuk menjalankan proyek ProjekPID dalam 5 menit!

## ⚡ Instalasi Cepat

### Opsi 1: Otomatis (Recommended)

**Windows:**
```cmd
# Double-click file ini:
run_pipeline.bat
```

**Linux/Mac:**
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### Opsi 2: Manual

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run pipeline
python run_pipeline.py

# 3. Buka dashboard
streamlit run dashboard/app.py
```

---

## 📊 Akses Dashboard

Setelah pipeline selesai, buka browser dan kunjungi:
```
http://localhost:8501
```

---

## 🎯 Langkah-langkah Pipeline

Pipeline akan otomatis menjalankan:

1. ✅ **Extract** - Membaca & validasi data CSV
2. ✅ **Transform** - Cleaning & feature engineering  
3. ✅ **Load** - Simpan ke database
4. ✅ **Train** - Training 4 ML models
5. ✅ **Predict** - Prediksi 5 tahun ke depan

Total waktu: ~5-10 menit (tergantung spesifikasi komputer)

---

## 📂 Output Files

Setelah selesai, cek folder:

```
data/processed/
  ├── extracted_data.pkl           # Data mentah
  ├── transformed_data.pkl         # Data transformasi
  ├── transformed_data.csv         # CSV version
  └── data_warehouse.db            # Database

models/
  ├── best_model.pkl               # Model terbaik
  ├── evaluation_results.json      # Hasil evaluasi
  ├── model_comparison.png         # Chart perbandingan
  ├── prediction_vs_actual.png     # Accuracy plot
  └── feature_importance.png       # Feature importance

data/predictions/
  ├── future_predictions.csv       # Prediksi 5 tahun
  └── prediction_summary.json      # Ringkasan
```

---

## 🔍 Cek Hasil

### 1. Lihat Model Performance
```bash
cat models/evaluation_results.json
```

### 2. Lihat Prediksi
```bash
cat data/predictions/prediction_summary.json
```

### 3. Query Database
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/processed/data_warehouse.db')
df = pd.read_sql("SELECT * FROM fakta_produksi LIMIT 10", conn)
print(df)
```

---

## 🎨 Dashboard Features

Navigasi dashboard:
- 📊 **Overview** → Statistik dataset
- 📈 **Time Series** → Trend produksi
- 🔥 **Correlation** → Analisis korelasi
- 🌦️ **Weather Impact** → Pengaruh cuaca
- 📊 **Province Comparison** → Perbandingan provinsi
- 🗺️ **Geographic** → Peta produktivitas
- 🤖 **Model Performance** → Evaluasi model
- 🔮 **Predictions** → Prediksi masa depan

---

## ❓ Troubleshooting

### Error: ModuleNotFoundError
```bash
pip install -r requirements.txt
```

### Error: File not found
```bash
# Pastikan berada di root folder
cd /workspaces/ProjekPID
```

### Dashboard tidak muncul
```bash
# Gunakan port berbeda
streamlit run dashboard/app.py --server.port 8502
```

---

## 📚 Dokumentasi Lengkap

Untuk panduan detail, lihat:
- **USAGE.md** - Dokumentasi lengkap
- **README.md** - Overview proyek
- **CHANGELOG.md** - Riwayat perubahan

---

## 💡 Tips

1. **Untuk laporan**: Screenshot semua visualisasi di dashboard
2. **Untuk analisis**: Ekspor CSV dari predictions
3. **Untuk development**: Edit `config/config.yaml` untuk customize

---

**Happy Coding!** 🌾

*ProjekPID Team - November 2025*
