# 🌾 Dashboard Prediksi Hasil Panen Padi Sumatera

Dashboard interaktif untuk analisis dan prediksi hasil panen padi di Sumatera berdasarkan data historis dan faktor cuaca.

## 🚀 Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/shandy225-beep/ProjekPID.git
cd ProjekPID

# Install dependencies
pip install -r requirements.txt

# Run pipeline (if needed)
python run_pipeline.py

# Start dashboard
streamlit run dashboard/app.py
```

Dashboard akan terbuka di `http://localhost:8501`

## 📊 Fitur Dashboard

- **📊 Overview** - Ringkasan dataset dan statistik
- **📈 Time Series Analysis** - Trend produksi per provinsi
- **🔥 Correlation Analysis** - Analisis korelasi dan feature importance
- **🌦️ Weather Impact** - Pengaruh cuaca terhadap produksi
- **📊 Province Comparison** - Perbandingan antar provinsi
- **🗺️ Geographic Visualization** - Visualisasi geografis
- **🤖 Model Performance** - Evaluasi performa model ML
- **🔮 Future Predictions** - Prediksi produksi masa depan

## 🎯 Model Machine Learning

Dashboard menggunakan 4 model prediksi:
- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost

## 📦 Dataset

**Source:** Kaggle - Data Tanaman Padi Sumatera (1993-2020)

**Variabel:**
- Produksi (ton)
- Luas Panen (ha)
- Produktivitas (ton/ha)
- Curah Hujan (mm)
- Kelembapan (%)
- Suhu Rata-rata (°C)

## 🛠️ Tech Stack

- **Backend:** Python 3.12
- **Dashboard:** Streamlit
- **ML Libraries:** scikit-learn, XGBoost
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Database:** SQLite (SQLAlchemy)

## 📝 License

MIT License - See LICENSE file for details

## 👨‍💻 Author

**shandy225-beep**

---

⭐ Star repo ini jika bermanfaat!
