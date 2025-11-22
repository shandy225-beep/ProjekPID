# 🚀 Deploy Dashboard ke Streamlit Community Cloud

## Langkah-langkah Deploy:

### 1. Pastikan Repository di GitHub
Repository ini sudah ada di: `https://github.com/shandy225-beep/ProjekPID`

### 2. Kunjungi Streamlit Community Cloud
Buka: **https://share.streamlit.io**

### 3. Login dengan GitHub
- Klik "Sign in with GitHub"
- Authorize Streamlit

### 4. Deploy Aplikasi Baru
1. Klik tombol **"New app"**
2. Isi form:
   - **Repository:** `shandy225-beep/ProjekPID`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
   - **App URL (optional):** pilih nama custom URL

3. Klik **"Deploy!"**

### 5. Tunggu Proses Deploy
- Streamlit akan install dependencies dari `requirements.txt`
- Proses memakan waktu 2-5 menit
- Log akan ditampilkan real-time

### 6. Dashboard Siap! 🎉
URL akan berbentuk:
```
https://[your-app-name].streamlit.app
```

## ⚠️ Catatan Penting:

### Data Files
Dashboard perlu file data yang sudah diproses:
- `data/processed/transformed_data.pkl`
- `models/evaluation_results.json`
- `models/metadata.json`

**Solusi:**
1. Jalankan pipeline lokal: `python run_pipeline.py`
2. Commit dan push semua file hasil ke GitHub
3. Redeploy atau restart app di Streamlit Cloud

### Environment Variables (Opsional)
Jika ada API keys atau secrets:
1. Buka app settings di Streamlit Cloud
2. Klik "⋮" → "Settings" → "Secrets"
3. Tambahkan secrets dalam format TOML

## 🔄 Update Dashboard

Setiap kali push ke branch `main`, Streamlit akan otomatis redeploy!

```bash
git add .
git commit -m "Update dashboard"
git push origin main
```

## 📊 Monitoring

- **Status:** Check di dashboard Streamlit Cloud
- **Logs:** Lihat di app settings → Logs
- **Analytics:** Tersedia di dashboard Streamlit Cloud

## 🐛 Troubleshooting

### App tidak jalan?
1. Cek logs di Streamlit Cloud
2. Pastikan `requirements.txt` lengkap
3. Pastikan file data ada di repo

### Out of Memory?
- Streamlit Cloud free tier punya limit 1GB RAM
- Optimalkan dengan caching: `@st.cache_data`

### App timeout?
- Loading data terlalu lama
- Gunakan data yang lebih kecil atau pre-processed

## 💡 Tips

1. **Optimize loading:** Gunakan `.pkl` bukan `.csv` untuk file besar
2. **Cache everything:** Gunakan `@st.cache_data` untuk fungsi load data
3. **Compress data:** Gzip file besar sebelum commit
4. **Monitor usage:** Check dashboard Streamlit untuk usage stats

## 🔗 Resources

- [Streamlit Docs](https://docs.streamlit.io)
- [Deploy Guide](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
- [Troubleshooting](https://docs.streamlit.io/knowledge-base)

---

**Happy Deploying! 🌾📊**
