"""
Export Dashboard ke HTML Statis
Script untuk generate HTML report yang bisa dibuka tanpa Streamlit server
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path

# Load data
print("Loading data...")
df = pd.read_csv('data/processed/transformed_data.csv')
predictions = pd.read_csv('data/predictions/future_predictions.csv')

with open('models/evaluation_results.json', 'r') as f:
    eval_results = json.load(f)

# Create HTML
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Dashboard Prediksi Hasil Panen Padi Sumatera</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2e7d32;
            text-align: center;
        }
        h2 {
            color: #1976d2;
            margin-top: 30px;
        }
        .section {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric {
            display: inline-block;
            padding: 15px;
            margin: 10px;
            background: #e3f2fd;
            border-radius: 5px;
            text-align: center;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #1976d2;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #1976d2;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
    </style>
</head>
<body>
    <h1>🌾 Dashboard Prediksi Hasil Panen Padi Sumatera</h1>
"""

# Overview Metrics
print("Creating overview metrics...")
html_content += """
    <div class="section">
        <h2>📊 Overview</h2>
"""

total_rows = len(df)
total_provinsi = df['Provinsi'].nunique()
tahun_min = int(df['Tahun'].min())
tahun_max = int(df['Tahun'].max())
total_produksi = df['Produksi'].sum() / 1_000_000  # dalam juta ton

html_content += f"""
        <div class="metric">
            <div class="metric-value">{total_rows}</div>
            <div class="metric-label">Total Data Points</div>
        </div>
        <div class="metric">
            <div class="metric-value">{total_provinsi}</div>
            <div class="metric-label">Provinsi</div>
        </div>
        <div class="metric">
            <div class="metric-value">{tahun_min}-{tahun_max}</div>
            <div class="metric-label">Rentang Tahun</div>
        </div>
        <div class="metric">
            <div class="metric-value">{total_produksi:.1f}M</div>
            <div class="metric-label">Total Produksi (ton)</div>
        </div>
    </div>
"""

# Time Series Plot
print("Creating time series plot...")
fig_ts = go.Figure()
for provinsi in df['Provinsi'].unique()[:5]:  # Top 5 provinsi
    prov_data = df[df['Provinsi'] == provinsi].sort_values('Tahun')
    fig_ts.add_trace(go.Scatter(
        x=prov_data['Tahun'],
        y=prov_data['Produksi'],
        mode='lines+markers',
        name=provinsi
    ))

fig_ts.update_layout(
    title='Trend Produksi Padi per Provinsi (1993-2020)',
    xaxis_title='Tahun',
    yaxis_title='Produksi (ton)',
    hovermode='x unified',
    height=500
)

html_content += """
    <div class="section">
        <h2>📈 Time Series Analysis</h2>
        <div id="timeseries"></div>
    </div>
"""

# Model Performance Table
print("Creating model performance table...")
html_content += """
    <div class="section">
        <h2>🤖 Model Performance</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>R² Score</th>
                <th>RMSE</th>
                <th>MAE</th>
                <th>MAPE (%)</th>
            </tr>
"""

for model_name, metrics in eval_results.items():
    html_content += f"""
            <tr>
                <td>{model_name}</td>
                <td>{metrics['r2_score']:.4f}</td>
                <td>{metrics['rmse']:,.0f}</td>
                <td>{metrics['mae']:,.0f}</td>
                <td>{metrics['mape']:.2f}</td>
            </tr>
"""

html_content += """
        </table>
    </div>
"""

# Future Predictions
print("Creating predictions plot...")
fig_pred = go.Figure()
for provinsi in predictions['Provinsi'].unique()[:5]:
    pred_data = predictions[predictions['Provinsi'] == provinsi].sort_values('Tahun')
    fig_pred.add_trace(go.Scatter(
        x=pred_data['Tahun'],
        y=pred_data['Produksi_Prediksi'],
        mode='lines+markers',
        name=provinsi
    ))

fig_pred.update_layout(
    title='Prediksi Produksi Padi 5 Tahun Ke Depan',
    xaxis_title='Tahun',
    yaxis_title='Produksi Prediksi (ton)',
    height=500
)

html_content += """
    <div class="section">
        <h2>🔮 Future Predictions</h2>
        <div id="predictions"></div>
    </div>
"""

# Province Comparison
print("Creating province comparison...")
prod_by_prov = df.groupby('Provinsi')['Produksi'].mean().sort_values(ascending=False)
fig_prov = go.Figure(data=[
    go.Bar(x=prod_by_prov.index, y=prod_by_prov.values)
])
fig_prov.update_layout(
    title='Rata-rata Produksi per Provinsi',
    xaxis_title='Provinsi',
    yaxis_title='Produksi Rata-rata (ton)',
    height=500
)

html_content += """
    <div class="section">
        <h2>📊 Province Comparison</h2>
        <div id="province"></div>
    </div>
"""

# Add Plotly scripts
html_content += f"""
    <script>
        Plotly.newPlot('timeseries', {fig_ts.to_json()});
        Plotly.newPlot('predictions', {fig_pred.to_json()});
        Plotly.newPlot('province', {fig_prov.to_json()});
    </script>
"""

html_content += """
    <footer style="text-align: center; margin-top: 50px; color: #666;">
        <p>Dashboard generated from ProjekPID - Rice Production Prediction Pipeline</p>
        <p>Data: 1993-2020 | Predictions: 2021-2025</p>
    </footer>
</body>
</html>
"""

# Save HTML
output_path = Path('dashboard_export.html')
output_path.write_text(html_content)
print(f"\n✓ Dashboard berhasil di-export ke: {output_path.absolute()}")
print(f"✓ Buka file ini di browser untuk melihat visualisasi!")
