"""
Dashboard Streamlit - Interactive Data Visualization
Dashboard interaktif untuk visualisasi data dan prediksi hasil panen padi
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Page config
st.set_page_config(
    page_title="Dashboard Prediksi Hasil Panen Padi",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Fix metric box styling */
    [data-testid="stMetricValue"] {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #1b5e20 !important;
        font-size: 1.5rem !important;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        color: #2e7d32 !important;
        font-weight: 600;
        font-size: 1rem;
    }
    [data-testid="stMetricDelta"] {
        color: #4caf50 !important;
    }
    div[data-testid="metric-container"] {
        background-color: #f1f8f4;
        border: 1px solid #a5d6a7;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)  # Cache 5 minutes, then auto-reload
def load_data():
    """Load semua data yang diperlukan"""
    try:
        # Load transformed data - try CSV first, then PKL
        csv_path = Path("data/processed/transformed_data.csv")
        pkl_path = Path("data/processed/transformed_data.pkl")
        
        if csv_path.exists():
            df_transformed = pd.read_csv(str(csv_path))
            st.success(f"✅ Data berhasil dimuat: {df_transformed.shape[0]} baris, {df_transformed.shape[1]} kolom")
        elif pkl_path.exists():
            df_transformed = pd.read_pickle(str(pkl_path))
            st.success(f"✅ Data berhasil dimuat: {df_transformed.shape[0]} baris, {df_transformed.shape[1]} kolom")
        else:
            st.error(f"❌ File tidak ditemukan: {csv_path} atau {pkl_path}")
            return None, None, None, None
        
        # Load predictions (multi-scenario)
        predictions = {}
        scenarios = ['pessimistic', 'realistic', 'optimistic']
        
        for scenario in scenarios:
            pred_path = Path(f"data/predictions/predictions_{scenario}.csv")
            if pred_path.exists():
                predictions[scenario] = pd.read_csv(pred_path)
                st.info(f"✅ Loaded {scenario} scenario: {predictions[scenario].shape[0]} predictions")
        
        # Fallback: backward compatible
        if not predictions:
            predictions_path = Path("data/predictions/future_predictions.csv")
            if predictions_path.exists():
                predictions['realistic'] = pd.read_csv(predictions_path)
                st.warning("⚠️ Using legacy single-scenario file")
        
        df_predictions = predictions if predictions else None
        
        # Show what was loaded
        if predictions:
            st.success(f"✅ Multi-scenario data loaded: {len(predictions)} scenarios")
        else:
            st.error("❌ No prediction data found")
        
        # Load model results
        results_path = Path("models/evaluation_results.json")
        if results_path.exists():
            with open(results_path, 'r') as f:
                model_results = json.load(f)
        else:
            model_results = None
        
        # Load metadata
        metadata_path = Path("models/metadata.json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = None
        
        return df_transformed, df_predictions, model_results, metadata
    
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None


def show_overview(df):
    """Tampilkan overview data"""
    st.header("📊 Overview Data")
    
    if df is None or len(df) == 0:
        st.warning("⚠️ Data tidak tersedia.")
        return
    
    try:
        # Tampilkan informasi dataset dalam format teks
        st.markdown(f"""
        ### Informasi Dataset
        
        - **Total Data Points:** {len(df):,} observasi
        - **Jumlah Provinsi:** {df['Provinsi'].nunique()} provinsi
        - **Rentang Tahun:** {int(df['Tahun'].min())} - {int(df['Tahun'].max())}
        - **Total Produksi:** {df['Produksi'].sum()/1e6:.2f} juta ton
        """)
        
    except Exception as e:
        st.error(f"❌ Error menampilkan overview: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def plot_time_series(df):
    """Plot time series produksi per provinsi"""
    st.header("📈 Trend Produksi Padi Per Provinsi (1993-2020)")
    
    # Pilih provinsi
    provinces = sorted(df['Provinsi'].unique())
    selected_provinces = st.multiselect(
        "Pilih Provinsi",
        provinces,
        default=provinces[:3] if len(provinces) >= 3 else provinces
    )
    
    if selected_provinces:
        # Filter data
        df_filtered = df[df['Provinsi'].isin(selected_provinces)]
        
        # Gunakan kolom original jika ada
        prod_col = 'Produksi_Original' if 'Produksi_Original' in df.columns else 'Produksi'
        
        # Create plot dengan satuan
        fig = px.line(
            df_filtered,
            x='Tahun',
            y=prod_col,
            color='Provinsi',
            title='Trend Produksi Padi (Historis)',
            labels={prod_col: 'Produksi (ton)', 'Tahun': 'Tahun'},
            markers=True
        )
        
        fig.update_layout(
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, width='stretch')


def plot_correlation_heatmap(df):
    """Plot heatmap korelasi"""
    st.header("🔥 Heatmap Korelasi")
    
    st.write("Korelasi antara variabel cuaca dan produksi")
    
    # Pilih kolom numerik yang relevan
    numeric_cols = ['Produksi', 'Luas Panen', 'Produktivitas', 
                   'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
    
    # Filter kolom yang ada
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    # Hitung korelasi
    corr_matrix = df[available_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Correlation Matrix',
        height=600,
        xaxis={'side': 'bottom'}
    )
    
    st.plotly_chart(fig, width='stretch')


def plot_scatter_weather(df):
    """Plot scatter hubungan cuaca dengan produksi"""
    st.header("🌦️ Hubungan Faktor Cuaca dengan Hasil Panen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        weather_var = st.selectbox(
            "Pilih Variabel Cuaca",
            ['Curah hujan', 'Kelembapan', 'Suhu rata-rata']
        )
    
    with col2:
        color_by = st.selectbox(
            "Warna Berdasarkan",
            ['Provinsi', 'Tahun', None]
        )
    
    prod_col = 'Produksi_Original' if 'Produksi_Original' in df.columns else 'Produksi'
    
    # Satuan untuk label
    satuan_map = {
        'Curah hujan': 'mm',
        'Kelembapan': '%',
        'Suhu rata-rata': '°C'
    }
    satuan = satuan_map.get(weather_var, '')
    
    # Create scatter plot (trendline removed to avoid statsmodels dependency)
    fig = px.scatter(
        df,
        x=weather_var,
        y=prod_col,
        color=color_by,
        title=f'Hubungan {weather_var} dengan Produksi',
        labels={prod_col: 'Produksi (ton)', weather_var: f'{weather_var} ({satuan})'},
        opacity=0.6
    )
    
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, width='stretch')


def plot_province_comparison(df):
    """Plot perbandingan antar provinsi"""
    st.header("📊 Perbandingan Produksi Antar Provinsi")
    
    # Agregat data per provinsi
    prod_col = 'Produksi_Original' if 'Produksi_Original' in df.columns else 'Produksi'
    
    province_summary = df.groupby('Provinsi').agg({
        prod_col: 'mean',
        'Produktivitas': 'mean',
        'Luas Panen': 'mean'
    }).reset_index()
    
    province_summary = province_summary.sort_values(prod_col, ascending=True)
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=province_summary['Provinsi'],
        x=province_summary[prod_col],
        orientation='h',
        marker=dict(
            color=province_summary[prod_col],
            colorscale='Greens',
            showscale=True,
            colorbar=dict(title="Produksi")
        ),
        text=province_summary[prod_col].round(0),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Produksi: %{x:.0f} ton<extra></extra>'
    ))
    
    fig.update_layout(
        title='Rata-rata Produksi Per Provinsi',
        xaxis_title='Produksi Rata-rata (ton)',
        yaxis_title='Provinsi',
        height=600,
        showlegend=False
    )
    
    st.plotly_chart(fig, width='stretch')


def plot_feature_importance():
    """Plot feature importance"""
    st.header("⭐ Feature Importance")
    
    st.write("Faktor-faktor yang paling berpengaruh terhadap prediksi")
    
    # Load feature importance dari model
    importance_path = Path("models/feature_importance.png")
    
    if importance_path.exists():
        st.image(str(importance_path), width='stretch')
    else:
        st.info("Feature importance plot belum tersedia. Jalankan training model terlebih dahulu.")


def plot_prediction_vs_actual():
    """Plot prediction vs actual"""
    st.header("🎯 Prediction vs Actual")
    
    # Load plot
    plot_path = Path("models/prediction_vs_actual.png")
    
    if plot_path.exists():
        st.image(str(plot_path), width='stretch')
    else:
        st.info("Prediction vs Actual plot belum tersedia. Jalankan training model terlebih dahulu.")


def show_model_performance(model_results, metadata):
    """Tampilkan performa model"""
    st.header("🤖 Performa Model Machine Learning")
    
    if model_results:
        # Best model info
        if metadata and 'best_model' in metadata:
            st.success(f"✅ Best Model: **{metadata['best_model']}**")
        
        # Create comparison table
        df_results = pd.DataFrame(model_results).T
        df_results = df_results.reset_index()
        
        # Rename the index column to Model
        df_results = df_results.rename(columns={'index': 'Model'})
        
        # Select and order columns
        column_mapping = {
            'model_name': 'Model Name',
            'r2_score': 'R² Score',
            'rmse': 'RMSE',
            'mae': 'MAE',
            'mape': 'MAPE',
            'cv_r2_mean': 'CV R² Mean',
            'cv_r2_std': 'CV R² Std'
        }
        
        # Build the final columns list
        final_cols = ['Model']
        for old_col, new_col in column_mapping.items():
            if old_col in df_results.columns:
                df_results = df_results.rename(columns={old_col: new_col})
                final_cols.append(new_col)
        
        # Select only the columns we have
        df_results = df_results[final_cols]
        
        # Format numeric columns
        numeric_cols = ['R² Score', 'RMSE', 'MAE', 'MAPE', 'CV R² Mean', 'CV R² Std']
        for col in numeric_cols:
            if col in df_results.columns:
                df_results[col] = df_results[col].apply(lambda x: f"{float(x):.4f}" if 'R²' in col else f"{float(x):,.2f}")
        
        st.dataframe(df_results, width='stretch', hide_index=True)
        
        # Model comparison visualization
        st.subheader("Perbandingan Model")
        plot_path = Path("models/model_comparison.png")
        if plot_path.exists():
            st.image(str(plot_path), width='stretch')
    else:
        st.info("Hasil evaluasi model belum tersedia. Jalankan training model terlebih dahulu.")


def show_future_predictions(df_predictions, df_historical=None):
    """Tampilkan prediksi masa depan dengan multi-scenario"""
    st.header("🔮 Prediksi Produksi Masa Depan (Multi-Scenario Forecasting)")
    
    # Check if predictions is a dict (multi-scenario) or DataFrame (legacy)
    if df_predictions is None:
        st.warning("⚠️ Tidak ada data prediksi yang tersedia")
        return
    
    # Handle both dict (multi-scenario) and DataFrame (legacy) formats
    if isinstance(df_predictions, dict):
        scenarios = list(df_predictions.keys())
        
        st.info(f"📊 Tersedia {len(scenarios)} scenario: {', '.join(scenarios)}")
        
        # Scenario selector
        selected_scenario = st.selectbox(
            "🎯 Pilih Scenario",
            scenarios,
            index=scenarios.index('realistic') if 'realistic' in scenarios else 0,
            help="Pilih scenario untuk melihat prediksi"
        )
        
        df_pred = df_predictions[selected_scenario]
    else:
        # Legacy: single DataFrame
        df_pred = df_predictions
        selected_scenario = "realistic"
        st.info("📊 Mode: Single scenario (realistic)")
    
    if df_pred is not None and len(df_pred) > 0:
        # Summary metrics dengan satuan
        years = sorted(df_pred['Tahun'].unique())
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total Produksi",
                f"{df_pred['Produksi_Prediksi'].sum()/1e6:.2f} juta ton",
                help=f"Total produksi prediksi untuk scenario {selected_scenario}"
            )
        with col2:
            st.metric(
                "Rata-rata/Tahun",
                f"{df_pred.groupby('Tahun')['Produksi_Prediksi'].sum().mean()/1e6:.2f} juta ton",
                help="Rata-rata produksi per tahun"
            )
        with col3:
            st.metric(
                "Produktivitas",
                f"{df_pred['Produktivitas_Prediksi'].mean():.2f} ton/ha",
                help="Rata-rata produktivitas"
            )
        with col4:
            st.metric(
                "Tahun Prediksi",
                f"{years[0]} - {years[-1]}",
                help="Rentang tahun prediksi"
            )
        
        st.markdown("---")
        
        # Tabs untuk berbagai visualisasi
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Produksi", 
            "🌦️ Cuaca", 
            "📊 Comparison",
            "🎯 Scenario Compare",
            "📋 Data Lengkap"
        ])
        
        with tab1:
            st.subheader("Trend Prediksi Produksi Per Provinsi")
            
            provinces = sorted(df_pred['Provinsi'].unique())
            selected_provinces = st.multiselect(
                "Pilih Provinsi",
                provinces,
                default=provinces[:3] if len(provinces) >= 3 else provinces,
                key="pred_prod_provinces"
            )
            
            if selected_provinces:
                df_filtered = df_pred[df_pred['Provinsi'].isin(selected_provinces)]
                
                fig = px.line(
                    df_filtered,
                    x='Tahun',
                    y='Produksi_Prediksi',
                    color='Provinsi',
                    title='Prediksi Produksi Padi (ton)',
                    labels={'Produksi_Prediksi': 'Produksi (ton)', 'Tahun': 'Tahun'},
                    markers=True
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, width='stretch')
        
        with tab2:
            st.subheader("Prediksi Fitur Cuaca")
            st.info("💡 Fitur cuaca diprediksi menggunakan time series forecasting (trend + seasonality)")
            
            provinces_weather = sorted(df_pred['Provinsi'].unique())
            selected_prov_weather = st.selectbox(
                "Pilih Provinsi untuk Analisis Cuaca",
                provinces_weather,
                key="weather_prov"
            )
            
            if selected_prov_weather:
                df_prov = df_pred[df_pred['Provinsi'] == selected_prov_weather]
                
                # Create subplot untuk 3 fitur cuaca
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=(
                        'Curah Hujan (mm)', 
                        'Kelembapan (%)', 
                        'Suhu Rata-rata (°C)'
                    ),
                    vertical_spacing=0.1
                )
                
                # Curah Hujan
                fig.add_trace(
                    go.Scatter(
                        x=df_prov['Tahun'],
                        y=df_prov['Curah hujan'],
                        mode='lines+markers',
                        name='Curah Hujan',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
                
                # Kelembapan
                fig.add_trace(
                    go.Scatter(
                        x=df_prov['Tahun'],
                        y=df_prov['Kelembapan'],
                        mode='lines+markers',
                        name='Kelembapan',
                        line=dict(color='green')
                    ),
                    row=2, col=1
                )
                
                # Suhu
                fig.add_trace(
                    go.Scatter(
                        x=df_prov['Tahun'],
                        y=df_prov['Suhu rata-rata'],
                        mode='lines+markers',
                        name='Suhu',
                        line=dict(color='red')
                    ),
                    row=3, col=1
                )
                
                fig.update_xaxes(title_text="Tahun", row=3, col=1)
                fig.update_yaxes(title_text="mm", row=1, col=1)
                fig.update_yaxes(title_text="%", row=2, col=1)
                fig.update_yaxes(title_text="°C", row=3, col=1)
                
                fig.update_layout(height=800, showlegend=False)
                st.plotly_chart(fig, width='stretch')
                
                # Weather statistics
                st.markdown("**Statistik Cuaca Prediksi:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Curah Hujan",
                        f"{df_prov['Curah hujan'].mean():.1f} mm",
                        f"±{df_prov['Curah hujan'].std():.1f} mm"
                    )
                with col2:
                    st.metric(
                        "Kelembapan",
                        f"{df_prov['Kelembapan'].mean():.1f}%",
                        f"±{df_prov['Kelembapan'].std():.1f}%"
                    )
                with col3:
                    st.metric(
                        "Suhu",
                        f"{df_prov['Suhu rata-rata'].mean():.1f}°C",
                        f"±{df_prov['Suhu rata-rata'].std():.1f}°C"
                    )
        
        with tab3:
            st.subheader("Comparison: Historis vs Prediksi")
            
            if df_historical is not None:
                provinces_comp = sorted(df_pred['Provinsi'].unique())
                selected_prov_comp = st.selectbox(
                    "Pilih Provinsi untuk Comparison",
                    provinces_comp,
                    key="comp_prov"
                )
                
                if selected_prov_comp:
                    # Historical data
                    df_hist = df_historical[df_historical['Provinsi'] == selected_prov_comp]
                    # Use original values if available
                    if 'Produksi_Original' in df_hist.columns:
                        df_hist = df_hist[['Tahun', 'Produksi_Original']].copy()
                        df_hist.columns = ['Tahun', 'Produksi']
                    else:
                        df_hist = df_hist[['Tahun', 'Produksi']].copy()
                    
                    # Prediction data
                    df_pred_comp = df_pred[df_pred['Provinsi'] == selected_prov_comp][['Tahun', 'Produksi_Prediksi']].copy()
                    df_pred_comp.columns = ['Tahun', 'Produksi']
                    
                    # Combine
                    df_hist['Type'] = 'Historis'
                    df_pred_comp['Type'] = f'Prediksi ({selected_scenario})'
                    df_combined = pd.concat([df_hist, df_pred_comp])
                    
                    # Plot
                    fig = px.line(
                        df_combined,
                        x='Tahun',
                        y='Produksi',
                        color='Type',
                        title=f'Historis vs Prediksi - {selected_prov_comp}',
                        labels={'Produksi': 'Produksi (ton)', 'Tahun': 'Tahun'},
                        markers=True
                    )
                    
                    # Add vertical line to separate historis and prediksi
                    last_hist_year = df_hist['Tahun'].max()
                    fig.add_vline(
                        x=last_hist_year, 
                        line_dash="dash", 
                        line_color="gray",
                        annotation_text="Data Historis | Prediksi"
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, width='stretch')
                    
                    # Statistics comparison
                    st.markdown("**Perbandingan Statistik:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Historis:**")
                        st.write(f"Rata-rata: {df_hist['Produksi'].mean():,.0f} ton")
                        st.write(f"Std Dev: {df_hist['Produksi'].std():,.0f} ton")
                        st.write(f"Min: {df_hist['Produksi'].min():,.0f} ton")
                        st.write(f"Max: {df_hist['Produksi'].max():,.0f} ton")
                    with col2:
                        st.markdown(f"**Prediksi ({selected_scenario}):**")
                        st.write(f"Rata-rata: {df_pred_comp['Produksi'].mean():,.0f} ton")
                        st.write(f"Std Dev: {df_pred_comp['Produksi'].std():,.0f} ton")
                        st.write(f"Min: {df_pred_comp['Produksi'].min():,.0f} ton")
                        st.write(f"Max: {df_pred_comp['Produksi'].max():,.0f} ton")
            else:
                st.warning("Data historis tidak tersedia untuk comparison")
        
        with tab4:
            st.subheader("Perbandingan Semua Scenario")
            
            # Only show if multi-scenario data available
            if isinstance(df_predictions, dict) and len(df_predictions) > 1:
                provinces_scenario = sorted(df_pred['Provinsi'].unique())
                selected_prov_scenario = st.selectbox(
                    "Pilih Provinsi untuk Perbandingan Scenario",
                    provinces_scenario,
                    key="scenario_prov"
                )
                
                if selected_prov_scenario:
                    # Collect data from all scenarios
                    scenario_data = []
                    for scenario, df_scenario in df_predictions.items():
                        df_temp = df_scenario[df_scenario['Provinsi'] == selected_prov_scenario][['Tahun', 'Produksi_Prediksi']].copy()
                        df_temp['Scenario'] = scenario
                        scenario_data.append(df_temp)
                    
                    df_all_scenarios = pd.concat(scenario_data, ignore_index=True)
                    
                    # Plot comparison
                    fig = px.line(
                        df_all_scenarios,
                        x='Tahun',
                        y='Produksi_Prediksi',
                        color='Scenario',
                        title=f'Perbandingan Scenario - {selected_prov_scenario}',
                        labels={'Produksi_Prediksi': 'Produksi (ton)', 'Tahun': 'Tahun'},
                        markers=True,
                        color_discrete_map={
                            'pessimistic': '#d32f2f',
                            'realistic': '#1976d2',
                            'optimistic': '#388e3c'
                        }
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, width='stretch')
                    
                    # Summary statistics per scenario
                    st.markdown("**Statistik Per Scenario:**")
                    stats_cols = st.columns(len(df_predictions))
                    
                    for i, (scenario, df_scenario) in enumerate(df_predictions.items()):
                        df_s = df_scenario[df_scenario['Provinsi'] == selected_prov_scenario]
                        with stats_cols[i]:
                            st.markdown(f"**{scenario.capitalize()}**")
                            st.metric(
                                "Total Produksi",
                                f"{df_s['Produksi_Prediksi'].sum()/1e3:.1f}K ton"
                            )
                            st.metric(
                                "Pertumbuhan",
                                f"{((df_s['Produksi_Prediksi'].iloc[-1] / df_s['Produksi_Prediksi'].iloc[0]) - 1) * 100:.1f}%"
                            )
            else:
                st.info("📊 Multi-scenario data tidak tersedia. Hanya ada single scenario.")
        
        with tab5:
            st.subheader(f"Tabel Prediksi Lengkap - {selected_scenario.capitalize()}")
            
            # Select columns that exist
            cols = ['Provinsi', 'Tahun', 'Produksi_Prediksi', 'Produktivitas_Prediksi', 
                    'Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
            
            # Add Scenario column if exists
            if 'Scenario' in df_pred.columns:
                cols.insert(2, 'Scenario')
            
            display_df = df_pred[cols].copy()
            
            # Rename dengan satuan
            rename_dict = {
                'Provinsi': 'Provinsi',
                'Tahun': 'Tahun',
                'Produksi_Prediksi': 'Produksi (ton)',
                'Produktivitas_Prediksi': 'Produktivitas (ton/ha)',
                'Luas Panen': 'Luas Panen (ha)',
                'Curah hujan': 'Curah Hujan (mm)',
                'Kelembapan': 'Kelembapan (%)',
                'Suhu rata-rata': 'Suhu (°C)'
            }
            if 'Scenario' in display_df.columns:
                rename_dict['Scenario'] = 'Scenario'
            
            display_df = display_df.rename(columns=rename_dict)
            
            # Round numbers
            display_df['Produksi (ton)'] = display_df['Produksi (ton)'].round(2)
            display_df['Produktivitas (ton/ha)'] = display_df['Produktivitas (ton/ha)'].round(2)
            display_df['Luas Panen (ha)'] = display_df['Luas Panen (ha)'].round(2)
            display_df['Curah Hujan (mm)'] = display_df['Curah Hujan (mm)'].round(1)
            display_df['Kelembapan (%)'] = display_df['Kelembapan (%)'].round(1)
            display_df['Suhu (°C)'] = display_df['Suhu (°C)'].round(1)
            
            st.dataframe(display_df, width='stretch', hide_index=True)
            
            # Download button
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download CSV ({selected_scenario})",
                data=csv,
                file_name=f"prediksi_padi_sumatera_{selected_scenario}.csv",
                mime="text/csv"
            )
        
    else:
        st.info("⚠️ Prediksi belum tersedia. Jalankan script predict.py terlebih dahulu.")


def plot_geographic_map(df):
    """Plot visualisasi geografis (simplified)"""
    st.header("🗺️ Visualisasi Produktivitas Per Provinsi")
    
    # Agregat produktivitas per provinsi
    productivity_summary = df.groupby('Provinsi')['Produktivitas'].mean().reset_index()
    productivity_summary = productivity_summary.sort_values('Produktivitas', ascending=False)
    
    # Create bar chart (karena tidak ada koordinat geografis)
    fig = px.bar(
        productivity_summary,
        x='Provinsi',
        y='Produktivitas',
        title='Produktivitas Rata-rata Per Provinsi',
        labels={'Produktivitas': 'Produktivitas (ton/ha)'},
        color='Produktivitas',
        color_continuous_scale='Greens'
    )
    
    fig.update_layout(
        height=500,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, width='stretch')


def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🌾 Dashboard Prediksi Hasil Panen Padi Sumatera</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        df, df_predictions, model_results, metadata = load_data()
    
    # Debug: Show what we got
    if df is None:
        st.error("❌ Gagal memuat data. Pastikan pipeline ETL sudah dijalankan.")
        st.info("💡 Jalankan: `python run_pipeline.py` untuk memproses data.")
        st.stop()
    
    # Additional validation
    if not isinstance(df, pd.DataFrame):
        st.error(f"❌ Data bukan DataFrame yang valid. Type: {type(df)}")
        st.stop()
        
    if len(df) == 0:
        st.error("❌ DataFrame kosong!")
        st.stop()
    
    # Sidebar
    st.sidebar.title("⚙️ Navigasi")
    page = st.sidebar.radio(
        "Pilih Halaman",
        [
            "📊 Overview",
            "📈 Time Series Analysis",
            "🔥 Correlation Analysis",
            "🌦️ Weather Impact",
            "📊 Province Comparison",
            "🗺️ Geographic Visualization",
            "🤖 Model Performance",
            "🔮 Future Predictions"
        ]
    )
    
    st.sidebar.markdown("---")
    
    # Add clear cache button
    if st.sidebar.button("🔄 Clear Cache & Reload Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Debug info in sidebar
    with st.sidebar.expander("🔍 Debug Info"):
        st.write(f"**Data Shape:** {df.shape}")
        if df_predictions:
            if isinstance(df_predictions, dict):
                st.write(f"**Predictions Type:** Multi-scenario")
                st.write(f"**Scenarios:** {list(df_predictions.keys())}")
                for scenario, pred_df in df_predictions.items():
                    st.write(f"  - {scenario}: {pred_df.shape[0]} rows")
            else:
                st.write(f"**Predictions Type:** Single scenario")
                st.write(f"**Predictions Shape:** {df_predictions.shape}")
        else:
            st.write("**Predictions:** None")
        st.rerun()
    
    st.sidebar.info(
        """
        **Tentang Dashboard:**
        
        Dashboard ini menampilkan analisis dan prediksi hasil panen padi 
        di Sumatera berdasarkan data historis dan faktor cuaca.
        
        **Data Source:** Kaggle Dataset - Data Tanaman Padi Sumatera
        """
    )
    
    # Main content
    if page == "📊 Overview":
        show_overview(df)
        st.markdown("---")
        
        # Quick stats
        st.subheader("Statistik Ringkas")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Produksi**")
            prod_col = 'Produksi_Original' if 'Produksi_Original' in df.columns else 'Produksi'
            st.write(df[prod_col].describe())
        
        with col2:
            st.write("**Produktivitas**")
            st.write(df['Produktivitas'].describe())
    
    elif page == "📈 Time Series Analysis":
        plot_time_series(df)
    
    elif page == "🔥 Correlation Analysis":
        plot_correlation_heatmap(df)
        st.markdown("---")
        plot_feature_importance()
    
    elif page == "🌦️ Weather Impact":
        plot_scatter_weather(df)
    
    elif page == "📊 Province Comparison":
        plot_province_comparison(df)
    
    elif page == "🗺️ Geographic Visualization":
        plot_geographic_map(df)
    
    elif page == "🤖 Model Performance":
        show_model_performance(model_results, metadata)
        st.markdown("---")
        plot_prediction_vs_actual()
    
    elif page == "🔮 Future Predictions":
        show_future_predictions(df_predictions, df)


if __name__ == "__main__":
    main()
