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
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load semua data yang diperlukan"""
    try:
        # Load transformed data
        pkl_path = Path("data/processed/transformed_data.pkl")
        if not pkl_path.exists():
            st.error(f"❌ File tidak ditemukan: {pkl_path}")
            return None, None, None, None
            
        df_transformed = pd.read_pickle(str(pkl_path))
        st.success(f"✅ Data berhasil dimuat: {df_transformed.shape[0]} baris, {df_transformed.shape[1]} kolom")
        
        # Load predictions if exists
        predictions_path = Path("data/predictions/future_predictions.csv")
        if predictions_path.exists():
            df_predictions = pd.read_csv(predictions_path)
        else:
            df_predictions = None
        
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
        
        # Create plot
        fig = px.line(
            df_filtered,
            x='Tahun',
            y=prod_col,
            color='Provinsi',
            title='Trend Produksi Padi',
            labels={prod_col: 'Produksi (ton)', 'Tahun': 'Tahun'},
            markers=True
        )
        
        fig.update_layout(
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)


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
    
    st.plotly_chart(fig, use_container_width=True)


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
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x=weather_var,
        y=prod_col,
        color=color_by,
        title=f'Hubungan {weather_var} dengan Produksi',
        labels={prod_col: 'Produksi (ton)', weather_var: weather_var},
        trendline="ols",
        opacity=0.6
    )
    
    fig.update_layout(height=500)
    
    st.plotly_chart(fig, use_container_width=True)


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
    
    st.plotly_chart(fig, use_container_width=True)


def plot_feature_importance():
    """Plot feature importance"""
    st.header("⭐ Feature Importance")
    
    st.write("Faktor-faktor yang paling berpengaruh terhadap prediksi")
    
    # Load feature importance dari model
    importance_path = Path("models/feature_importance.png")
    
    if importance_path.exists():
        st.image(str(importance_path), use_container_width=True)
    else:
        st.info("Feature importance plot belum tersedia. Jalankan training model terlebih dahulu.")


def plot_prediction_vs_actual():
    """Plot prediction vs actual"""
    st.header("🎯 Prediction vs Actual")
    
    # Load plot
    plot_path = Path("models/prediction_vs_actual.png")
    
    if plot_path.exists():
        st.image(str(plot_path), use_container_width=True)
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
        
        st.dataframe(df_results, use_container_width=True, hide_index=True)
        
        # Model comparison visualization
        st.subheader("Perbandingan Model")
        plot_path = Path("models/model_comparison.png")
        if plot_path.exists():
            st.image(str(plot_path), use_container_width=True)
    else:
        st.info("Hasil evaluasi model belum tersedia. Jalankan training model terlebih dahulu.")


def show_future_predictions(df_predictions):
    """Tampilkan prediksi masa depan"""
    st.header("🔮 Prediksi Produksi Masa Depan")
    
    if df_predictions is not None and len(df_predictions) > 0:
        # Summary metrics dalam format teks
        years = sorted(df_predictions['Tahun'].unique())
        st.markdown(f"""
        ### Ringkasan Prediksi
        
        - **Total Prediksi Produksi:** {df_predictions['Produksi_Prediksi'].sum()/1e6:.2f} juta ton
        - **Rata-rata per Tahun:** {df_predictions.groupby('Tahun')['Produksi_Prediksi'].sum().mean()/1e6:.2f} juta ton
        - **Tahun Prediksi:** {years[0]} - {years[-1]}
        """)
        
        st.markdown("---")
        
        # Time series prediction
        st.subheader("Trend Prediksi Per Provinsi")
        
        provinces = sorted(df_predictions['Provinsi'].unique())
        selected_provinces = st.multiselect(
            "Pilih Provinsi untuk Prediksi",
            provinces,
            default=provinces[:3] if len(provinces) >= 3 else provinces,
            key="pred_provinces"
        )
        
        if selected_provinces:
            df_filtered = df_predictions[df_predictions['Provinsi'].isin(selected_provinces)]
            
            fig = px.line(
                df_filtered,
                x='Tahun',
                y='Produksi_Prediksi',
                color='Provinsi',
                title='Prediksi Produksi Padi',
                labels={'Produksi_Prediksi': 'Produksi Prediksi (ton)', 'Tahun': 'Tahun'},
                markers=True
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("Tabel Prediksi")
        display_df = df_predictions[['Provinsi', 'Tahun', 'Produksi_Prediksi', 
                                     'Produktivitas_Prediksi', 'Luas Panen']].copy()
        display_df['Produksi_Prediksi'] = display_df['Produksi_Prediksi'].round(2)
        display_df['Produktivitas_Prediksi'] = display_df['Produktivitas_Prediksi'].round(2)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
    else:
        st.info("Prediksi belum tersedia. Jalankan script predict.py terlebih dahulu.")


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
    
    st.plotly_chart(fig, use_container_width=True)


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
        show_future_predictions(df_predictions)


if __name__ == "__main__":
    main()
