"""
Predict Module - Batch Prediction untuk Tahun Mendatang
Melakukan prediksi produksi padi untuk tahun-tahun mendatang
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Predictor:
    """Class untuk melakukan prediksi"""
    
    def __init__(self, model_path: str = "models/best_model.pkl"):
        """
        Initialize Predictor
        
        Args:
            model_path: Path ke model yang sudah ditraining
        """
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = None
        self.metadata = None
        
    def load_model(self):
        """Load model dan metadata"""
        logger.info(f"Loading model dari {self.model_path}")
        
        try:
            self.model = joblib.load(self.model_path)
            logger.info("✓ Model berhasil di-load")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # Load feature names
        feature_path = self.model_path.parent / "feature_names.json"
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                self.feature_names = json.load(f)
            logger.info(f"✓ Feature names loaded ({len(self.feature_names)} features)")
        
        # Load metadata
        metadata_path = self.model_path.parent / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"✓ Metadata loaded (Best model: {self.metadata.get('best_model')})")
    
    def prepare_future_data(self, df: pd.DataFrame, future_years: list) -> pd.DataFrame:
        """
        Menyiapkan data untuk prediksi tahun-tahun mendatang
        
        Args:
            df: DataFrame historis
            future_years: List tahun yang akan diprediksi
            
        Returns:
            DataFrame untuk prediksi
        """
        logger.info(f"Menyiapkan data untuk prediksi tahun: {future_years}")
        
        future_data = []
        
        # Get unique provinces
        provinces = df['Provinsi'].unique()
        
        for province in provinces:
            province_data = df[df['Provinsi'] == province].sort_values('Tahun')
            
            # Get latest data for the province
            latest_data = province_data.iloc[-1].copy()
            
            # Get historical averages
            avg_curah_hujan = province_data['Curah hujan'].mean()
            avg_kelembapan = province_data['Kelembapan'].mean()
            avg_suhu = province_data['Suhu rata-rata'].mean()
            avg_luas_panen = province_data['Luas Panen'].mean()
            
            for future_year in future_years:
                future_row = {
                    'Provinsi': province,
                    'Tahun': future_year,
                    'Luas Panen': avg_luas_panen,  # Gunakan rata-rata historis
                    'Curah hujan': avg_curah_hujan,
                    'Kelembapan': avg_kelembapan,
                    'Suhu rata-rata': avg_suhu,
                }
                
                # Add lag features dari data terakhir
                if 'Produksi' in latest_data:
                    future_row['Produksi_Lag1'] = latest_data['Produksi']
                    future_row['Produksi_Lag2'] = latest_data.get('Produksi_Lag1', latest_data['Produksi'])
                
                if 'Produktivitas' in latest_data:
                    future_row['Produktivitas'] = latest_data['Produktivitas']
                    future_row['Produktivitas_Lag1'] = latest_data['Produktivitas']
                
                # Add encoded province
                if 'Provinsi_Encoded' in latest_data:
                    future_row['Provinsi_Encoded'] = latest_data['Provinsi_Encoded']
                
                # Add weather features
                curah_hujan_mean = province_data['Curah hujan'].mean()
                future_row['Curah_Hujan_Anomaly'] = avg_curah_hujan - curah_hujan_mean
                future_row['Comfort_Index'] = (avg_kelembapan / 100) * (30 - abs(avg_suhu - 27))
                
                # Add one-hot encoded features
                for col in df.columns:
                    if col.startswith('Curah_Hujan_') and col not in future_row:
                        future_row[col] = latest_data.get(col, 0)
                
                future_data.append(future_row)
        
        future_df = pd.DataFrame(future_data)
        logger.info(f"✓ Data prediksi disiapkan untuk {len(future_df)} kombinasi provinsi-tahun")
        
        return future_df
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Melakukan prediksi
        
        Args:
            X: DataFrame dengan features
            
        Returns:
            Array dengan hasil prediksi
        """
        if self.model is None:
            raise ValueError("Model belum di-load. Jalankan load_model() terlebih dahulu.")
        
        # Ensure all required features are present
        missing_features = [f for f in self.feature_names if f not in X.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features with default value 0
            for feature in missing_features:
                X[feature] = 0
        
        # Reorder columns to match training data
        X = X[self.feature_names]
        
        # Handle missing values
        X = X.fillna(0)
        
        # Predict
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_future(self, df: pd.DataFrame, future_years: list, 
                      output_path: str = "data/predictions/future_predictions.csv") -> pd.DataFrame:
        """
        Prediksi untuk tahun-tahun mendatang
        
        Args:
            df: DataFrame historis
            future_years: List tahun yang akan diprediksi
            output_path: Path untuk menyimpan hasil prediksi
            
        Returns:
            DataFrame dengan hasil prediksi
        """
        logger.info("=" * 60)
        logger.info("MEMULAI PREDIKSI BATCH")
        logger.info("=" * 60)
        
        # Prepare future data
        future_df = self.prepare_future_data(df, future_years)
        
        # Make predictions
        logger.info("Melakukan prediksi...")
        predictions = self.predict(future_df)
        
        # Add predictions to dataframe
        future_df['Produksi_Prediksi'] = predictions
        
        # Calculate predicted productivity
        future_df['Produktivitas_Prediksi'] = future_df['Produksi_Prediksi'] / future_df['Luas Panen']
        
        logger.info(f"✓ Prediksi selesai untuk {len(predictions)} data points")
        
        # Save predictions
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        future_df.to_csv(output_path, index=False)
        logger.info(f"✓ Prediksi disimpan ke {output_path}")
        
        # Save summary
        self.save_prediction_summary(future_df, output_path.parent)
        
        logger.info("=" * 60)
        logger.info("✓ PREDIKSI BATCH SELESAI")
        logger.info("=" * 60)
        
        return future_df
    
    def save_prediction_summary(self, predictions_df: pd.DataFrame, output_dir: Path):
        """
        Menyimpan ringkasan prediksi
        
        Args:
            predictions_df: DataFrame dengan prediksi
            output_dir: Direktori untuk menyimpan summary
        """
        summary = {
            'timestamp': str(datetime.now()),
            'total_predictions': len(predictions_df),
            'years_predicted': sorted(predictions_df['Tahun'].unique().tolist()),
            'provinces': sorted(predictions_df['Provinsi'].unique().tolist()),
            'total_predicted_production': float(predictions_df['Produksi_Prediksi'].sum()),
            'avg_predicted_production': float(predictions_df['Produksi_Prediksi'].mean()),
            'predictions_by_province': {}
        }
        
        # Summary per provinsi
        for province in predictions_df['Provinsi'].unique():
            province_data = predictions_df[predictions_df['Provinsi'] == province]
            summary['predictions_by_province'][province] = {
                'total_production': float(province_data['Produksi_Prediksi'].sum()),
                'avg_production': float(province_data['Produksi_Prediksi'].mean()),
                'avg_productivity': float(province_data['Produktivitas_Prediksi'].mean())
            }
        
        # Save summary
        summary_path = output_dir / "prediction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Ringkasan prediksi disimpan ke {summary_path}")
    
    def compare_with_actual(self, actual_df: pd.DataFrame, 
                           predicted_df: pd.DataFrame) -> pd.DataFrame:
        """
        Membandingkan prediksi dengan data aktual
        
        Args:
            actual_df: DataFrame dengan data aktual
            predicted_df: DataFrame dengan prediksi
            
        Returns:
            DataFrame dengan perbandingan
        """
        # Merge actual and predicted
        comparison = predicted_df.merge(
            actual_df[['Provinsi', 'Tahun', 'Produksi']],
            on=['Provinsi', 'Tahun'],
            how='left',
            suffixes=('_pred', '_actual')
        )
        
        # Calculate errors
        mask = comparison['Produksi'].notna()
        if mask.any():
            comparison.loc[mask, 'Error'] = comparison.loc[mask, 'Produksi_Prediksi'] - comparison.loc[mask, 'Produksi']
            comparison.loc[mask, 'Absolute_Error'] = np.abs(comparison.loc[mask, 'Error'])
            comparison.loc[mask, 'Percentage_Error'] = (comparison.loc[mask, 'Error'] / comparison.loc[mask, 'Produksi']) * 100
        
        return comparison


def main():
    """Fungsi utama untuk menjalankan prediksi"""
    # Load historical data
    input_path = "data/processed/transformed_data.pkl"
    logger.info(f"Memuat data historis dari {input_path}")
    df = pd.read_pickle(input_path)
    
    # Initialize predictor
    predictor = Predictor(model_path="models/best_model.pkl")
    
    # Load model
    predictor.load_model()
    
    # Define future years to predict
    latest_year = df['Tahun'].max()
    future_years = list(range(int(latest_year) + 1, int(latest_year) + 6))  # Prediksi 5 tahun ke depan
    
    logger.info(f"Tahun terakhir dalam data: {latest_year}")
    logger.info(f"Akan memprediksi untuk tahun: {future_years}")
    
    # Predict future
    predictions_df = predictor.predict_future(
        df, 
        future_years, 
        output_path="data/predictions/future_predictions.csv"
    )
    
    # Display summary
    print("\n" + "=" * 60)
    print("RINGKASAN PREDIKSI")
    print("=" * 60)
    print(f"\nTotal prediksi: {len(predictions_df)} data points")
    print(f"Tahun yang diprediksi: {future_years}")
    print(f"\nTotal produksi prediksi: {predictions_df['Produksi_Prediksi'].sum():,.2f} ton")
    print(f"Rata-rata produksi per provinsi per tahun: {predictions_df['Produksi_Prediksi'].mean():,.2f} ton")
    
    print("\n" + "-" * 60)
    print("PREDIKSI PER PROVINSI (Rata-rata)")
    print("-" * 60)
    
    summary_by_province = predictions_df.groupby('Provinsi').agg({
        'Produksi_Prediksi': 'mean',
        'Produktivitas_Prediksi': 'mean'
    }).round(2)
    
    print(summary_by_province)
    
    print("\n" + "=" * 60)
    
    return predictions_df


if __name__ == "__main__":
    predictions = main()
