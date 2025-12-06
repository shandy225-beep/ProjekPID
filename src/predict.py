"""
Predict Module - Advanced Multi-Feature Prediction
Memprediksi semua fitur secara dinamis menggunakan time series dan ML
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import json
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeatherPredictor:
    """Class untuk memprediksi fitur cuaca menggunakan time series"""
    
    def __init__(self, method='simple'):
        """
        Initialize Weather Predictor
        
        Args:
            method: 'simple' (trend-based forecasting)
        """
        self.method = method
    
    def predict_series(self, series: pd.Series, steps: int) -> np.ndarray:
        """
        Prediksi time series untuk beberapa step ke depan
        
        Args:
            series: Historical time series data
            steps: Number of future steps to predict
            
        Returns:
            Array of predictions
        """
        return self._predict_simple(series, steps)
    
    def _predict_simple(self, series: pd.Series, steps: int) -> np.ndarray:
        """
        Forecasting menggunakan trend, seasonality, dan noise
        Lebih realistis dari rata-rata statis
        """
        series = series.dropna()
        
        # Calculate linear trend
        x = np.arange(len(series))
        z = np.polyfit(x, series.values, 1)  # Linear trend
        trend_slope = z[0]
        
        # Calculate seasonality (if series is long enough)
        if len(series) >= 12:
            # Detrend
            detrended = series - (z[0] * x + z[1])
            # Get seasonal pattern (yearly)
            seasonal_pattern = detrended.tail(12).values
        else:
            seasonal_pattern = np.zeros(12)
        
        # Generate predictions with trend, seasonality and noise
        last_value = series.iloc[-1]
        predictions = []
        
        for i in range(steps):
            # 1. Trend component
            trend_value = last_value + trend_slope * (i + 1)
            
            # 2. Seasonal component
            seasonal_idx = (len(series) + i) % len(seasonal_pattern)
            seasonal_value = seasonal_pattern[seasonal_idx] if len(seasonal_pattern) > 0 else 0
            
            # 3. Add realistic noise (±5% variation)
            noise = np.random.normal(0, abs(trend_value) * 0.05)
            
            predicted = trend_value + seasonal_value + noise
            predictions.append(predicted)
        
        return np.array(predictions)


class AdvancedPredictor:
    """Advanced Predictor dengan multi-feature forecasting"""
    
    def __init__(self, model_path: str = "models/best_model.pkl"):
        """
        Initialize Advanced Predictor
        
        Args:
            model_path: Path ke model produksi
        """
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = None
        self.metadata = None
        self.weather_predictor = WeatherPredictor(method='simple')
        
    def load_model(self):
        """Load production model"""
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
            logger.info(f"✓ Metadata loaded")
    
    def predict_weather_features(self, df: pd.DataFrame, province: str, 
                                future_years: List[int],
                                scenario: str = 'realistic') -> Dict[str, np.ndarray]:
        """
        Prediksi fitur cuaca untuk provinsi tertentu dengan scenario
        
        Args:
            df: Historical data
            province: Nama provinsi
            future_years: List tahun yang akan diprediksi
            scenario: 'pessimistic', 'realistic', or 'optimistic'
            
        Returns:
            Dictionary dengan prediksi untuk setiap fitur cuaca
        """
        province_data = df[df['Provinsi'] == province].sort_values('Tahun')
        steps = len(future_years)
        
        predictions = {}
        
        # Scenario adjustments
        weather_adjustments = {
            'pessimistic': {
                'curah_hujan_mult': 0.90,  # -10% rainfall (drought)
                'kelembapan_mult': 0.95,   # -5% humidity
                'suhu_add': 1.5            # +1.5°C warming
            },
            'realistic': {
                'curah_hujan_mult': 1.0,   # no change
                'kelembapan_mult': 1.0,    # no change
                'suhu_add': 0.5            # +0.5°C slight warming
            },
            'optimistic': {
                'curah_hujan_mult': 1.05,  # +5% rainfall (good)
                'kelembapan_mult': 1.02,   # +2% humidity
                'suhu_add': 0.0            # stable temperature
            }
        }
        
        adj = weather_adjustments.get(scenario, weather_adjustments['realistic'])
        
        # Predict Curah Hujan (mm)
        curah_hujan_series = province_data['Curah hujan']
        predictions['Curah hujan'] = self.weather_predictor.predict_series(
            curah_hujan_series, steps
        ) * adj['curah_hujan_mult']
        
        # Predict Kelembapan (%)
        kelembapan_series = province_data['Kelembapan']
        predictions['Kelembapan'] = self.weather_predictor.predict_series(
            kelembapan_series, steps
        ) * adj['kelembapan_mult']
        
        # Predict Suhu (°C)
        suhu_series = province_data['Suhu rata-rata']
        predictions['Suhu rata-rata'] = self.weather_predictor.predict_series(
            suhu_series, steps
        ) + adj['suhu_add']
        
        # Ensure realistic bounds
        predictions['Curah hujan'] = np.clip(predictions['Curah hujan'], 100, 500)  # mm
        predictions['Kelembapan'] = np.clip(predictions['Kelembapan'], 50, 100)  # %
        predictions['Suhu rata-rata'] = np.clip(predictions['Suhu rata-rata'], 22, 33)  # °C
        
        return predictions
    
    def predict_luas_panen(self, df: pd.DataFrame, province: str, 
                          future_years: List[int],
                          scenario: str = 'realistic') -> np.ndarray:
        """
        Prediksi Luas Panen berdasarkan trend historis dengan scenario
        
        Args:
            df: Historical data
            province: Nama provinsi
            future_years: List tahun yang akan diprediksi
            scenario: 'pessimistic', 'realistic', or 'optimistic'
            
        Returns:
            Array dengan prediksi luas panen (ha)
        """
        province_data = df[df['Provinsi'] == province].sort_values('Tahun')
        luas_panen_series = province_data['Luas Panen']
        
        # Scenario-based growth multipliers
        growth_multiplier = {
            'pessimistic': 0.98,  # -2% per year (declining)
            'realistic': 1.01,    # +1% per year (stable growth)
            'optimistic': 1.03    # +3% per year (strong growth)
        }
        
        multiplier = growth_multiplier.get(scenario, 1.01)
        
        # Calculate growth rate
        if len(luas_panen_series) >= 2:
            last_value = luas_panen_series.iloc[-1]
            predictions = []
            
            for i, year in enumerate(future_years):
                # Apply scenario-based growth
                pred = last_value * (multiplier ** (i + 1))
                # Add small random variation (±1%)
                pred = pred * (1 + np.random.uniform(-0.01, 0.01))
                predictions.append(max(pred, 0))  # Ensure positive
            
            return np.array(predictions)
        else:
            # Fallback: use mean with growth
            mean_val = luas_panen_series.mean()
            return np.array([mean_val * (multiplier ** (i+1)) for i in range(len(future_years))])
    
    def prepare_future_data(self, df: pd.DataFrame, 
                           future_years: List[int],
                           scenario: str = 'realistic') -> pd.DataFrame:
        """
        Menyiapkan data prediksi dengan forecasting semua fitur (with scenario)
        
        Args:
            df: Historical DataFrame
            future_years: List tahun yang akan diprediksi
            scenario: 'pessimistic', 'realistic', or 'optimistic'
            
        Returns:
            DataFrame dengan semua fitur diprediksi
        """
        logger.info(f"🔮 Memprediksi fitur untuk tahun: {future_years} (Scenario: {scenario})")
        
        future_data = []
        provinces = df['Provinsi'].unique()
        
        for province in provinces:
            logger.info(f"  📍 Processing {province}...")
            
            province_data = df[df['Provinsi'] == province].sort_values('Tahun')
            latest_data = province_data.iloc[-1].copy()
            
            # STEP 1: Predict weather features with scenario
            weather_predictions = self.predict_weather_features(
                df, province, future_years, scenario=scenario
            )
            
            # STEP 2: Predict Luas Panen with scenario
            luas_panen_predictions = self.predict_luas_panen(
                df, province, future_years, scenario=scenario
            )
            
            # STEP 3: Build future dataframe
            for i, future_year in enumerate(future_years):
                future_row = {
                    'Provinsi': province,
                    'Tahun': future_year,
                    'Scenario': scenario,
                    'Luas Panen': luas_panen_predictions[i],  # ha
                    'Curah hujan': weather_predictions['Curah hujan'][i],  # mm
                    'Kelembapan': weather_predictions['Kelembapan'][i],  # %
                    'Suhu rata-rata': weather_predictions['Suhu rata-rata'][i],  # °C
                }
                
                # Add lag features (chained from previous predictions)
                if i == 0:
                    # First prediction: use last historical value
                    if 'Produksi' in latest_data:
                        future_row['Produksi_Lag1'] = latest_data['Produksi']
                        future_row['Produksi_Lag2'] = latest_data.get(
                            'Produksi_Lag1', latest_data['Produksi']
                        )
                else:
                    # Use previous prediction
                    prev_row = future_data[-1]
                    future_row['Produksi_Lag1'] = prev_row.get('Produksi_Prediksi', 0)
                    future_row['Produksi_Lag2'] = prev_row.get('Produksi_Lag1', 0)
                
                # Calculate derived features
                curah_hujan_mean = province_data['Curah hujan'].mean()
                future_row['Curah_Hujan_Anomaly'] = (
                    future_row['Curah hujan'] - curah_hujan_mean
                )
                
                future_row['Comfort_Index'] = (
                    (future_row['Kelembapan'] / 100) * 
                    (30 - abs(future_row['Suhu rata-rata'] - 27))
                )
                
                # Add encoded province
                if 'Provinsi_Encoded' in latest_data:
                    future_row['Provinsi_Encoded'] = latest_data['Provinsi_Encoded']
                
                # Add productivity (will be calculated after production prediction)
                if 'Produktivitas' in latest_data:
                    future_row['Produktivitas'] = latest_data['Produktivitas']
                    future_row['Produktivitas_Lag1'] = latest_data['Produktivitas']
                
                # Add one-hot encoded features
                for col in df.columns:
                    if (col.startswith('Curah_Hujan_') or 
                        col.startswith('Kelembapan_') or 
                        col.startswith('Suhu_')) and col not in future_row:
                        future_row[col] = latest_data.get(col, 0)
                
                future_data.append(future_row)
        
        future_df = pd.DataFrame(future_data)
        logger.info(f"✓ Data prediksi disiapkan untuk {len(future_df)} kombinasi")
        
        return future_df
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict production (ton)
        
        Args:
            X: DataFrame dengan features
            
        Returns:
            Array dengan hasil prediksi (ton)
        """
        if self.model is None:
            raise ValueError("Model belum di-load")
        
        # Ensure all features present
        missing_features = [f for f in self.feature_names if f not in X.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features[:5]}...")
            for feature in missing_features:
                X[feature] = 0
        
        # Reorder columns
        X = X[self.feature_names]
        X = X.fillna(0)
        
        # Predict
        predictions = self.model.predict(X)
        return predictions
    
    def predict_future_scenarios(self, df: pd.DataFrame, future_years: List[int],
                                output_dir: str = "data/predictions/") -> Dict[str, pd.DataFrame]:
        """
        Prediksi dengan 3 scenario berbeda
        
        Returns:
            Dictionary dengan 3 dataframe (pessimistic, realistic, optimistic)
        """
        scenarios = ['pessimistic', 'realistic', 'optimistic']
        results = {}
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for scenario in scenarios:
            logger.info(f"\n{'='*70}")
            logger.info(f"PREDIKSI SCENARIO: {scenario.upper()}")
            logger.info(f"{'='*70}")
            
            # Prepare data with scenario
            future_df = self.prepare_future_data(df, future_years, scenario=scenario)
            
            # Predict production
            logger.info("🎯 Memprediksi produksi...")
            predictions = self.predict(future_df)
            future_df['Produksi_Prediksi'] = predictions
            
            # Calculate productivity
            future_df['Produktivitas_Prediksi'] = (
                future_df['Produksi_Prediksi'] / future_df['Luas Panen']
            )
            
            # Save
            csv_path = output_path / f"predictions_{scenario}.csv"
            future_df.to_csv(csv_path, index=False)
            logger.info(f"✓ Prediksi {scenario} disimpan ke {csv_path}")
            
            results[scenario] = future_df
        
        # Save combined
        combined_df = pd.concat(results.values(), ignore_index=True)
        combined_path = output_path / "predictions_all_scenarios.csv"
        combined_df.to_csv(combined_path, index=False)
        logger.info(f"✓ Semua scenario disimpan ke {combined_path}")
        
        # Save backward compatibility file (realistic scenario)
        compat_path = output_path / "future_predictions.csv"
        results['realistic'].to_csv(compat_path, index=False)
        logger.info(f"✓ Backward compatibility file disimpan ke {compat_path}")
        
        return results
    
    def predict_future(self, df: pd.DataFrame, future_years: List[int],
                      output_path: str = "data/predictions/future_predictions.csv") -> pd.DataFrame:
        """
        Prediksi lengkap dengan semua fitur menggunakan time series forecasting
        
        Args:
            df: Historical data
            future_years: Years to predict
            output_path: Output file path
            
        Returns:
            DataFrame with predictions (dengan satuan)
        """
        logger.info("=" * 70)
        logger.info("🚀 MEMULAI PREDIKSI MULTI-FEATURE dengan TIME SERIES")
        logger.info("=" * 70)
        
        # Prepare future data (with predicted features)
        future_df = self.prepare_future_data(df, future_years)
        
        # Predict production
        logger.info("🎯 Memprediksi produksi...")
        predictions = self.predict(future_df)
        future_df['Produksi_Prediksi'] = predictions
        
        # Calculate productivity (ton/ha)
        future_df['Produktivitas_Prediksi'] = (
            future_df['Produksi_Prediksi'] / future_df['Luas Panen']
        )
        
        logger.info(f"✓ Prediksi selesai untuk {len(predictions)} data points")
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        future_df.to_csv(output_path, index=False)
        logger.info(f"✓ Prediksi disimpan ke {output_path}")
        
        # Summary
        self.save_prediction_summary(future_df, output_path.parent)
        
        logger.info("=" * 70)
        logger.info("✅ PREDIKSI MULTI-FEATURE SELESAI")
        logger.info("=" * 70)
        
        return future_df
    
    def save_prediction_summary(self, predictions_df: pd.DataFrame, output_dir: Path):
        """Save prediction summary dengan satuan"""
        summary = {
            'timestamp': str(datetime.now()),
            'method': 'Multi-Feature Time Series Forecasting + ML',
            'total_predictions': len(predictions_df),
            'years_predicted': sorted(predictions_df['Tahun'].unique().tolist()),
            'provinces': sorted(predictions_df['Provinsi'].unique().tolist()),
            'total_predicted_production_ton': float(predictions_df['Produksi_Prediksi'].sum()),
            'avg_predicted_production_ton': float(predictions_df['Produksi_Prediksi'].mean()),
            'feature_ranges': {
                'curah_hujan_mm': {
                    'min': float(predictions_df['Curah hujan'].min()),
                    'max': float(predictions_df['Curah hujan'].max()),
                    'mean': float(predictions_df['Curah hujan'].mean())
                },
                'kelembapan_persen': {
                    'min': float(predictions_df['Kelembapan'].min()),
                    'max': float(predictions_df['Kelembapan'].max()),
                    'mean': float(predictions_df['Kelembapan'].mean())
                },
                'suhu_celcius': {
                    'min': float(predictions_df['Suhu rata-rata'].min()),
                    'max': float(predictions_df['Suhu rata-rata'].max()),
                    'mean': float(predictions_df['Suhu rata-rata'].mean())
                },
                'luas_panen_ha': {
                    'min': float(predictions_df['Luas Panen'].min()),
                    'max': float(predictions_df['Luas Panen'].max()),
                    'mean': float(predictions_df['Luas Panen'].mean())
                }
            },
            'predictions_by_province': {}
        }
        
        # Per province summary
        for province in predictions_df['Provinsi'].unique():
            province_data = predictions_df[predictions_df['Provinsi'] == province]
            summary['predictions_by_province'][province] = {
                'total_production_ton': float(province_data['Produksi_Prediksi'].sum()),
                'avg_production_ton': float(province_data['Produksi_Prediksi'].mean()),
                'avg_productivity_ton_per_ha': float(province_data['Produktivitas_Prediksi'].mean()),
                'avg_curah_hujan_mm': float(province_data['Curah hujan'].mean()),
                'avg_kelembapan_persen': float(province_data['Kelembapan'].mean()),
                'avg_suhu_celcius': float(province_data['Suhu rata-rata'].mean())
            }
        
        # Save
        summary_path = output_dir / "prediction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Summary disimpan ke {summary_path}")


def main():
    """Main function"""
    # Load data
    input_path = "data/processed/transformed_data.csv"
    logger.info(f"📂 Memuat data dari {input_path}")
    
    if Path(input_path).exists():
        df = pd.read_csv(input_path)
    else:
        # Fallback to pickle
        input_path = "data/processed/transformed_data.pkl"
        df = pd.read_pickle(input_path)
    
    # Initialize predictor
    predictor = AdvancedPredictor(model_path="models/best_model.pkl")
    predictor.load_model()
    
    # Define future years
    latest_year = df['Tahun'].max()
    future_years = list(range(int(latest_year) + 1, int(latest_year) + 6))
    
    logger.info(f"📅 Tahun terakhir: {latest_year}")
    logger.info(f"🔮 Prediksi untuk: {future_years}")
    
    # Predict with 3 scenarios
    results = predictor.predict_future_scenarios(df, future_years)
    
    # Display comparison
    print("\n" + "=" * 90)
    print("📊 PERBANDINGAN SCENARIO PREDIKSI")
    print("=" * 90)
    
    for scenario, predictions_df in results.items():
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario.upper()}")
        print(f"{'='*70}")
        print(f"\n🎯 Metode: Time Series Forecasting + Machine Learning")
        print(f"📈 Total prediksi: {len(predictions_df)} data points")
        print(f"📅 Tahun: {future_years}")
        
        print("\n" + "-" * 70)
        print("🌾 PREDIKSI PRODUKSI")
        print("-" * 70)
        print(f"Total produksi prediksi: {predictions_df['Produksi_Prediksi'].sum():,.2f} ton")
        print(f"Rata-rata produksi: {predictions_df['Produksi_Prediksi'].mean():,.2f} ton/tahun")
        print(f"Produktivitas rata-rata: {predictions_df['Produktivitas_Prediksi'].mean():.2f} ton/ha")
    
        print("\n" + "-" * 70)
        print("🌦️  PREDIKSI FITUR CUACA (Rata-rata)")
        print("-" * 70)
        print(f"Curah hujan: {predictions_df['Curah hujan'].mean():.2f} mm "
              f"(range: {predictions_df['Curah hujan'].min():.0f} - {predictions_df['Curah hujan'].max():.0f} mm)")
        print(f"Kelembapan: {predictions_df['Kelembapan'].mean():.2f}% "
              f"(range: {predictions_df['Kelembapan'].min():.0f} - {predictions_df['Kelembapan'].max():.0f}%)")
        print(f"Suhu: {predictions_df['Suhu rata-rata'].mean():.2f}°C "
              f"(range: {predictions_df['Suhu rata-rata'].min():.1f} - {predictions_df['Suhu rata-rata'].max():.1f}°C)")
        print(f"Luas Panen: {predictions_df['Luas Panen'].mean():,.2f} ha "
              f"(range: {predictions_df['Luas Panen'].min():,.0f} - {predictions_df['Luas Panen'].max():,.0f} ha)")
        
        print("\n" + "-" * 70)
        print("📅 VARIASI PREDIKSI PER TAHUN")
        print("-" * 70)
        
        yearly_summary = predictions_df.groupby('Tahun').agg({
            'Produksi_Prediksi': 'sum'
        }).round(2)
        
        print(yearly_summary.to_string())
    
    print("\n" + "=" * 90)
    print("✅ SEMUA PREDIKSI TERSIMPAN")
    print("=" * 90)
    print("📁 data/predictions/predictions_pessimistic.csv")
    print("📁 data/predictions/predictions_realistic.csv")
    print("📁 data/predictions/predictions_optimistic.csv")
    print("📁 data/predictions/predictions_all_scenarios.csv (combined)")
    print("📁 data/predictions/future_predictions.csv (realistic - backward compatible)")
    print("=" * 90 + "\n")
    
    return results


if __name__ == "__main__":
    predictions = main()
