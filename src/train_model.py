"""
Train Model - Training Multiple Regression Models
Melatih berbagai model regresi untuk prediksi hasil panen
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class untuk training dan evaluasi model"""
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'Produksi'):
        """
        Initialize ModelTrainer
        
        Args:
            df: DataFrame dengan features dan target
            target_col: Nama kolom target
        """
        self.df = df.copy()
        self.target_col = target_col
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
    def prepare_features(self) -> tuple:
        """
        Menyiapkan features dan target untuk training
        
        Returns:
            Tuple (X, y, feature_names)
        """
        logger.info("Menyiapkan features untuk training...")
        
        # Definisikan feature columns (semua kecuali target dan kolom identifikasi)
        feature_columns = [
            'Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata',
            'Produktivitas', 'Produksi_Lag1', 'Produksi_Lag2',
            'Produktivitas_Lag1', 'Curah_Hujan_Anomaly', 'Comfort_Index',
            'Provinsi_Encoded', 'Tahun'
        ]
        
        # Tambahkan one-hot encoded features jika ada (exclude kolom kategorikal asli)
        curah_hujan_cols = [col for col in self.df.columns 
                           if col.startswith('Curah_Hujan_') and col != 'Curah_Hujan_Kategori' 
                           and col != 'Curah_Hujan_Mean_Provinsi' and col != 'Curah_Hujan_Anomaly']
        feature_columns.extend(curah_hujan_cols)
        
        # Filter hanya kolom yang ada
        available_features = [col for col in feature_columns if col in self.df.columns]
        
        X = self.df[available_features].copy()
        y = self.df[self.target_col].copy()
        
        # Handle missing values - hanya untuk kolom numerik
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        
        self.feature_names = available_features
        
        logger.info(f"✓ Total features: {len(available_features)}")
        logger.info(f"  Features: {available_features}")
        logger.info(f"  Target: {self.target_col}")
        logger.info(f"  Dataset size: {len(X)} samples")
        
        return X, y, available_features
    
    def split_data(self, X, y, test_size: float = 0.2) -> tuple:
        """
        Split data menjadi training dan testing set
        Menggunakan strategi time-series (data terbaru untuk testing)
        
        Args:
            X: Features
            y: Target
            test_size: Proporsi data testing
            
        Returns:
            Tuple (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Membagi data (test_size={test_size})...")
        
        # Sort by year untuk time-series split
        if 'Tahun' in X.columns:
            sort_idx = X['Tahun'].argsort()
            X = X.iloc[sort_idx]
            y = y.iloc[sort_idx]
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"✓ Training set: {len(X_train)} samples")
        logger.info(f"✓ Testing set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_linear_regression(self, X_train, y_train):
        """Train Linear Regression model"""
        logger.info("Training Linear Regression...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models['Linear Regression'] = model
        logger.info("✓ Linear Regression trained")
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        logger.info("Training Random Forest...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['Random Forest'] = model
        logger.info("✓ Random Forest trained")
        return model
    
    def train_gradient_boosting(self, X_train, y_train):
        """Train Gradient Boosting model"""
        logger.info("Training Gradient Boosting...")
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models['Gradient Boosting'] = model
        logger.info("✓ Gradient Boosting trained")
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        logger.info("Training XGBoost...")
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['XGBoost'] = model
        logger.info("✓ XGBoost trained")
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> dict:
        """
        Evaluasi model dengan berbagai metrik
        
        Args:
            model: Model yang akan dievaluasi
            X_test: Features testing
            y_test: Target testing
            model_name: Nama model
            
        Returns:
            Dictionary dengan metrik evaluasi
        """
        y_pred = model.predict(X_test)
        
        # Hitung metrik
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # MAPE (handle division by zero)
        mask = y_test != 0
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        
        results = {
            'model_name': model_name,
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
        
        self.results[model_name] = results
        
        logger.info(f"  {model_name} - R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
        
        return results
    
    def cross_validate_models(self, X, y, cv: int = 5):
        """
        Cross-validation untuk semua model
        
        Args:
            X: Features
            y: Target
            cv: Number of folds
        """
        logger.info(f"\nMelakukan {cv}-Fold Cross-Validation...")
        
        tscv = TimeSeriesSplit(n_splits=cv)
        
        for model_name, model in self.models.items():
            scores = cross_val_score(
                model, X, y, 
                cv=tscv, 
                scoring='r2',
                n_jobs=-1
            )
            
            logger.info(f"  {model_name} - CV R² Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            if model_name in self.results:
                self.results[model_name]['cv_r2_mean'] = scores.mean()
                self.results[model_name]['cv_r2_std'] = scores.std()
    
    def get_feature_importance(self, model_name: str = None):
        """
        Mendapatkan feature importance dari model
        
        Args:
            model_name: Nama model (jika None, gunakan best model)
            
        Returns:
            DataFrame dengan feature importance
        """
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models.get(model_name)
        
        if model is None:
            logger.warning(f"Model {model_name} tidak ditemukan")
            return None
        
        # Get feature importance berdasarkan tipe model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            logger.warning(f"Model {model_name} tidak memiliki feature importance")
            return None
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def select_best_model(self):
        """Memilih model terbaik berdasarkan R² score"""
        logger.info("\nMemilih model terbaik...")
        
        best_r2 = -np.inf
        best_name = None
        
        for model_name, results in self.results.items():
            if results['r2_score'] > best_r2:
                best_r2 = results['r2_score']
                best_name = model_name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        logger.info(f"✓ Model terbaik: {best_name} (R² = {best_r2:.4f})")
        
        return best_name, self.best_model
    
    def save_models(self, output_dir: str = "models"):
        """
        Menyimpan semua model dan hasil evaluasi
        
        Args:
            output_dir: Direktori untuk menyimpan model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nMenyimpan models ke {output_dir}...")
        
        # Simpan setiap model
        for model_name, model in self.models.items():
            filename = model_name.lower().replace(' ', '_')
            model_path = output_path / f"{filename}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"  ✓ {model_name} saved to {model_path}")
        
        # Simpan best model secara terpisah
        if self.best_model:
            best_model_path = output_path / "best_model.pkl"
            joblib.dump(self.best_model, best_model_path)
            logger.info(f"  ✓ Best model saved to {best_model_path}")
        
        # Simpan feature names
        feature_path = output_path / "feature_names.json"
        with open(feature_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        # Simpan hasil evaluasi
        results_path = output_path / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"  ✓ Evaluation results saved to {results_path}")
        
        # Simpan metadata
        metadata = {
            'best_model': self.best_model_name,
            'target_column': self.target_col,
            'n_features': len(self.feature_names),
            'timestamp': str(datetime.now())
        }
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  ✓ Metadata saved to {metadata_path}")
        
    def plot_results(self, X_test, y_test, output_dir: str = "models"):
        """
        Membuat visualisasi hasil evaluasi
        
        Args:
            X_test: Features testing
            y_test: Target testing
            output_dir: Direktori untuk menyimpan plot
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Model Comparison
        plt.figure(figsize=(12, 6))
        
        metrics = ['r2_score', 'rmse', 'mae', 'mape']
        metric_names = ['R² Score', 'RMSE', 'MAE', 'MAPE (%)']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names), 1):
            plt.subplot(2, 2, idx)
            
            model_names = list(self.results.keys())
            values = [self.results[name][metric] for name in model_names]
            
            bars = plt.bar(model_names, values, color='steelblue', alpha=0.7)
            
            # Highlight best model
            if metric == 'r2_score':
                best_idx = values.index(max(values))
            else:
                best_idx = values.index(min(values))
            bars[best_idx].set_color('green')
            
            plt.title(metric_name, fontsize=12, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel(metric_name)
            plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        comparison_path = output_path / "model_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Model comparison plot saved to {comparison_path}")
        plt.close()
        
        # 2. Prediction vs Actual (Best Model)
        plt.figure(figsize=(10, 6))
        
        y_pred = self.best_model.predict(X_test)
        
        plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Production', fontsize=12)
        plt.ylabel('Predicted Production', fontsize=12)
        plt.title(f'Prediction vs Actual - {self.best_model_name}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        pred_vs_actual_path = output_path / "prediction_vs_actual.png"
        plt.savefig(pred_vs_actual_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Prediction vs Actual plot saved to {pred_vs_actual_path}")
        plt.close()
        
        # 3. Feature Importance (Best Model)
        feature_importance = self.get_feature_importance()
        
        if feature_importance is not None:
            plt.figure(figsize=(10, 8))
            
            top_n = min(15, len(feature_importance))
            top_features = feature_importance.head(top_n)
            
            plt.barh(range(top_n), top_features['importance'], color='steelblue', alpha=0.7)
            plt.yticks(range(top_n), top_features['feature'])
            plt.xlabel('Importance', fontsize=12)
            plt.title(f'Top {top_n} Feature Importance - {self.best_model_name}', 
                     fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            
            importance_path = output_path / "feature_importance.png"
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            logger.info(f"  ✓ Feature importance plot saved to {importance_path}")
            plt.close()
    
    def train(self, test_size: float = 0.2, cv_folds: int = 5):
        """
        Pipeline lengkap training model
        
        Args:
            test_size: Proporsi data testing
            cv_folds: Jumlah folds untuk cross-validation
        """
        logger.info("=" * 60)
        logger.info("MEMULAI TRAINING MODEL")
        logger.info("=" * 60)
        
        # Prepare features
        X, y, feature_names = self.prepare_features()
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size)
        
        # Train all models
        logger.info("\nTraining semua model...")
        self.train_linear_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        
        # Evaluate models
        logger.info("\nEvaluasi model pada test set...")
        for model_name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, model_name)
        
        # Cross-validation
        self.cross_validate_models(X, y, cv_folds)
        
        # Select best model
        self.select_best_model()
        
        # Save models
        self.save_models()
        
        # Plot results
        self.plot_results(X_test, y_test)
        
        logger.info("=" * 60)
        logger.info("✓ TRAINING MODEL SELESAI")
        logger.info("=" * 60)
        
        return self.best_model, self.results


def main():
    """Fungsi utama untuk menjalankan training model"""
    # Load data yang sudah ditransformasi
    input_path = "data/processed/transformed_data.pkl"
    logger.info(f"Memuat data dari {input_path}")
    df = pd.read_pickle(input_path)
    
    # Inisialisasi trainer
    trainer = ModelTrainer(df, target_col='Produksi')
    
    # Train models
    best_model, results = trainer.train(test_size=0.2, cv_folds=5)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RINGKASAN HASIL TRAINING")
    print("=" * 60)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  R² Score: {result['r2_score']:.4f}")
        print(f"  RMSE: {result['rmse']:.2f}")
        print(f"  MAE: {result['mae']:.2f}")
        print(f"  MAPE: {result['mape']:.2f}%")
        if 'cv_r2_mean' in result:
            print(f"  CV R² Score: {result['cv_r2_mean']:.4f} (+/- {result['cv_r2_std']:.4f})")
    
    print("\n" + "=" * 60)
    print(f"✓ Best Model: {trainer.best_model_name}")
    print("=" * 60)
    
    return best_model


if __name__ == "__main__":
    model = main()
