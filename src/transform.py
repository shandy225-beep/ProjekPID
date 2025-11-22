"""
Transform Module - Transformasi dan Feature Engineering
Melakukan data cleaning, feature engineering, join operations, dan normalisasi
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
from typing import Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataTransformer:
    """Class untuk transformasi data"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataTransformer
        
        Args:
            df: DataFrame input
        """
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Menangani missing values dengan imputation
        
        Returns:
            DataFrame dengan missing values yang sudah ditangani
        """
        logger.info("Menangani missing values...")
        
        missing_before = self.df.isnull().sum().sum()
        logger.info(f"Missing values sebelum: {missing_before}")
        
        # Imputation untuk kolom numerik menggunakan median
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if self.df[col].isnull().sum() > 0:
                median_value = self.df[col].median()
                self.df[col].fillna(median_value, inplace=True)
                logger.info(f"  - {col}: diisi dengan median {median_value:.2f}")
        
        # Imputation untuk kolom kategorikal menggunakan mode
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if self.df[col].isnull().sum() > 0:
                mode_value = self.df[col].mode()[0]
                self.df[col].fillna(mode_value, inplace=True)
                logger.info(f"  - {col}: diisi dengan mode {mode_value}")
        
        missing_after = self.df.isnull().sum().sum()
        logger.info(f"Missing values setelah: {missing_after}")
        logger.info("✓ Missing values berhasil ditangani")
        
        return self.df
    
    def detect_outliers_iqr(self, column: str) -> Tuple[pd.Series, int]:
        """
        Deteksi outliers menggunakan metode IQR
        
        Args:
            column: Nama kolom untuk deteksi outlier
            
        Returns:
            Tuple (outlier_mask, outlier_count)
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        outlier_count = outlier_mask.sum()
        
        return outlier_mask, outlier_count
    
    def handle_outliers(self) -> pd.DataFrame:
        """
        Menangani outliers dengan capping (winsorization)
        
        Returns:
            DataFrame dengan outliers yang sudah ditangani
        """
        logger.info("Mendeteksi dan menangani outliers...")
        
        numeric_columns = ['Produksi', 'Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
        
        for col in numeric_columns:
            if col in self.df.columns:
                outlier_mask, outlier_count = self.detect_outliers_iqr(col)
                
                if outlier_count > 0:
                    logger.info(f"  - {col}: {outlier_count} outliers ditemukan")
                    
                    # Capping menggunakan percentile
                    lower_cap = self.df[col].quantile(0.01)
                    upper_cap = self.df[col].quantile(0.99)
                    
                    self.df[col] = np.clip(self.df[col], lower_cap, upper_cap)
                    logger.info(f"    Capped ke range [{lower_cap:.2f}, {upper_cap:.2f}]")
        
        logger.info("✓ Outliers berhasil ditangani")
        return self.df
    
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Menghapus baris duplikat
        
        Returns:
            DataFrame tanpa duplikasi
        """
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        
        removed = before - after
        if removed > 0:
            logger.info(f"✓ {removed} baris duplikat dihapus")
        else:
            logger.info("✓ Tidak ada duplikasi ditemukan")
        
        return self.df
    
    def create_productivity_feature(self) -> pd.DataFrame:
        """
        Membuat feature produktivitas per hektar
        
        Returns:
            DataFrame dengan feature produktivitas
        """
        logger.info("Membuat feature: Produktivitas per hektar...")
        
        # Produktivitas = Produksi / Luas Panen
        self.df['Produktivitas'] = self.df['Produksi'] / self.df['Luas Panen']
        
        # Handle division by zero
        self.df['Produktivitas'].replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df['Produktivitas'].fillna(self.df['Produktivitas'].median(), inplace=True)
        
        logger.info(f"  Rata-rata produktivitas: {self.df['Produktivitas'].mean():.2f} ton/ha")
        logger.info("✓ Feature produktivitas berhasil dibuat")
        
        return self.df
    
    def create_lag_features(self) -> pd.DataFrame:
        """
        Membuat lag features untuk time-series
        
        Returns:
            DataFrame dengan lag features
        """
        logger.info("Membuat lag features untuk time-series...")
        
        # Sort by Provinsi dan Tahun
        self.df = self.df.sort_values(['Provinsi', 'Tahun'])
        
        # Lag features untuk produksi tahun sebelumnya
        self.df['Produksi_Lag1'] = self.df.groupby('Provinsi')['Produksi'].shift(1)
        self.df['Produksi_Lag2'] = self.df.groupby('Provinsi')['Produksi'].shift(2)
        
        # Lag features untuk produktivitas
        self.df['Produktivitas_Lag1'] = self.df.groupby('Provinsi')['Produktivitas'].shift(1)
        
        # Fill missing lag values (untuk tahun pertama)
        self.df['Produksi_Lag1'].fillna(self.df.groupby('Provinsi')['Produksi'].transform('mean'), inplace=True)
        self.df['Produksi_Lag2'].fillna(self.df.groupby('Provinsi')['Produksi'].transform('mean'), inplace=True)
        self.df['Produktivitas_Lag1'].fillna(self.df.groupby('Provinsi')['Produktivitas'].transform('mean'), inplace=True)
        
        logger.info("✓ Lag features berhasil dibuat")
        return self.df
    
    def create_weather_features(self) -> pd.DataFrame:
        """
        Membuat feature engineering untuk cuaca
        
        Returns:
            DataFrame dengan weather features
        """
        logger.info("Membuat weather features...")
        
        # Anomali curah hujan (deviasi dari rata-rata per provinsi)
        self.df['Curah_Hujan_Mean_Provinsi'] = self.df.groupby('Provinsi')['Curah hujan'].transform('mean')
        self.df['Curah_Hujan_Anomaly'] = self.df['Curah hujan'] - self.df['Curah_Hujan_Mean_Provinsi']
        
        # Kategori curah hujan
        self.df['Curah_Hujan_Kategori'] = pd.cut(
            self.df['Curah hujan'],
            bins=[0, 1000, 2000, 3000, 10000],
            labels=['Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
        )
        
        # Indeks kenyamanan tanaman (kombinasi kelembapan dan suhu)
        self.df['Comfort_Index'] = (self.df['Kelembapan'] / 100) * (30 - abs(self.df['Suhu rata-rata'] - 27))
        
        logger.info("✓ Weather features berhasil dibuat")
        return self.df
    
    def encode_categorical(self) -> pd.DataFrame:
        """
        Encoding variabel kategorikal
        
        Returns:
            DataFrame dengan encoding kategorikal
        """
        logger.info("Encoding variabel kategorikal...")
        
        # Label Encoding untuk Provinsi
        self.df['Provinsi_Encoded'] = self.label_encoder.fit_transform(self.df['Provinsi'])
        
        # One-Hot Encoding untuk Curah_Hujan_Kategori
        if 'Curah_Hujan_Kategori' in self.df.columns:
            curah_hujan_dummies = pd.get_dummies(
                self.df['Curah_Hujan_Kategori'], 
                prefix='Curah_Hujan',
                drop_first=True
            )
            self.df = pd.concat([self.df, curah_hujan_dummies], axis=1)
        
        logger.info("✓ Encoding kategorikal selesai")
        return self.df
    
    def normalize_features(self, columns_to_normalize: list = None) -> pd.DataFrame:
        """
        Normalisasi fitur numerik menggunakan StandardScaler
        
        Args:
            columns_to_normalize: List kolom yang akan dinormalisasi
            
        Returns:
            DataFrame dengan fitur yang dinormalisasi
        """
        logger.info("Normalisasi fitur numerik...")
        
        if columns_to_normalize is None:
            columns_to_normalize = [
                'Produksi', 'Luas Panen', 'Curah hujan', 'Kelembapan', 
                'Suhu rata-rata', 'Produktivitas', 'Produksi_Lag1', 
                'Produksi_Lag2', 'Produktivitas_Lag1', 'Comfort_Index',
                'Curah_Hujan_Anomaly'
            ]
        
        # Filter hanya kolom yang ada
        columns_to_normalize = [col for col in columns_to_normalize if col in self.df.columns]
        
        # Simpan kolom asli
        for col in columns_to_normalize:
            self.df[f'{col}_Original'] = self.df[col]
        
        # Normalisasi
        self.df[columns_to_normalize] = self.scaler.fit_transform(self.df[columns_to_normalize])
        
        logger.info(f"✓ {len(columns_to_normalize)} kolom berhasil dinormalisasi")
        return self.df
    
    def transform(self, normalize: bool = True) -> pd.DataFrame:
        """
        Pipeline lengkap transformasi data
        
        Args:
            normalize: Apakah melakukan normalisasi atau tidak
            
        Returns:
            DataFrame yang sudah ditransformasi
        """
        logger.info("=" * 60)
        logger.info("MEMULAI TRANSFORMASI DATA")
        logger.info("=" * 60)
        
        # 1. Data Cleaning
        self.handle_missing_values()
        self.remove_duplicates()
        self.handle_outliers()
        
        # 2. Feature Engineering
        self.create_productivity_feature()
        self.create_lag_features()
        self.create_weather_features()
        
        # 3. Encoding
        self.encode_categorical()
        
        # 4. Normalization (optional)
        if normalize:
            self.normalize_features()
        
        logger.info("=" * 60)
        logger.info("✓ TRANSFORMASI DATA SELESAI")
        logger.info("=" * 60)
        
        return self.df


def main():
    """Fungsi utama untuk menjalankan transformasi data"""
    # Load data yang sudah diekstrak
    input_path = "data/processed/extracted_data.pkl"
    logger.info(f"Memuat data dari {input_path}")
    df = pd.read_pickle(input_path)
    
    # Inisialisasi transformer
    transformer = DataTransformer(df)
    
    # Transformasi data (tanpa normalisasi untuk visualisasi)
    df_transformed = transformer.transform(normalize=False)
    
    # Simpan hasil transformasi
    output_path = "data/processed/transformed_data.pkl"
    df_transformed.to_pickle(output_path)
    logger.info(f"✓ Data transformasi disimpan ke {output_path}")
    
    # Simpan juga versi CSV untuk keperluan lain
    csv_output_path = "data/processed/transformed_data.csv"
    df_transformed.to_csv(csv_output_path, index=False)
    logger.info(f"✓ Data transformasi disimpan ke {csv_output_path}")
    
    return df_transformed


if __name__ == "__main__":
    df_transformed = main()
    print("\n✓ Transformasi data selesai!")
    print(f"DataFrame shape: {df_transformed.shape}")
    print("\nKolom yang tersedia:")
    print(df_transformed.columns.tolist())
    print("\nContoh data:")
    print(df_transformed.head())
