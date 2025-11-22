"""
Extract Module - Ekstraksi Data dari CSV
Membaca data dari Kaggle Dataset dan melakukan validasi dasar
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataExtractor:
    """Class untuk ekstraksi data dari file CSV"""
    
    def __init__(self, data_path: str):
        """
        Initialize DataExtractor
        
        Args:
            data_path: Path ke file CSV
        """
        self.data_path = Path(data_path)
        self.df = None
        
    def read_csv(self) -> pd.DataFrame:
        """
        Membaca file CSV ke DataFrame
        
        Returns:
            DataFrame dengan data mentah
        """
        try:
            logger.info(f"Membaca data dari {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data berhasil dibaca. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error membaca file CSV: {e}")
            raise
    
    def validate_data(self) -> Tuple[bool, list]:
        """
        Validasi integritas data dan tipe data
        
        Returns:
            Tuple (is_valid, list_of_issues)
        """
        issues = []
        
        # Cek kolom yang diperlukan
        required_columns = [
            'Provinsi', 'Tahun', 'Produksi', 'Luas Panen',
            'Curah hujan', 'Kelembapan', 'Suhu rata-rata'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            issues.append(f"Kolom yang hilang: {missing_columns}")
        
        # Cek tipe data
        numeric_columns = ['Tahun', 'Produksi', 'Luas Panen', 'Curah hujan', 
                          'Kelembapan', 'Suhu rata-rata']
        
        for col in numeric_columns:
            if col in self.df.columns:
                try:
                    pd.to_numeric(self.df[col], errors='raise')
                except:
                    issues.append(f"Kolom {col} bukan numerik")
        
        # Cek missing values
        missing_summary = self.df.isnull().sum()
        if missing_summary.sum() > 0:
            issues.append(f"Missing values ditemukan:\n{missing_summary[missing_summary > 0]}")
        
        # Cek duplikasi
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Ditemukan {duplicates} baris duplikat")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("✓ Validasi data berhasil")
        else:
            logger.warning(f"⚠ Validasi menemukan {len(issues)} masalah:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid, issues
    
    def get_data_summary(self) -> dict:
        """
        Mendapatkan ringkasan data
        
        Returns:
            Dictionary dengan informasi ringkasan data
        """
        if self.df is None:
            raise ValueError("Data belum dimuat. Jalankan read_csv() terlebih dahulu.")
        
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'provinsi_count': self.df['Provinsi'].nunique(),
            'provinsi_list': self.df['Provinsi'].unique().tolist(),
            'year_range': (self.df['Tahun'].min(), self.df['Tahun'].max()),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            'dtypes': self.df.dtypes.to_dict()
        }
        
        logger.info("=" * 60)
        logger.info("RINGKASAN DATA")
        logger.info("=" * 60)
        logger.info(f"Total Baris: {summary['total_rows']}")
        logger.info(f"Total Kolom: {summary['total_columns']}")
        logger.info(f"Jumlah Provinsi: {summary['provinsi_count']}")
        logger.info(f"Rentang Tahun: {summary['year_range'][0]} - {summary['year_range'][1]}")
        logger.info(f"Penggunaan Memory: {summary['memory_usage_mb']:.2f} MB")
        logger.info("=" * 60)
        
        return summary
    
    def extract(self) -> pd.DataFrame:
        """
        Pipeline lengkap ekstraksi data
        
        Returns:
            DataFrame yang sudah divalidasi
        """
        # Baca data
        self.read_csv()
        
        # Validasi data
        is_valid, issues = self.validate_data()
        
        # Tampilkan ringkasan
        self.get_data_summary()
        
        return self.df


def main():
    """Fungsi utama untuk menjalankan ekstraksi data"""
    # Path ke data
    data_path = "data/raw/Data_Tanaman_Padi_Sumatera_version_1.csv"
    
    # Inisialisasi extractor
    extractor = DataExtractor(data_path)
    
    # Ekstraksi data
    df = extractor.extract()
    
    # Simpan ke pickle untuk proses selanjutnya
    output_path = "data/processed/extracted_data.pkl"
    df.to_pickle(output_path)
    logger.info(f"✓ Data mentah disimpan ke {output_path}")
    
    return df


if __name__ == "__main__":
    df = main()
    print("\n✓ Ekstraksi data selesai!")
    print(f"DataFrame shape: {df.shape}")
    print("\nContoh data:")
    print(df.head())
