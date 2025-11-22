"""
Load Module - Pemuatan Data ke Database
Menyimpan data ke SQL Database dengan schema star schema
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import sqlite3
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

Base = declarative_base()


# ============== SCHEMA DESIGN ==============

class DimProvinsi(Base):
    """Tabel Dimensi Provinsi"""
    __tablename__ = 'dim_provinsi'
    
    provinsi_id = Column(Integer, primary_key=True, autoincrement=True)
    nama_provinsi = Column(String(100), unique=True, nullable=False)
    provinsi_code = Column(Integer, nullable=False)
    
    # Relationship
    fakta_produksi = relationship("FaktaProduksi", back_populates="provinsi")


class DimWaktu(Base):
    """Tabel Dimensi Waktu"""
    __tablename__ = 'dim_waktu'
    
    waktu_id = Column(Integer, primary_key=True, autoincrement=True)
    tahun = Column(Integer, unique=True, nullable=False)
    dekade = Column(String(10))
    periode = Column(String(20))
    
    # Relationship
    fakta_produksi = relationship("FaktaProduksi", back_populates="waktu")


class FaktaProduksi(Base):
    """Tabel Fakta Produksi dan Cuaca"""
    __tablename__ = 'fakta_produksi'
    
    fakta_id = Column(Integer, primary_key=True, autoincrement=True)
    provinsi_id = Column(Integer, ForeignKey('dim_provinsi.provinsi_id'), nullable=False)
    waktu_id = Column(Integer, ForeignKey('dim_waktu.waktu_id'), nullable=False)
    
    # Metrics Produksi
    produksi = Column(Float, nullable=False)
    luas_panen = Column(Float, nullable=False)
    produktivitas = Column(Float)
    produksi_lag1 = Column(Float)
    produksi_lag2 = Column(Float)
    produktivitas_lag1 = Column(Float)
    
    # Metrics Cuaca
    curah_hujan = Column(Float)
    kelembapan = Column(Float)
    suhu_rata_rata = Column(Float)
    curah_hujan_anomaly = Column(Float)
    comfort_index = Column(Float)
    curah_hujan_kategori = Column(String(50))
    
    # Timestamps
    created_at = Column(String(50), default=str(datetime.now()))
    
    # Relationships
    provinsi = relationship("DimProvinsi", back_populates="fakta_produksi")
    waktu = relationship("DimWaktu", back_populates="fakta_produksi")


# ============== DATA LOADER CLASS ==============

class DataLoader:
    """Class untuk loading data ke database"""
    
    def __init__(self, db_path: str = "data/processed/data_warehouse.db"):
        """
        Initialize DataLoader
        
        Args:
            db_path: Path ke database SQLite
        """
        self.db_path = Path(db_path)
        self.engine = None
        self.session = None
        
    def create_connection(self):
        """Membuat koneksi ke database"""
        logger.info(f"Membuat koneksi ke database: {self.db_path}")
        
        # Pastikan direktori ada
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create engine
        self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False)
        
        # Create all tables
        Base.metadata.create_all(self.engine)
        logger.info("✓ Tabel database berhasil dibuat")
        
        # Create session
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
    def load_dimension_provinsi(self, df: pd.DataFrame):
        """
        Load data ke tabel dimensi provinsi
        
        Args:
            df: DataFrame dengan kolom Provinsi
        """
        logger.info("Loading Dimensi Provinsi...")
        
        # Get unique provinsi
        provinsi_list = df[['Provinsi', 'Provinsi_Encoded']].drop_duplicates()
        
        # Clear existing data
        self.session.query(DimProvinsi).delete()
        
        # Insert data
        for _, row in provinsi_list.iterrows():
            provinsi = DimProvinsi(
                nama_provinsi=row['Provinsi'],
                provinsi_code=int(row['Provinsi_Encoded'])
            )
            self.session.add(provinsi)
        
        self.session.commit()
        count = self.session.query(DimProvinsi).count()
        logger.info(f"✓ {count} provinsi berhasil di-load")
        
    def load_dimension_waktu(self, df: pd.DataFrame):
        """
        Load data ke tabel dimensi waktu
        
        Args:
            df: DataFrame dengan kolom Tahun
        """
        logger.info("Loading Dimensi Waktu...")
        
        # Get unique years
        years = df['Tahun'].unique()
        
        # Clear existing data
        self.session.query(DimWaktu).delete()
        
        # Insert data
        for year in sorted(years):
            dekade = f"{(year // 10) * 10}s"
            
            if year < 2000:
                periode = "1990-1999"
            elif year < 2010:
                periode = "2000-2009"
            elif year < 2020:
                periode = "2010-2019"
            else:
                periode = "2020+"
            
            waktu = DimWaktu(
                tahun=int(year),
                dekade=dekade,
                periode=periode
            )
            self.session.add(waktu)
        
        self.session.commit()
        count = self.session.query(DimWaktu).count()
        logger.info(f"✓ {count} tahun berhasil di-load")
        
    def load_fakta_produksi(self, df: pd.DataFrame):
        """
        Load data ke tabel fakta produksi
        
        Args:
            df: DataFrame dengan semua kolom
        """
        logger.info("Loading Fakta Produksi...")
        
        # Get dimension mappings
        provinsi_map = {p.nama_provinsi: p.provinsi_id 
                       for p in self.session.query(DimProvinsi).all()}
        waktu_map = {w.tahun: w.waktu_id 
                    for w in self.session.query(DimWaktu).all()}
        
        # Clear existing data
        self.session.query(FaktaProduksi).delete()
        
        # Insert data
        for _, row in df.iterrows():
            provinsi_id = provinsi_map.get(row['Provinsi'])
            waktu_id = waktu_map.get(row['Tahun'])
            
            if provinsi_id is None or waktu_id is None:
                continue
            
            # Gunakan kolom original (sebelum normalisasi) jika ada
            produksi_val = row.get('Produksi_Original', row['Produksi'])
            luas_panen_val = row.get('Luas Panen_Original', row['Luas Panen'])
            produktivitas_val = row.get('Produktivitas_Original', row['Produktivitas'])
            curah_hujan_val = row.get('Curah hujan_Original', row['Curah hujan'])
            kelembapan_val = row.get('Kelembapan_Original', row['Kelembapan'])
            suhu_val = row.get('Suhu rata-rata_Original', row['Suhu rata-rata'])
            
            fakta = FaktaProduksi(
                provinsi_id=provinsi_id,
                waktu_id=waktu_id,
                produksi=float(produksi_val),
                luas_panen=float(luas_panen_val),
                produktivitas=float(produktivitas_val) if pd.notna(produktivitas_val) else None,
                produksi_lag1=float(row['Produksi_Lag1']) if pd.notna(row.get('Produksi_Lag1')) else None,
                produksi_lag2=float(row['Produksi_Lag2']) if pd.notna(row.get('Produksi_Lag2')) else None,
                produktivitas_lag1=float(row['Produktivitas_Lag1']) if pd.notna(row.get('Produktivitas_Lag1')) else None,
                curah_hujan=float(curah_hujan_val) if pd.notna(curah_hujan_val) else None,
                kelembapan=float(kelembapan_val) if pd.notna(kelembapan_val) else None,
                suhu_rata_rata=float(suhu_val) if pd.notna(suhu_val) else None,
                curah_hujan_anomaly=float(row['Curah_Hujan_Anomaly']) if pd.notna(row.get('Curah_Hujan_Anomaly')) else None,
                comfort_index=float(row['Comfort_Index']) if pd.notna(row.get('Comfort_Index')) else None,
                curah_hujan_kategori=str(row['Curah_Hujan_Kategori']) if pd.notna(row.get('Curah_Hujan_Kategori')) else None
            )
            self.session.add(fakta)
        
        self.session.commit()
        count = self.session.query(FaktaProduksi).count()
        logger.info(f"✓ {count} fakta produksi berhasil di-load")
        
    def verify_data(self):
        """Verifikasi data yang sudah di-load"""
        logger.info("\n" + "=" * 60)
        logger.info("VERIFIKASI DATA WAREHOUSE")
        logger.info("=" * 60)
        
        provinsi_count = self.session.query(DimProvinsi).count()
        waktu_count = self.session.query(DimWaktu).count()
        fakta_count = self.session.query(FaktaProduksi).count()
        
        logger.info(f"Dim Provinsi: {provinsi_count} records")
        logger.info(f"Dim Waktu: {waktu_count} records")
        logger.info(f"Fakta Produksi: {fakta_count} records")
        
        # Sample data
        logger.info("\nContoh data dari Fakta Produksi:")
        sample = self.session.query(FaktaProduksi).limit(5).all()
        for s in sample:
            logger.info(f"  Provinsi ID: {s.provinsi_id}, Tahun ID: {s.waktu_id}, "
                       f"Produksi: {s.produksi:.2f}, Produktivitas: {s.produktivitas:.2f}")
        
        logger.info("=" * 60)
        
    def load(self, df: pd.DataFrame):
        """
        Pipeline lengkap loading data
        
        Args:
            df: DataFrame yang sudah ditransformasi
        """
        logger.info("=" * 60)
        logger.info("MEMULAI LOADING DATA KE DATABASE")
        logger.info("=" * 60)
        
        # Create connection
        self.create_connection()
        
        # Load dimensions
        self.load_dimension_provinsi(df)
        self.load_dimension_waktu(df)
        
        # Load facts
        self.load_fakta_produksi(df)
        
        # Verify
        self.verify_data()
        
        logger.info("=" * 60)
        logger.info("✓ LOADING DATA SELESAI")
        logger.info("=" * 60)
        
    def close(self):
        """Tutup koneksi database"""
        if self.session:
            self.session.close()
        logger.info("✓ Koneksi database ditutup")


def main():
    """Fungsi utama untuk menjalankan loading data"""
    # Load data yang sudah ditransformasi
    input_path = "data/processed/transformed_data.pkl"
    logger.info(f"Memuat data dari {input_path}")
    df = pd.read_pickle(input_path)
    
    # Inisialisasi loader
    loader = DataLoader()
    
    # Load data ke database
    loader.load(df)
    
    # Close connection
    loader.close()
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✓ Loading data ke database selesai!")
        print("Database tersimpan di: data/processed/data_warehouse.db")
