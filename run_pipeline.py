"""
Main Pipeline - Run Complete ETL and ML Pipeline
Menjalankan semua tahapan pipeline secara otomatis
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.extract import main as extract_main
from src.transform import main as transform_main
from src.load import main as load_main
from src.train_model import main as train_main
from src.predict import main as predict_main

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def print_banner(text):
    """Print formatted banner"""
    banner = f"\n{'='*70}\n{text.center(70)}\n{'='*70}\n"
    print(banner)
    logger.info(banner)


def run_pipeline(skip_steps=None):
    """
    Run complete data pipeline
    
    Args:
        skip_steps: List of steps to skip (e.g., ['extract', 'transform'])
    """
    if skip_steps is None:
        skip_steps = []
    
    start_time = datetime.now()
    
    print_banner("🌾 MEMULAI PIPELINE DATA - PREDIKSI HASIL PANEN PADI 🌾")
    logger.info(f"Pipeline started at: {start_time}")
    
    try:
        # Step 1: Extract
        if 'extract' not in skip_steps:
            print_banner("TAHAP 1: EXTRACT (EKSTRAKSI DATA)")
            logger.info("Starting extraction...")
            df_extracted = extract_main()
            logger.info("✓ Extraction completed successfully")
        else:
            logger.info("⊘ Extraction skipped")
        
        # Step 2: Transform
        if 'transform' not in skip_steps:
            print_banner("TAHAP 2: TRANSFORM (TRANSFORMASI DATA)")
            logger.info("Starting transformation...")
            df_transformed = transform_main()
            logger.info("✓ Transformation completed successfully")
        else:
            logger.info("⊘ Transformation skipped")
        
        # Step 3: Load
        if 'load' not in skip_steps:
            print_banner("TAHAP 3: LOAD (PEMUATAN KE DATABASE)")
            logger.info("Starting data loading...")
            load_main()
            logger.info("✓ Data loading completed successfully")
        else:
            logger.info("⊘ Data loading skipped")
        
        # Step 4: Train Model
        if 'train' not in skip_steps:
            print_banner("TAHAP 4: TRAIN MODEL (TRAINING MACHINE LEARNING)")
            logger.info("Starting model training...")
            model = train_main()
            logger.info("✓ Model training completed successfully")
        else:
            logger.info("⊘ Model training skipped")
        
        # Step 5: Predict
        if 'predict' not in skip_steps:
            print_banner("TAHAP 5: PREDICT (PREDIKSI MASA DEPAN)")
            logger.info("Starting predictions...")
            predictions = predict_main()
            logger.info("✓ Predictions completed successfully")
        else:
            logger.info("⊘ Predictions skipped")
        
        # Pipeline completed
        end_time = datetime.now()
        duration = end_time - start_time
        
        print_banner("✅ PIPELINE SELESAI DENGAN SUKSES ✅")
        logger.info(f"Pipeline completed at: {end_time}")
        logger.info(f"Total duration: {duration}")
        
        # Summary
        print("\n" + "="*70)
        print("📊 RINGKASAN PIPELINE")
        print("="*70)
        print(f"⏱️  Waktu Mulai     : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Waktu Selesai   : {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Durasi Total    : {duration}")
        print("\n📁 Output Files:")
        print("   ✓ data/processed/extracted_data.pkl")
        print("   ✓ data/processed/transformed_data.pkl")
        print("   ✓ data/processed/transformed_data.csv")
        print("   ✓ data/processed/data_warehouse.db")
        print("   ✓ models/best_model.pkl")
        print("   ✓ models/evaluation_results.json")
        print("   ✓ models/*.png (visualizations)")
        print("   ✓ data/predictions/future_predictions.csv")
        print("   ✓ data/predictions/prediction_summary.json")
        print("\n🎯 Next Steps:")
        print("   1. Lihat hasil evaluasi model di: models/evaluation_results.json")
        print("   2. Lihat prediksi di: data/predictions/future_predictions.csv")
        print("   3. Jalankan dashboard: streamlit run dashboard/app.py")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        print_banner("❌ PIPELINE GAGAL ❌")
        print(f"\nError: {e}")
        print("\nPeriksa log file di: logs/pipeline.log")
        return False


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run complete data pipeline for rice production prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_pipeline.py
  
  # Skip extraction and transformation (use existing data)
  python run_pipeline.py --skip extract transform
  
  # Only run model training and prediction
  python run_pipeline.py --skip extract transform load
        """
    )
    
    parser.add_argument(
        '--skip',
        nargs='+',
        choices=['extract', 'transform', 'load', 'train', 'predict'],
        help='Skip specific pipeline steps'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    success = run_pipeline(skip_steps=args.skip)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
