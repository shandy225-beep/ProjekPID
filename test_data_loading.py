"""
Quick test to verify data loading issue
"""
import pandas as pd
import sys
from pathlib import Path

print("Testing data loading...")
print("=" * 60)

try:
    # Test 1: Check if file exists
    pkl_path = Path("data/processed/transformed_data.pkl")
    print(f"1. File exists: {pkl_path.exists()}")
    
    # Test 2: Try to load the pickle file
    print("\n2. Loading pickle file...")
    df = pd.read_pickle("data/processed/transformed_data.pkl")
    print(f"   ✅ Loaded successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Test 3: Check key columns
    print("\n3. Checking key columns:")
    required_cols = ['Provinsi', 'Tahun', 'Produksi', 'Produktivitas']
    for col in required_cols:
        if col in df.columns:
            print(f"   ✅ {col}: OK")
        else:
            print(f"   ❌ {col}: MISSING")
    
    # Test 4: Check if Produksi_Original exists
    print(f"\n4. Produksi_Original exists: {'Produksi_Original' in df.columns}")
    
    # Test 5: Calculate overview metrics
    print("\n5. Calculating overview metrics:")
    print(f"   Total data points: {len(df):,}")
    print(f"   Number of provinces: {df['Provinsi'].nunique()}")
    print(f"   Year range: {int(df['Tahun'].min())} - {int(df['Tahun'].max())}")
    
    if 'Produksi' in df.columns:
        total_prod = df['Produksi'].sum()
        print(f"   Total production: {total_prod/1e6:.2f}M ton")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Data should load correctly!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    print("\nFull traceback:")
    print(traceback.format_exc())
    sys.exit(1)
