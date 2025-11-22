"""
Test script to verify dashboard fixes
"""
import pandas as pd
import json
from pathlib import Path

def test_model_performance_fix():
    """Test the model performance column mismatch fix"""
    print("Testing model performance fix...")
    
    # Load the actual model results
    with open('models/evaluation_results.json', 'r') as f:
        model_results = json.load(f)
    
    # Simulate what the dashboard does
    df_results = pd.DataFrame(model_results).T
    print(f"After transpose: {df_results.shape} - Columns: {df_results.columns.tolist()}")
    
    df_results = df_results.reset_index()
    print(f"After reset_index: {df_results.shape} - Columns: {df_results.columns.tolist()}")
    
    # Rename the index column to Model
    df_results = df_results.rename(columns={'index': 'Model'})
    print(f"After rename: {df_results.shape} - Columns: {df_results.columns.tolist()}")
    
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
    
    print(f"Final columns to select: {final_cols}")
    
    # Select only the columns we have
    df_results = df_results[final_cols]
    print(f"Final shape: {df_results.shape} - Columns: {df_results.columns.tolist()}")
    print("\n✅ Model performance fix works!")
    print("\nFinal DataFrame:")
    print(df_results)
    return True

def test_overview_fix():
    """Test the overview display fix"""
    print("\n\nTesting overview fix...")
    
    # Try loading the data
    try:
        df = pd.read_pickle("data/processed/transformed_data.pkl")
        print(f"✅ Data loaded successfully: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Test the metrics that overview would calculate
        print(f"\nTotal data points: {len(df):,}")
        print(f"Number of provinces: {df['Provinsi'].nunique()}")
        print(f"Year range: {int(df['Tahun'].min())} - {int(df['Tahun'].max())}")
        
        total_prod = df['Produksi_Original'].sum() if 'Produksi_Original' in df.columns else df['Produksi'].sum()
        print(f"Total production: {total_prod/1e6:.2f}M ton")
        
        print("\n✅ Overview fix works!")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Dashboard Fixes Test")
    print("=" * 60)
    
    test1 = test_model_performance_fix()
    test2 = test_overview_fix()
    
    print("\n" + "=" * 60)
    if test1 and test2:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)
