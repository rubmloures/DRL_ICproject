"""
Example 1: Complete Data Pipeline
==================================

This example demonstrates:
1. Loading raw CSV data (Brazilian format)
2. Cleaning and preprocessing
3. Adding technical indicators
4. Normalizing features
5. Splitting into train/test sets
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data import DataLoader, DataProcessor


def main():
    print("\n" + "="*60)
    print("EXAMPLE 1: Data Pipeline")
    print("="*60)
    
    # Initialize loader
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    loader = DataLoader(data_path=data_dir)
    
    # Load data for multiple assets
    print("\n[Step 1] Loading raw data...")
    try:
        df = loader.load_multiple_assets(
            assets=["PETR4", "VALE3"],
            start_date="2022-01-01",
            end_date="2024-12-31",
        )
        print(f"✓ Loaded {len(df)} records")
        print(f"  Columns: {list(df.columns)}")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        print("  Make sure you have CSV files in data/raw/")
        return
    
    # Initialize processor
    processor = DataProcessor()
    
    # Clean data
    print("\n[Step 2] Cleaning data...")
    df_clean = processor.clean_data(df)
    print(f"✓ Cleaned data: {len(df_clean)} records")
    
    # Add technical indicators
    print("\n[Step 3] Adding technical indicators...")
    df_features = processor.add_technical_indicators(df_clean)
    indicator_cols = [c for c in df_features.columns if 'SMA' in c or 'RSI' in c or 'MACD' in c]
    print(f"✓ Added {len(indicator_cols)} indicators: {indicator_cols[:5]}...")
    
    # Normalize Greeks (if options data exists)
    if 'delta' in df_features.columns:
        print("\n[Step 4] Normalizing Greeks...")
        df_features = processor.normalize_greeks(df_features)
        print("✓ Normalized Greeks")
    
    # Split train/test
    print("\n[Step 5] Splitting train/test...")
    train_data, test_data = DataProcessor.split_data(df_features, train_ratio=0.8)
    print(f"✓ Train: {len(train_data)} | Test: {len(test_data)}")
    
    # Fit scaler on training data
    print("\n[Step 6] Fitting feature scaler...")
    scaler_cols = [
        'SMA_20', 'SMA_50', 'RSI_14', 'ATR_14',
        'acao_close_ajustado'
    ]
    available_cols = [c for c in scaler_cols if c in train_data.columns]
    processor.fit_scaler(train_data, columns=available_cols, scaler_name="features")
    print(f"✓ Fitted scaler on {len(available_cols)} columns")
    
    # Transform data
    print("\n[Step 7] Transforming features...")
    train_scaled = processor.transform(train_data, scaler_name="features")
    test_scaled = processor.transform(test_data, scaler_name="features")
    print(f"✓ Transformed train/test data")
    
    # Show summary
    print("\n[Summary] Data Pipeline Complete")
    print(f"  Raw data: {len(df)} records, {len(df.columns)} columns")
    print(f"  Features: {len(available_cols)} normalized indicators")
    print(f"  Train set: {len(train_scaled)} records")
    print(f"  Test set: {len(test_scaled)} records")
    print(f"  Feature columns: {[c for c in train_scaled.columns if '_scaled' in c][:5]}...")
    
    # Export processed data
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_scaled.to_csv(output_dir / "train_data.csv", index=False)
    test_scaled.to_csv(output_dir / "test_data.csv", index=False)
    print(f"\n✓ Saved processed data to {output_dir}")
    
    print("\n" + "="*60)
    print("Example 1 Completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
