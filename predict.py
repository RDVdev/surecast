import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from pandas.api.types import is_numeric_dtype

# Import the model architecture
from phase4_model_architecture import MultiBranchSequenceModel

def main():
    print("==================================================")
    print("SUREcast Inference Test Case")
    print("==================================================")
    
    model_path = "models/surecast_dl.pth"
    scaler_path = "models/target_scaler.pkl"
    data_path = "data/engineered_dataset.csv"
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file {model_path} not found.")
        print("Please run phase 4 (or run_pipeline.py) first to generate and save the model.")
        sys.exit(1)
        
    print("1. Loading required scalers and determining features...")
    target_scaler = joblib.load(scaler_path)
    
    # Load dataset just to get the exact feature columns we trained on
    df = pd.read_csv(data_path, nrows=50) # only need a small chunk
    target_col = "Sales"
    if target_col not in df.columns:
        target_col = next((c for c in ['Sales per customer','Order Item Quantity'] if c in df.columns), df.columns[-1])
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    cat_group = 'Category Name' if 'Category Name' in df.columns else next((c for c in df.columns if 'category' in c.lower()), None)
    region_group = 'Order Region' if 'Order Region' in df.columns else next((c for c in df.columns if 'region' in c.lower()), None)
    
    ignore_cols = [target_col, cat_group, region_group, date_col, 'YearWeek']
    feature_cols = [c for c in df.columns if c not in ignore_cols and is_numeric_dtype(df[c])]
    
    seq_len = 8
    dl_features = len(feature_cols)
    
    print("2. Initializing Model Architecture...")
    # Initialize the architecture (Must match what we trained!)
    model = MultiBranchSequenceModel(input_size=dl_features, seq_len=seq_len)
    
    # Load the saved brain/weights into the model
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set model to evaluation mode (turns off dropout)
    print(" -> Model loaded successfully!")
    
    print("\n3. Formatting Sample New Data...")
    # For this test case, we will simulate receiving the last 8 days of data for a specific product
    # In a real system, you would load this from your live database.
    # We grab 8 rows of real features from our dataset to act as the "new" sequence
    sample_data_raw = df[feature_cols].values[:seq_len]
    
    # The PyTorch model expects shape: (batch_size, sequence_length, features)
    # We have 1 sample, 8 days, and 'dl_features' columns
    sample_tensor = torch.tensor(sample_data_raw, dtype=torch.float32).unsqueeze(0) 
    
    print(f" -> Sample data shape: {sample_tensor.shape}")
    
    print("\n4. Running Prediction...")
    with torch.no_grad(): # Don't track gradients during inference (saves memory/time)
        scaled_prediction = model(sample_tensor)
        
    # The model outputs a scaled prediction, we need to convert it back to actual Sales dollars
    scaled_value = scaled_prediction.numpy()[0][0]
    actual_sales_prediction = target_scaler.inverse_transform([[scaled_value]])[0][0]
    
    print("==================================================")
    print(f"RESULT: The model predicts the next period's sales will be: ${actual_sales_prediction:.2f}")
    print("==================================================")

if __name__ == "__main__":
    main()
