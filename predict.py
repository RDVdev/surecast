"""
Single-sample inference test.
BUG-14 fix: torch.load uses weights_only=True.
ARCH-9: Added device support.
"""
import torch
import numpy as np
import pandas as pd
import joblib
import logging
import os
import sys

import config
import data_utils
from phase4_model_architecture import MultiBranchSequenceModel

logging.basicConfig(level=logging.INFO, format='%(message)s')


def predict_single_sample(model, sample_seq, target_scaler, device=None):
    """Run inference on a single sequence sample."""
    device = device or config.DEVICE
    model = model.to(device)
    model.eval()
    
    if isinstance(sample_seq, np.ndarray):
        sample_seq = torch.tensor(sample_seq, dtype=torch.float32)
    
    if sample_seq.dim() == 2:
        sample_seq = sample_seq.unsqueeze(0)  # add batch dim
    
    sample_seq = sample_seq.to(device)
    
    with torch.no_grad():
        mu_scaled = model(sample_seq)
    
    mu_original = target_scaler.inverse_transform(
        mu_scaled.cpu().numpy().reshape(-1, 1)
    ).flatten()[0]
    
    return mu_original


def main():
    data_utils.set_seed()
    device = config.DEVICE
    logging.info(f"Using device: {device}")
    
    # Load data
    try:
        X_seq_all, y_seq_all, _, feature_cols, cols = data_utils.load_and_build_data()
    except FileNotFoundError:
        logging.error("Missing engineered dataset. Run Phase 3 first.")
        sys.exit(1)
    
    seq_len = config.load_best_seq_len()
    input_size = len(feature_cols)
    
    # Load model
    if not os.path.exists(config.DL_MODEL_PATH):
        logging.error(f"Missing model at {config.DL_MODEL_PATH}. Run Phase 4 first.")
        sys.exit(1)
    
    model = MultiBranchSequenceModel(input_size=input_size, seq_len=seq_len)
    # BUG-14 fix: weights_only=True for security
    model.load_state_dict(torch.load(config.DL_MODEL_PATH, weights_only=True,
                                      map_location=device))
    
    # Load scaler
    target_scaler = joblib.load(config.TARGET_SCALER_PATH)
    
    # Test prediction on last sample
    test_sample = X_seq_all[-1]
    actual = y_seq_all[-1]
    
    prediction = predict_single_sample(model, test_sample, target_scaler, device)
    
    logging.info(f"\n=== Single Sample Prediction ===")
    logging.info(f"Actual:     {actual:.4f}")
    logging.info(f"Predicted:  {prediction:.4f}")
    logging.info(f"Error:      {abs(prediction - actual):.4f}")
    logging.info(f"================================")


if __name__ == "__main__":
    main()
