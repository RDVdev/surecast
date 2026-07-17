"""
Robustness Audit — Multi-seed sensitivity, walk-forward CV, and stress testing.
Uses shared data_utils to eliminate duplicated sequence-building code.
"""
import numpy as np
import torch
import logging
import sys
import os
import joblib

import config
import data_utils
from phase4_model_architecture import train_dl_branch

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ==========================================
# Synthetic Disruption Stress Test
# ==========================================

def run_synthetic_stress_test(X_test, y_test_orig, model, target_scaler, device=None):
    """
    Injects a 3x demand shock to 10% of sequences to simulate supply chain disruption.
    """
    device = device or config.DEVICE
    logging.info("\n--- Synthetic Disruption Stress Test ---")
    
    X_shocked = X_test.copy()
    y_shocked = y_test_orig.copy()
    
    num_sequences = len(y_shocked)
    shock_indices = np.random.choice(num_sequences, size=int(num_sequences * 0.1), replace=False)
    
    # Apply 3x multiplier to the last 4 timesteps of the selected sequences
    X_shocked[shock_indices, -4:, 0] *= 3.0
    y_shocked[shock_indices] *= 3.0
    
    logging.info(f"Injected 3x demand shock into {len(shock_indices)} sequences.")
    
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        preds_clean_scaled = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        preds_shocked_scaled = model(torch.tensor(X_shocked, dtype=torch.float32).to(device)).cpu().numpy()
    
    preds_clean = target_scaler.inverse_transform(preds_clean_scaled.reshape(-1, 1)).flatten()
    preds_shocked = target_scaler.inverse_transform(preds_shocked_scaled.reshape(-1, 1)).flatten()
    
    mae_clean = np.mean(np.abs(preds_clean - y_test_orig))
    mae_shocked = np.mean(np.abs(preds_shocked - y_shocked))
    
    logging.info(f"Baseline MAE (No Shock): {mae_clean:.4f}")
    logging.info(f"Disrupted MAE (With Shock): {mae_shocked:.4f}")
    degradation = ((mae_shocked - mae_clean) / mae_clean) * 100
    logging.info(f"Performance degradation: {degradation:.2f}%\n")
    return mae_clean, mae_shocked


def main():
    logging.info("═══════════════════════════════════════")
    logging.info("ROBUSTNESS AUDIT: Deep Learning Branch")
    logging.info("═══════════════════════════════════════\n")
    
    data_utils.set_seed()
    device = config.DEVICE
    logging.info(f"Using device: {device}\n")
    
    # Load data using shared utilities (eliminates duplicated code)
    try:
        X_all, y_all, _, feature_cols, cols = data_utils.load_and_build_data()
    except FileNotFoundError:
        logging.error("[ERROR] Missing engineered dataset. Run Phase 3 first.")
        sys.exit(1)
    
    input_size = len(feature_cols)
    seq_len = config.load_best_seq_len()
    target_scaler = joblib.load(config.TARGET_SCALER_PATH)
    
    y_all_scaled = target_scaler.transform(y_all.reshape(-1, 1)).flatten()
    
    # Use proper 3-way split
    splits = data_utils.temporal_train_val_test_split(X_all, y_all)
    X_train = splits['train']['X_seq']
    X_val = splits['val']['X_seq']
    y_train_scaled = target_scaler.transform(splits['train']['y'].reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(splits['val']['y'].reshape(-1, 1)).flatten()
    y_val_orig = splits['val']['y']
    
    # 1. Initialization Sensitivity (5 Random Seeds)
    logging.info("1. Initialization Sensitivity (5 Random Seeds)")
    seeds = [42, 100, 2026, 777, 12345]
    maes = []
    
    for s in seeds:
        data_utils.set_seed(s)
        logging.info(f" -> Training with seed {s}...")
        mae_scaled, preds_scaled, _ = train_dl_branch(
            X_train, y_train_scaled, X_val, y_val_scaled,
            input_size, seq_len=seq_len, epochs=30, device=device)
        preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        mae_orig = np.mean(np.abs(preds - y_val_orig))
        maes.append(mae_orig)
        logging.info(f"    Validation MAE: {mae_orig:.4f}")
        
    logging.info(f"\n[RESULTS] 5-Seed MAE: Mean = {np.mean(maes):.4f}, Std = {np.std(maes):.4f}")
    
    # 2. Walk-Forward Validation (3 Folds)
    logging.info("\n2. Walk-Forward Validation (3 Folds)")
    chunk_size = int(len(X_all) * 0.25)
    wf_maes = []
    final_model = None
    
    for fold in range(1, 4):
        data_utils.set_seed()
        train_end = chunk_size * fold
        val_end = chunk_size * (fold + 1)
        
        X_wf_train = X_all[:train_end]
        y_wf_train = y_all_scaled[:train_end]
        X_wf_val = X_all[train_end:val_end]
        y_wf_val = y_all_scaled[train_end:val_end]
        y_wf_val_orig = y_all[train_end:val_end]
        
        logging.info(f" -> Fold {fold}: Train={len(X_wf_train)}, Val={len(X_wf_val)}")
        mae_scaled, preds_scaled, final_model = train_dl_branch(
            X_wf_train, y_wf_train, X_wf_val, y_wf_val,
            input_size, seq_len=seq_len, epochs=30, device=device)
        preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        mae_orig = np.mean(np.abs(preds - y_wf_val_orig))
        wf_maes.append(mae_orig)
        logging.info(f"    Fold {fold} MAE: {mae_orig:.4f}")
        
    logging.info(f"\n[RESULTS] Walk-Forward MAE: Mean = {np.mean(wf_maes):.4f}, Std = {np.std(wf_maes):.4f}")
    
    # 3. Synthetic Stress Test
    if final_model is not None:
        X_wf_val = X_all[chunk_size * 3:]
        y_wf_val_orig = y_all[chunk_size * 3:]
        run_synthetic_stress_test(X_wf_val, y_wf_val_orig, final_model, target_scaler, device)
    
    logging.info("\nRobustness Audit Complete.")

if __name__ == "__main__":
    data_utils.set_seed()
    main()
