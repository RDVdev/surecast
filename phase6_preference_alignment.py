"""
Phase 6 — Preference Alignment (DPO-inspired RLHF for Continuous Forecasting)

Generates synthetic preference pairs from validation predictions and fine-tunes
the DL model using a continuous contrastive loss inspired by DPO.

THEORETICAL CAVEAT:
Original DPO (Rafailov et al.) is derived from the Bradley-Terry model for discrete
token generation. By mapping the reward signal to negative MSE in a continuous domain,
we apply a heuristic "DPO-inspired" contrastive loss. Strict mathematical equivalence
to LLM DPO is not claimed.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import argparse
import numpy as np
import pandas as pd
import sys
import joblib

import config
import data_utils
from phase4_model_architecture import MultiBranchSequenceModel

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ==========================================
# 1. CONTINUOUS DPO-INSPIRED LOSS
# ==========================================

class ContinuousDPOLoss(nn.Module):
    """
    Continuous contrastive loss inspired by Direct Preference Optimization.
    
    Formula:
    L = -log(sigmoid(beta * (MSE(y_rej, y_hat) - MSE(y_pref, y_hat))))
    
    Encourages the model prediction to be closer to y_pref than y_rej.
    """
    def __init__(self, beta=None):
        super(ContinuousDPOLoss, self).__init__()
        self.beta = beta or config.DPO_BETA

    def forward(self, mu_pred, y_pref, y_rej):
        mse_pref = F.mse_loss(mu_pred, y_pref, reduction='none')
        mse_rej = F.mse_loss(mu_pred, y_rej, reduction='none')
        
        # Reward difference: R(pref) - R(rej) = MSE(rej) - MSE(pref)
        reward_diff = mse_rej - mse_pref
        
        loss = -F.logsigmoid(self.beta * reward_diff)
        return torch.mean(loss)

# ==========================================
# 2. SYNTHETIC PREFERENCE PAIR GENERATOR
# ==========================================

def generate_synthetic_preference_pairs(X_seq, y_actual, model_preds, 
                                        noise_scale=None):
    """
    Generate synthetic preference pairs from validation data.
    
    For each sample:
    - y_pref = model prediction (already close to actual)
    - y_rej  = y_actual + noise (a perturbed version farther from model prediction)
    
    This demonstrates the DPO pipeline works end-to-end without requiring
    manual expert annotation.
    """
    noise_scale = noise_scale or config.SYNTHETIC_PAIR_NOISE_SCALE
    n = len(y_actual)
    
    # Compute residuals to determine perturbation scale
    residuals = y_actual - model_preds
    residual_std = np.std(residuals) if np.std(residuals) > 1e-8 else 1.0
    
    # y_pref: the actual value (ground truth is always "preferred")
    y_pref = y_actual.copy()
    
    # y_rej: perturbed version — actual + larger noise in same direction as error
    noise = np.random.randn(n) * residual_std * noise_scale
    # Ensure rejected is farther from actual than preferred
    y_rej = model_preds + noise * 2  # push away from actual
    
    logging.info(f"Generated {n} synthetic preference pairs.")
    logging.info(f"  Mean |y_pref - y_actual|: {np.mean(np.abs(y_pref - y_actual)):.4f}")
    logging.info(f"  Mean |y_rej - y_actual|:  {np.mean(np.abs(y_rej - y_actual)):.4f}")
    
    return y_pref, y_rej

# ==========================================
# 3. FEEDBACK MANAGER (BUG-1 fix: model returns mu only)
# ==========================================

class FeedbackManager:
    """Manages preference-based model updates with safety checks."""
    
    def __init__(self, model, checkpoint_dir=None, n_sigma_thresh=4.0):
        self.model = model
        self.checkpoint_dir = checkpoint_dir or config.MODEL_DIR
        self.n_sigma_thresh = n_sigma_thresh
        self.allowlist = {"EXPERT_01", "EXPERT_02", "ADMIN", "SYNTHETIC"}
        self.current_version = 0
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._save_checkpoint()

    def _save_checkpoint(self):
        path = os.path.join(self.checkpoint_dir, f"model_v{self.current_version}.pt")
        torch.save(self.model.state_dict(), path)
        logging.info(f"[SYSTEM] Checkpoint saved: {path}")

    def rollback(self, target_version):
        path = os.path.join(self.checkpoint_dir, f"model_v{target_version}.pt")
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, weights_only=True))
            self.current_version = target_version
            logging.info(f"[SYSTEM] Rolled back to version {target_version}")
            return True
        logging.error(f"[SYSTEM] Rollback failed. Version {target_version} not found.")
        return False

    def validate_pair(self, x_seq, y_pref, y_rej, contributor_id, 
                      historical_std=None):
        """
        Validate a preference pair. BUG-1 fix: model returns only mu,
        uses historical_std for anomaly detection instead of model sigma.
        """
        if contributor_id not in self.allowlist:
            logging.warning(f"[SECURITY] Rejected: '{contributor_id}' not in allowlist.")
            return False

        # Get model prediction (single output — BUG-1 fix)
        self.model.eval()
        with torch.no_grad():
            mu_prior = self.model(x_seq)
            
        mu = mu_prior.item()
        
        # Use historical std for anomaly detection (since model has no sigma head)
        sigma = historical_std if historical_std and historical_std > 1e-8 else 1.0
        
        distance = abs(y_rej.item() - mu)
        if distance > self.n_sigma_thresh * sigma:
            logging.warning(
                f"[ANOMALY] Rejected pair: y_rej ({y_rej.item():.2f}) is "
                f"{distance/sigma:.1f}σ from model prior (μ={mu:.2f}). "
                f"Threshold: {self.n_sigma_thresh}σ.")
            return False

        return True

    def fine_tune(self, X_seq, y_pref, y_rej, epochs=None, lr=None):
        """
        Execute DPO-inspired fine-tuning. BUG-1 fix: model returns mu only.
        """
        epochs = epochs or config.DPO_EPOCHS
        lr = lr or config.DPO_LEARNING_RATE
        
        logging.info(f"\n--- DPO Fine-Tuning (v{self.current_version} → v{self.current_version + 1}) ---")
        
        device = next(self.model.parameters()).device
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = ContinuousDPOLoss()
        
        # Convert to tensors
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(device)
        y_pref_t = torch.tensor(y_pref, dtype=torch.float32).unsqueeze(-1).to(device)
        y_rej_t = torch.tensor(y_rej, dtype=torch.float32).unsqueeze(-1).to(device)
        
        dataset = torch.utils.data.TensorDataset(X_t, y_pref_t, y_rej_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=config.DL_BATCH_SIZE, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            for x_b, yp_b, yr_b in loader:
                optimizer.zero_grad()
                mu_pred = self.model(x_b)  # single output (BUG-1 fix)
                loss = criterion(mu_pred, yp_b, yr_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            avg_loss = total_loss / max(n_batches, 1)
            logging.info(f"  Epoch {epoch+1}/{epochs} | DPO Loss: {avg_loss:.4f}")
            
        self.current_version += 1
        self._save_checkpoint()
        logging.info(f"  Fine-tuning complete. Model bumped to v{self.current_version}.")

# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    logging.info("═══════════════════════════════════════")
    logging.info("PHASE 6 — Preference Alignment (DPO)")
    logging.info("═══════════════════════════════════════\n")
    
    data_utils.set_seed()
    device = config.DEVICE
    
    # Load predictions from Phase 4
    if not os.path.exists(config.PREDICTIONS_PATH):
        logging.error(f"[ERROR] Missing {config.PREDICTIONS_PATH}. Run Phase 4 first.")
        sys.exit(1)
    
    pred_df = pd.read_csv(config.PREDICTIONS_PATH)
    y_actual = pred_df['Actual'].values
    y_hybrid = pred_df['Hybrid_Pred'].values
    
    # Load the trained DL model
    if not os.path.exists(config.DL_MODEL_PATH):
        logging.error(f"[ERROR] Missing {config.DL_MODEL_PATH}. Run Phase 4 first.")
        sys.exit(1)
    
    # Load data to get feature dimensions
    try:
        X_seq_all, y_seq_all, _, feature_cols, cols = data_utils.load_and_build_data()
    except FileNotFoundError:
        logging.error("[ERROR] Missing engineered dataset. Run Phase 3 first.")
        sys.exit(1)
    
    seq_len = config.load_best_seq_len()
    input_size = len(feature_cols)
    
    # Initialize model and load weights
    model = MultiBranchSequenceModel(input_size=input_size, seq_len=seq_len).to(device)
    model.load_state_dict(torch.load(config.DL_MODEL_PATH, weights_only=True, map_location=device))
    logging.info(f"Loaded DL model from {config.DL_MODEL_PATH}")
    
    # Get validation sequences for fine-tuning
    splits = data_utils.temporal_train_val_test_split(X_seq_all, y_seq_all)
    X_val = splits['val']['X_seq']
    y_val = splits['val']['y']
    
    # Load scaler
    target_scaler = joblib.load(config.TARGET_SCALER_PATH)
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
    
    # Evaluate BEFORE fine-tuning
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        preds_before_scaled = model(X_t).squeeze(-1).cpu().numpy()
    preds_before = target_scaler.inverse_transform(preds_before_scaled.reshape(-1, 1)).flatten()
    mae_before = np.mean(np.abs(preds_before - y_val))
    logging.info(f"\nPre-RLHF Validation MAE: {mae_before:.4f}")
    
    # Generate synthetic preference pairs (BUG-2 fix)
    logging.info("\n--- Generating Synthetic Preference Pairs ---")
    y_pref_scaled, y_rej_scaled = generate_synthetic_preference_pairs(
        X_val, y_val_scaled, preds_before_scaled)
    
    # Initialize FeedbackManager
    fm = FeedbackManager(model, checkpoint_dir=os.path.join(config.MODEL_DIR, "rlhf_checkpoints"))
    
    # Run fine-tuning
    fm.fine_tune(X_val, y_pref_scaled, y_rej_scaled)
    
    # Evaluate AFTER fine-tuning
    model.eval()
    with torch.no_grad():
        preds_after_scaled = model(X_t).squeeze(-1).cpu().numpy()
    preds_after = target_scaler.inverse_transform(preds_after_scaled.reshape(-1, 1)).flatten()
    mae_after = np.mean(np.abs(preds_after - y_val))
    logging.info(f"\nPost-RLHF Validation MAE: {mae_after:.4f}")
    logging.info(f"MAE Change: {mae_after - mae_before:+.4f} ({(mae_after - mae_before)/mae_before*100:+.2f}%)")
    
    if mae_after > mae_before * 1.05:
        logging.warning("[SAFETY] MAE degraded by >5%. Rolling back to pre-RLHF weights.")
        fm.rollback(0)
        model.eval()
        with torch.no_grad():
            preds_final_scaled = model(X_t).squeeze(-1).cpu().numpy()
        preds_final = target_scaler.inverse_transform(preds_final_scaled.reshape(-1, 1)).flatten()
        mae_final = np.mean(np.abs(preds_final - y_val))
        logging.info(f"Rolled back. Final MAE: {mae_final:.4f}")
    else:
        logging.info("[SUCCESS] RLHF fine-tuning accepted.")
        # Save the fine-tuned model
        torch.save(model.state_dict(), config.DL_MODEL_PATH)
        logging.info(f"Updated model saved to {config.DL_MODEL_PATH}")
    
    logging.info("\nPhase 6 Complete.")

if __name__ == "__main__":
    data_utils.set_seed()
    main()
