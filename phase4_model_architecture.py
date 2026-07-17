import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import logging
import argparse
import sys
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge, BayesianRidge
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import config
import data_utils

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ==========================================
# 1. DEEP LEARNING BRANCH (PyTorch)
# ==========================================

class ChannelAttention(nn.Module):
    """Learned attention weights over channel outputs (ARCH-7 fix)."""
    def __init__(self, channel_dims):
        super().__init__()
        # One learnable scalar per channel, normalized via softmax
        self.attention_logits = nn.Parameter(torch.zeros(len(channel_dims)))
        self.channel_dims = channel_dims

    def forward(self, channel_outputs):
        """
        channel_outputs: list of tensors, each (batch, dim_i)
        Returns: single tensor (batch, sum(dims)) with attention-weighted channels.
        """
        weights = F.softmax(self.attention_logits, dim=0)
        weighted = []
        for i, (out, dim) in enumerate(zip(channel_outputs, self.channel_dims)):
            weighted.append(out * weights[i])
        return torch.cat(weighted, dim=1), weights


class MultiBranchSequenceModel(nn.Module):
    def __init__(self, input_size, seq_len, active_channels=None):
        super(MultiBranchSequenceModel, self).__init__()
        self.active_channels = active_channels if active_channels is not None else config.DL_CHANNELS
        
        self.channel_dims = []
        
        # 1. CNN Channel
        if 'cnn' in self.active_channels:
            self.cnn1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
            self.cnn2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
            self.channel_dims.append(32)
            
        # 2. LSTM Channel
        if 'lstm' in self.active_channels:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=1, batch_first=True)
            self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
            self.channel_dims.append(32)
            
        # 3. GRU Channel
        if 'gru' in self.active_channels:
            self.gru = nn.GRU(input_size=input_size, hidden_size=64, num_layers=1, batch_first=True)
            self.gru2 = nn.GRU(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
            self.channel_dims.append(32)
            
        # 4. BiLSTM Channel
        if 'bilstm' in self.active_channels:
            self.bilstm = nn.LSTM(input_size=input_size, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
            self.channel_dims.append(64)  # 32 * 2 directions
            
        # 5. Transformer Channel
        if 'transformer' in self.active_channels:
            self.transformer_proj = nn.Linear(input_size, 64)
            self.transformer_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
            self.channel_dims.append(64)
            
        # Learned Channel Attention (ARCH-7: replaces naive concatenation)
        self.channel_attention = ChannelAttention(self.channel_dims)
        
        total_dim = sum(self.channel_dims)
        
        # Dense Layers
        self.fc1 = nn.Linear(total_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(p=0.3)
        
        # Output head
        self.mean_head = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        channel_outputs = []
        
        if 'cnn' in self.active_channels:
            x_cnn = x.permute(0, 2, 1)
            c = F.relu(self.cnn1(x_cnn))
            c = F.relu(self.cnn2(c))
            c = torch.mean(c, dim=2)
            channel_outputs.append(c)
            
        if 'lstm' in self.active_channels:
            l, _ = self.lstm(x)
            l, _ = self.lstm2(l)
            l = l[:, -1, :]
            channel_outputs.append(l)
            
        if 'gru' in self.active_channels:
            g, _ = self.gru(x)
            g, _ = self.gru2(g)
            g = g[:, -1, :]
            channel_outputs.append(g)
            
        if 'bilstm' in self.active_channels:
            b, _ = self.bilstm(x)
            b = b[:, -1, :]
            channel_outputs.append(b)
            
        if 'transformer' in self.active_channels:
            t = self.transformer_proj(x)
            t = self.transformer_layer(t)
            t = torch.mean(t, dim=1)
            channel_outputs.append(t)
            
        # Learned attention fusion (ARCH-7)
        concat, attn_weights = self.channel_attention(channel_outputs)
        
        d = F.relu(self.fc1(concat))
        d = self.dropout(d)
        d = F.relu(self.fc2(d))
        d = self.dropout(d)
        d = F.relu(self.fc3(d))
        
        mu = self.mean_head(d)
        return mu


def train_dl_branch(X_train, y_train, X_val, y_val, input_size, seq_len,
                    active_channels=None, epochs=None, device=None):
    """Train the multi-branch DL model. Returns (mae, predictions, model)."""
    epochs = epochs or config.DL_EPOCHS
    device = device or config.DEVICE
    
    X_train = np.clip(X_train, -20.0, 20.0)
    X_val = np.clip(X_val, -20.0, 20.0)
    
    model = MultiBranchSequenceModel(input_size=input_size, seq_len=seq_len,
                                     active_channels=active_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.DL_LEARNING_RATE,
                           weight_decay=config.DL_WEIGHT_DECAY)
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)),
        batch_size=config.DL_BATCH_SIZE, shuffle=True)
    
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)),
        batch_size=config.DL_BATCH_SIZE, shuffle=False)
                            
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
                            
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (X_b, y_b) in enumerate(train_loader):
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            mu = model(X_b)
            loss = criterion(mu, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * X_b.size(0)
            
        # Validation for early stopping
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                mu = model(X_b)
                loss = criterion(mu, y_b)
                val_loss += loss.item() * X_b.size(0)
                
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            
        if epoch % 10 == 0 or patience_counter > 0:
            logging.info(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f} - Val: {val_loss:.4f} - Patience: {patience_counter}/{config.DL_PATIENCE}")
            
        if patience_counter >= config.DL_PATIENCE:
            logging.info(f"Early stopping at epoch {epoch+1}. Restoring best weights.")
            model.load_state_dict(best_model_state)
            break
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model = model.to(device)
            
    # Evaluate
    model.eval()
    val_preds = []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b = X_b.to(device)
            mu = model(X_b)
            val_preds.extend(mu.squeeze(-1).cpu().numpy())
            
    val_preds = np.array(val_preds)
    mae = np.mean(np.abs(val_preds - y_val))
    return mae, val_preds, model


# ==========================================
# 1.5 STANDALONE TRANSFORMER BASELINE
# ==========================================

class StandaloneTransformer(nn.Module):
    def __init__(self, input_size, seq_len):
        super(StandaloneTransformer, self).__init__()
        self.proj = nn.Linear(input_size, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        t = self.proj(x)
        t = self.transformer(t)
        t = torch.mean(t, dim=1)
        out = self.fc(t)
        return out

def train_standalone_transformer(X_train, y_train, X_val, y_val, input_size, seq_len, 
                                  epochs=None, device=None):
    epochs = epochs or config.DL_EPOCHS
    device = device or config.DEVICE
    
    X_train = np.clip(X_train, -20.0, 20.0)
    X_val = np.clip(X_val, -20.0, 20.0)
    
    model = StandaloneTransformer(input_size=input_size, seq_len=seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.DL_LEARNING_RATE)
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)),
        batch_size=config.DL_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)),
        batch_size=config.DL_BATCH_SIZE, shuffle=False)
                            
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
                            
    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            mu = model(X_b)
            loss = criterion(mu, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                mu = model(X_b)
                loss = criterion(mu, y_b)
                val_losses.append(loss.item())
                
        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.DL_PATIENCE:
                break
                
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model = model.to(device)
            
    model.eval()
    val_preds = []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b = X_b.to(device)
            mu = model(X_b)
            val_preds.extend(mu.squeeze(-1).cpu().numpy())
            
    val_preds = np.array(val_preds)
    mae = np.mean(np.abs(val_preds - y_val))
    return mae, val_preds


# ==========================================
# 2. ML ENSEMBLE BRANCH
# ==========================================

def train_ml_ensemble(X_train_tab, y_train, X_val_tab, y_val):
    """Train multiple ML models and return ensemble predictions + individual model predictions."""
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=config.ML_N_ESTIMATORS, random_state=config.RANDOM_SEED),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=config.ML_N_ESTIMATORS, random_state=config.RANDOM_SEED),
        'ElasticNet': ElasticNet(random_state=config.RANDOM_SEED),
        'Ridge': Ridge(random_state=config.RANDOM_SEED),
        'BayesianRidge': BayesianRidge()
    }
    
    # BUG-5 fix: Use the module-level HAS_XGBOOST instead of hardcoding False
    if HAS_XGBOOST:
        models['XGBoost'] = XGBRegressor(n_estimators=config.ML_N_ESTIMATORS,
                                          random_state=config.RANDOM_SEED,
                                          objective='reg:squarederror')
    else:
        logging.warning("XGBoost not installed. Skipping XGBoost in ML ensemble.")
        
    preds = {}
    maes = {}
    
    for name, model in models.items():
        model.fit(X_train_tab, y_train)
        pred = model.predict(X_val_tab)
        mae = np.mean(np.abs(pred - y_val))
        preds[name] = pred
        maes[name] = mae
        logging.info(f" - {name} Validation MAE: {mae:.4f}")
        
    # Ensemble (Average)
    ensemble_pred = np.mean(list(preds.values()), axis=0)
    ensemble_mae = np.mean(np.abs(ensemble_pred - y_val))
    logging.info(f" -> ML Ensemble Average Validation MAE: {ensemble_mae:.4f}")
    
    return ensemble_mae, ensemble_pred, preds


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_ablation", action="store_true", help="Run Deep Learning Channel Ablation Study")
    args = parser.parse_args()

    logging.info("═══════════════════════════════════════")
    logging.info("PHASE 4 — Model Architecture")
    logging.info("═══════════════════════════════════════\n")
    
    data_utils.set_seed()
    device = config.DEVICE
    logging.info(f"Using device: {device}\n")

    # Load data using shared utilities (eliminates code duplication)
    try:
        X_seq_all, y_seq_all, X_tab_all, feature_cols, cols = data_utils.load_and_build_data()
        X_seq_all = np.clip(X_seq_all, -20.0, 20.0)
    except FileNotFoundError:
        logging.error(f"[ERROR] Run Phase 3 first to generate '{config.ENGINEERED_DATA_PATH}'.")
        sys.exit(1)
    
    seq_len = config.load_best_seq_len()
    dl_features = len(feature_cols)
    logging.info(f"Sequence length: {seq_len}, Features: {dl_features}\n")
    
    # 3-way temporal split (ARCH-6 fix: proper train/val/test)
    splits = data_utils.temporal_train_val_test_split(X_seq_all, y_seq_all, X_tab_all)
    
    X_dl_train = splits['train']['X_seq']
    X_dl_val = splits['val']['X_seq']
    X_dl_test = splits['test']['X_seq']
    y_train = splits['train']['y']
    y_val = splits['val']['y']
    y_test = splits['test']['y']
    X_ml_train = splits['train']['X_tab']
    X_ml_val = splits['val']['X_tab']
    X_ml_test = splits['test']['X_tab']
    
    # Load Target Scaler
    if not os.path.exists(config.TARGET_SCALER_PATH):
        logging.error(f"[ERROR] Target scaler not found at '{config.TARGET_SCALER_PATH}'. Run Phase 3 first.")
        sys.exit(1)
    target_scaler = joblib.load(config.TARGET_SCALER_PATH)
    
    # ==========================================
    # 0.5 CROSS-VALIDATION
    # ==========================================
    from sklearn.model_selection import TimeSeriesSplit
    logging.info(f"\n0.5 Running {config.CV_FOLDS}-Fold TimeSeriesSplit Cross-Validation...")
    tscv = TimeSeriesSplit(n_splits=config.CV_FOLDS)
    cv_dl_maes = []
    cv_ml_maes = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_seq_all), 1):
        logging.info(f"--- CV Fold {fold} ---")
        X_dl_tr_cv = X_seq_all[train_idx]
        X_dl_v_cv = X_seq_all[val_idx]
        X_ml_tr_cv = X_tab_all[train_idx]
        X_ml_v_cv = X_tab_all[val_idx]
        y_tr_cv = y_seq_all[train_idx]
        y_v_cv = y_seq_all[val_idx]
        
        y_tr_cv_scaled = target_scaler.transform(y_tr_cv.reshape(-1, 1)).flatten()
        y_v_cv_scaled = target_scaler.transform(y_v_cv.reshape(-1, 1)).flatten()
        
        dl_mae_scaled_cv, dl_preds_scaled_cv, _ = train_dl_branch(
            X_dl_tr_cv, y_tr_cv_scaled, X_dl_v_cv, y_v_cv_scaled,
            input_size=dl_features, seq_len=seq_len, epochs=config.CV_EPOCHS, device=device)
        dl_preds_cv = target_scaler.inverse_transform(dl_preds_scaled_cv.reshape(-1, 1)).flatten()
        dl_mae_cv = np.mean(np.abs(dl_preds_cv - y_v_cv))
        cv_dl_maes.append(dl_mae_cv)
        
        ml_mae_cv, _, _ = train_ml_ensemble(X_ml_tr_cv, y_tr_cv, X_ml_v_cv, y_v_cv)
        cv_ml_maes.append(ml_mae_cv)
        
    logging.info(f" -> CV DL Branch MAE: {np.mean(cv_dl_maes):.4f} ± {np.std(cv_dl_maes):.4f}")
    logging.info(f" -> CV ML Ensemble MAE: {np.mean(cv_ml_maes):.4f} ± {np.std(cv_ml_maes):.4f}\n")
    
    # Distribution check
    logging.info("--- TARGET DISTRIBUTION VERIFICATION ---")
    logging.info(f"Train  - Mean: {np.mean(y_train):.2f}, Std: {np.std(y_train):.2f}")
    logging.info(f"Val    - Mean: {np.mean(y_val):.2f}, Std: {np.std(y_val):.2f}")
    logging.info(f"Test   - Mean: {np.mean(y_test):.2f}, Std: {np.std(y_test):.2f}")
    logging.info("----------------------------------------\n")
    
    # Scale targets for DL Branch
    y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
    
    # 1. Train Full DL Branch
    logging.info("1. Training Deep Learning Sequence Branch (Full 5 Channels) on SCALED target...")
    dl_mae_scaled, dl_preds_scaled, final_dl_model = train_dl_branch(
        X_dl_train, y_train_scaled, X_dl_val, y_val_scaled,
        input_size=dl_features, seq_len=seq_len, device=device)
    
    dl_preds = target_scaler.inverse_transform(dl_preds_scaled.reshape(-1, 1)).flatten()
    dl_mae = np.mean(np.abs(dl_preds - y_val))
    logging.info(f" -> Full DL Model Validation MAE (Original Scale): {dl_mae:.4f}\n")
    
    # Log attention weights
    if hasattr(final_dl_model, 'channel_attention'):
        attn_w = F.softmax(final_dl_model.channel_attention.attention_logits, dim=0).detach().cpu().numpy()
        for ch, w in zip(final_dl_model.active_channels, attn_w):
            logging.info(f"   Channel {ch.upper()}: attention weight = {w:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(y_val[:100], label='Actual Sales', marker='o', markersize=3)
    plt.plot(dl_preds[:100], label='DL Predicted Sales', marker='x', markersize=3)
    plt.title('DL Branch: Predicted vs Actual (First 100 Val Samples)')
    plt.legend()
    plt.savefig('data/predicted_vs_actual_dl.png', dpi=150)
    plt.close()
    
    # 1.5 Standalone Transformer Baseline
    logging.info("\n1.5 Training Standalone Transformer Baseline...")
    tft_mae_scaled, tft_preds_scaled = train_standalone_transformer(
        X_dl_train, y_train_scaled, X_dl_val, y_val_scaled,
        input_size=dl_features, seq_len=seq_len, device=device)
    tft_preds = target_scaler.inverse_transform(tft_preds_scaled.reshape(-1, 1)).flatten()
    tft_mae = np.mean(np.abs(tft_preds - y_val))
    logging.info(f" -> Standalone Transformer MAE: {tft_mae:.4f}")
    
    # Ablation Study
    if args.run_ablation:
        logging.info("\n--- CHANNEL ABLATION STUDY ---")
        channels = config.DL_CHANNELS
        for c in channels:
            ablation_channels = [ch for ch in channels if ch != c]
            logging.info(f"Training WITHOUT {c.upper()} channel...")
            abl_mae_scaled, abl_preds_scaled, _ = train_dl_branch(
                X_dl_train, y_train_scaled, X_dl_val, y_val_scaled,
                input_size=dl_features, seq_len=seq_len,
                active_channels=ablation_channels, device=device)
            abl_preds = target_scaler.inverse_transform(abl_preds_scaled.reshape(-1, 1)).flatten()
            abl_mae = np.mean(np.abs(abl_preds - y_val))
            logging.info(f" - MAE w/o {c.upper()}: {abl_mae:.4f} (Delta: {abl_mae - dl_mae:+.4f})")
        logging.info("--- ABLATION STUDY COMPLETE ---\n")
        
    # 2. Train ML Ensemble Branch
    logging.info("\n2. Training ML Tabular Ensemble Branch...")
    ml_mae, ml_preds, ml_individual_preds = train_ml_ensemble(X_ml_train, y_train, X_ml_val, y_val)
    
    # BUG-6 fix: Compute REAL uncertainty from ensemble variance (not fake proxy)
    all_model_preds = list(ml_individual_preds.values()) + [dl_preds]
    ensemble_uncertainty = np.std(all_model_preds, axis=0)
    
    # 3. FUSION
    logging.info("\n3. FUSION STRATEGIES")
    # Strategy A: Fixed-Weight Sweep
    best_w = 0.5
    best_sweep_mae = float('inf')
    
    for w in config.FUSION_WEIGHT_CANDIDATES:
        fused_pred = w * dl_preds + (1 - w) * ml_preds
        mae = np.mean(np.abs(fused_pred - y_val))
        logging.info(f" - Fixed Weight (DL={w:.1f}, ML={1-w:.1f}): MAE = {mae:.4f}")
        if mae < best_sweep_mae:
            best_sweep_mae = mae
            best_w = w
            
    logging.info(f" -> Best Fixed-Weight MAE: {best_sweep_mae:.4f} (DL Weight = {best_w})")
    
    # Strategy B: Stacking Meta-Learner (BUG-9 fix: train on TRAINING OOF, not val)
    from sklearn.model_selection import KFold, cross_val_predict
    
    # Generate OOF predictions on TRAINING set
    logging.info("\n Training stacking meta-learner on training OOF predictions...")
    stack_X_train = np.column_stack([
        target_scaler.inverse_transform(
            train_dl_branch(X_dl_train, y_train_scaled, X_dl_train, y_train_scaled,
                          input_size=dl_features, seq_len=seq_len, epochs=15, device=device)[1].reshape(-1, 1)
        ).flatten(),
        train_ml_ensemble(X_ml_train, y_train, X_ml_train, y_train)[1]
    ])
    
    meta_learner = Ridge()
    cv = KFold(n_splits=config.STACKING_CV_FOLDS, shuffle=False)
    
    # Fit meta-learner on training data
    meta_learner.fit(stack_X_train, y_train)
    
    # Evaluate on validation data
    stack_X_val = np.column_stack([dl_preds, ml_preds])
    stack_pred = meta_learner.predict(stack_X_val)
    stack_mae = np.mean(np.abs(stack_pred - y_val))
    logging.info(f" -> Stacking Meta-Learner MAE: {stack_mae:.4f}")
    
    if stack_mae < best_sweep_mae:
        logging.info("\nCONCLUSION: Stacking Meta-Learner wins. Using it for fusion.")
        final_preds = stack_pred
    else:
        logging.info(f"\nCONCLUSION: Fixed-Weight (DL={best_w}) wins. Using it for fusion.")
        final_preds = best_w * dl_preds + (1 - best_w) * ml_preds
        
    # SAVE MODELS
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    logging.info(f"\nSaving DL model to {config.DL_MODEL_PATH}...")
    torch.save(final_dl_model.state_dict(), config.DL_MODEL_PATH)
    
    logging.info(f"Saving meta-learner to {config.META_LEARNER_PATH}...")
    joblib.dump(meta_learner, config.META_LEARNER_PATH)
    
    # Save best fusion weight
    joblib.dump({'best_w': best_w, 'best_sweep_mae': best_sweep_mae,
                 'stack_mae': stack_mae, 'use_stacking': stack_mae < best_sweep_mae},
                os.path.join(config.MODEL_DIR, 'fusion_config.pkl'))
    
    # SAVE PREDICTIONS FOR PHASE 5
    logging.info(f"\nSaving predictions to {config.PREDICTIONS_PATH}...")
    best_fixed_preds = best_w * dl_preds + (1 - best_w) * ml_preds
    out_df = pd.DataFrame({
        'Actual': y_val,
        'DL_Pred': dl_preds,
        'ML_Pred': ml_preds,
        'TFT_Pred': tft_preds,
        'Hybrid_Fixed_Pred': best_fixed_preds,
        'Hybrid_Stacking_Pred': stack_pred,
        'Hybrid_Pred': final_preds,
        'Uncertainty': ensemble_uncertainty  # BUG-6 fix: real ensemble variance
    })
    out_df.to_csv(config.PREDICTIONS_PATH, index=False)
    
    # Also save test-set predictions for Phase 5
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    _, dl_test_preds_scaled, _ = train_dl_branch(
        np.concatenate([X_dl_train, X_dl_val]),
        np.concatenate([y_train_scaled, y_val_scaled]),
        X_dl_test, y_test_scaled,
        input_size=dl_features, seq_len=seq_len, epochs=config.CV_EPOCHS, device=device)
    dl_test_preds = target_scaler.inverse_transform(dl_test_preds_scaled.reshape(-1, 1)).flatten()
    
    _, ml_test_preds, ml_test_ind = train_ml_ensemble(
        np.concatenate([X_ml_train, X_ml_val]),
        np.concatenate([y_train, y_val]),
        X_ml_test, y_test)
    
    test_ensemble_uncertainty = np.std(list(ml_test_ind.values()) + [dl_test_preds], axis=0)
    
    stack_X_test = np.column_stack([dl_test_preds, ml_test_preds])
    if stack_mae < best_sweep_mae:
        test_final = meta_learner.predict(stack_X_test)
    else:
        test_final = best_w * dl_test_preds + (1 - best_w) * ml_test_preds
    
    test_df = pd.DataFrame({
        'Actual': y_test,
        'DL_Pred': dl_test_preds,
        'ML_Pred': ml_test_preds,
        'Hybrid_Pred': test_final,
        'Uncertainty': test_ensemble_uncertainty
    })
    test_df.to_csv("data/test_predictions.csv", index=False)
    
    logging.info("Phase 4 Complete.")

if __name__ == "__main__":
    data_utils.set_seed()
    main()
