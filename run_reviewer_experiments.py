"""
SUREcast Reviewer Experiments Runner (1.2 - 1.6) — FAST VERSION
Optimized for CPU: 5 epochs max, batch_size=2048, patience=3
"""
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib, warnings
warnings.filterwarnings('ignore')

def set_seed(s=42):
    np.random.seed(s); torch.manual_seed(s)

set_seed(42)

# ==========================================
# LOAD DATA
# ==========================================
print("=" * 60)
print("LOADING DATA...")
print("=" * 60, flush=True)

df = pd.read_csv("data/engineered_dataset.csv")
target_col = "Sales"
if target_col not in df.columns:
    target_col = next((c for c in ['Sales per customer','Order Item Quantity'] if c in df.columns), df.columns[-1])
date_col = next((c for c in df.columns if 'date' in c.lower()), None)
if date_col: df = df.sort_values(date_col)
cat_group = 'Category Name' if 'Category Name' in df.columns else next((c for c in df.columns if 'category' in c.lower()), None)
region_group = 'Order Region' if 'Order Region' in df.columns else next((c for c in df.columns if 'region' in c.lower()), None)
ignore_cols = [target_col, cat_group, region_group, date_col, 'YearWeek']
from pandas.api.types import is_numeric_dtype
feature_cols = [c for c in df.columns if c not in ignore_cols and is_numeric_dtype(df[c])]
seq_len = 8; dl_features = len(feature_cols)

X_seq_all, y_seq_all, X_tab_all = [], [], []
for _, group in df.groupby([cat_group, region_group]):
    vals = group[feature_cols].values; targets = group[target_col].values; dates = group[date_col].values
    if len(vals) < seq_len:
        pad = seq_len - len(vals)
        vals = np.vstack([np.zeros((pad, vals.shape[1])), vals])
        targets = np.concatenate([np.zeros(pad), targets])
        dates = np.concatenate([np.full(pad, dates[0]), dates])
    for i in range(len(vals) - seq_len):
        X_seq_all.append(vals[i:i+seq_len]); y_seq_all.append(targets[i+seq_len]); X_tab_all.append(vals[i+seq_len-1])

X_seq_all = np.array(X_seq_all); y_seq_all = np.array(y_seq_all); X_tab_all = np.array(X_tab_all)
split_idx = int(len(X_seq_all) * 0.8)
X_dl_train, X_dl_val = X_seq_all[:split_idx], X_seq_all[split_idx:]
y_train, y_val = y_seq_all[:split_idx], y_seq_all[split_idx:]
X_ml_train, X_ml_val = X_tab_all[:split_idx], X_tab_all[split_idx:]

target_scaler = joblib.load("models/target_scaler.pkl")
y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
print(f"Data: {len(X_seq_all)} seqs, {len(X_dl_train)} train, {len(X_dl_val)} val, {dl_features} feats", flush=True)

# ==========================================
# FAST TRAINING FUNCTIONS
# ==========================================
from phase4_model_architecture import MultiBranchSequenceModel

BS = 2048  # large batch for speed
EPOCHS = 5
PATIENCE = 3

def fast_train(model, X_tr, y_tr, X_v, y_v, epochs=EPOCHS):
    opt = optim.Adam(model.parameters(), lr=0.002)
    crit = nn.MSELoss()
    tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr,dtype=torch.float32),torch.tensor(y_tr,dtype=torch.float32).unsqueeze(-1)),batch_size=BS,shuffle=True)
    v_loader = DataLoader(TensorDataset(torch.tensor(X_v,dtype=torch.float32),torch.tensor(y_v,dtype=torch.float32).unsqueeze(-1)),batch_size=BS,shuffle=False)
    best_vl=float('inf'); best_st=None; pat=0
    for ep in range(epochs):
        model.train()
        for xb,yb in tr_loader:
            opt.zero_grad(); loss=crit(model(xb),yb); loss.backward(); opt.step()
        model.eval(); vls=[]
        with torch.no_grad():
            for xb,yb in v_loader: vls.append(crit(model(xb),yb).item())
        vl=np.mean(vls)
        if vl<best_vl: best_vl=vl; pat=0; best_st={k:v.clone() for k,v in model.state_dict().items()}
        else:
            pat+=1
            if pat>=PATIENCE: break
    if best_st: model.load_state_dict(best_st)
    model.eval(); preds=[]
    with torch.no_grad():
        for xb,_ in v_loader: preds.extend(model(xb).squeeze(-1).numpy())
    return np.array(preds), model

def metrics(y_true, y_pred):
    mae=np.mean(np.abs(y_true-y_pred))
    rmse=np.sqrt(np.mean((y_true-y_pred)**2))
    mape=np.mean(np.abs((y_true-y_pred)/(y_true+1e-8)))*100
    r2=1-np.sum((y_true-y_pred)**2)/np.sum((y_true-np.mean(y_true))**2)
    return mae,rmse,mape,r2

# ==========================================
# 1.2: Standalone Transformer
# ==========================================
print("\n" + "="*60); print("EXPERIMENT 1.2: Standalone Transformer Baseline"); print("="*60, flush=True)

class StandaloneTransformer(nn.Module):
    def __init__(self, input_size, seq_len):
        super().__init__()
        self.proj = nn.Linear(input_size, 64)
        el = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(el, num_layers=2)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        return self.fc(torch.mean(self.transformer(self.proj(x)), dim=1))

set_seed(42)
tft_model = StandaloneTransformer(dl_features, seq_len)
tft_preds_s, tft_model = fast_train(tft_model, X_dl_train, y_train_scaled, X_dl_val, y_val_scaled)
tft_preds = target_scaler.inverse_transform(tft_preds_s.reshape(-1,1)).flatten()
mae,rmse,mape,r2 = metrics(y_val, tft_preds)
print(f"\n--- 1.2 RESULTS: Standalone Transformer ---")
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAPE: {mape:.2f}%")
print(f"  R2:   {r2:.4f}", flush=True)

# ==========================================
# 1.3: Channel Ablation
# ==========================================
print("\n" + "="*60); print("EXPERIMENT 1.3: Channel Ablation Study"); print("="*60, flush=True)

set_seed(42)
full_m = MultiBranchSequenceModel(input_size=dl_features, seq_len=seq_len)
full_preds_s, full_model = fast_train(full_m, X_dl_train, y_train_scaled, X_dl_val, y_val_scaled, epochs=EPOCHS)
full_preds = target_scaler.inverse_transform(full_preds_s.reshape(-1,1)).flatten()
full_mae, full_rmse, full_mape, full_r2 = metrics(y_val, full_preds)
print(f"\nFull Model Baseline -> MAE: {full_mae:.4f}, R2: {full_r2:.4f}\n", flush=True)

channels = ['cnn','lstm','gru','bilstm','transformer']
abl_results = []
for c in channels:
    set_seed(42)
    active = [ch for ch in channels if ch != c]
    print(f"Training WITHOUT {c.upper()}...", flush=True)
    m = MultiBranchSequenceModel(input_size=dl_features, seq_len=seq_len, active_channels=active)
    ps, _ = fast_train(m, X_dl_train, y_train_scaled, X_dl_val, y_val_scaled, epochs=EPOCHS)
    p = target_scaler.inverse_transform(ps.reshape(-1,1)).flatten()
    am, ar, ap, a2 = metrics(y_val, p)
    delta = am - full_mae
    abl_results.append((c.upper(), am, a2, delta))
    print(f"  -> MAE: {am:.4f} (Delta {delta:+.4f}), R2: {a2:.4f}", flush=True)

print(f"\n--- 1.3 RESULTS: Channel Ablation ---")
print(f"{'Channel Removed':<20} {'MAE':<12} {'R2':<12} {'Delta MAE':<12} {'Verdict'}")
print("-"*68)
print(f"{'NONE (Full Model)':<20} {full_mae:<12.4f} {full_r2:<12.4f} {'---':<12} {'Baseline'}")
for name,mae,r2,delta in abl_results:
    v = "REDUNDANT" if delta<0 else ("CRITICAL" if delta>full_mae*0.05 else "HELPFUL")
    print(f"{name:<20} {mae:<12.4f} {r2:<12.4f} {delta:<+12.4f} {v}")
sys.stdout.flush()

# ==========================================
# 1.4: Cross-Validation
# ==========================================
print("\n" + "="*60); print("EXPERIMENT 1.4: 3-Fold TimeSeriesSplit Cross-Validation"); print("="*60, flush=True)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge, BayesianRidge

tscv = TimeSeriesSplit(n_splits=3)
cv_dl, cv_ml, cv_hyb = [],[],[]

for fold,(tidx,vidx) in enumerate(tscv.split(X_seq_all),1):
    set_seed(42)
    print(f"\n--- CV Fold {fold} ---", flush=True)
    Xtr,Xv = X_seq_all[tidx],X_seq_all[vidx]
    Xmltr,Xmlv = X_tab_all[tidx],X_tab_all[vidx]
    ytr,yv = y_seq_all[tidx],y_seq_all[vidx]
    ytrs = target_scaler.transform(ytr.reshape(-1,1)).flatten()
    yvs = target_scaler.transform(yv.reshape(-1,1)).flatten()
    
    # DL
    m = MultiBranchSequenceModel(input_size=dl_features, seq_len=seq_len)
    ps, _ = fast_train(m, Xtr, ytrs, Xv, yvs, epochs=5)
    dp = target_scaler.inverse_transform(ps.reshape(-1,1)).flatten()
    dl_mae = np.mean(np.abs(dp - yv))
    cv_dl.append(dl_mae)
    
    # ML
    ml_models = {
        'RF': RandomForestRegressor(n_estimators=50, random_state=42),
        'GB': GradientBoostingRegressor(n_estimators=50, random_state=42),
        'EN': ElasticNet(random_state=42),
        'Ridge': Ridge(random_state=42),
        'BR': BayesianRidge()
    }
    ml_preds_list = []
    for nm,mdl in ml_models.items():
        mdl.fit(Xmltr, ytr)
        ml_preds_list.append(mdl.predict(Xmlv))
    ml_p = np.mean(ml_preds_list, axis=0)
    ml_mae = np.mean(np.abs(ml_p - yv))
    cv_ml.append(ml_mae)
    
    hyb = 0.5*dp + 0.5*ml_p
    hyb_mae = np.mean(np.abs(hyb - yv))
    cv_hyb.append(hyb_mae)
    print(f"  Fold {fold} -> DL: {dl_mae:.4f}, ML: {ml_mae:.4f}, Hybrid: {hyb_mae:.4f}", flush=True)

print(f"\n--- 1.4 RESULTS: Cross-Validation ---")
print(f"{'Model':<25} {'Mean MAE':<15} {'Std Dev':<15} {'Consistent?'}")
print("-"*65)
for nm,arr in [("DL Branch",cv_dl),("ML Ensemble",cv_ml),("Hybrid (0.5/0.5)",cv_hyb)]:
    cons = "YES" if np.std(arr)<np.mean(arr)*0.3 else "NO"
    print(f"{nm:<25} {np.mean(arr):<15.4f} {np.std(arr):<15.4f} {cons}")
sys.stdout.flush()

# ==========================================
# 1.5: Full Metrics Table
# ==========================================
print("\n" + "="*60); print("EXPERIMENT 1.5: Complete Metrics Table"); print("="*60, flush=True)

pred_df = pd.read_csv("data/model_predictions.csv")
yt = pred_df['Actual'].values

def snaive(y,s=4):
    p=np.zeros_like(y); p[:s]=y[:s]
    for i in range(s,len(y)): p[i]=y[i-s]
    return p

ya = pd.Series(yt).shift(1).rolling(window=2,min_periods=1).mean().fillna(np.mean(yt)).values
yn = snaive(yt)

models_eval = {
    "Seasonal Naive": yn,
    "ARIMA (MA Proxy)": ya,
    "Standalone Transformer": np.resize(tft_preds, len(yt)),
    "ML Ensemble": pred_df['ML_Pred'].values,
    "DL Branch (5-ch)": pred_df['DL_Pred'].values,
    "SUREcast Fixed Weight": pred_df['Hybrid_Fixed_Pred'].values,
    "SUREcast Stacking": pred_df['Hybrid_Stacking_Pred'].values,
}

print(f"\n{'Model':<25} {'MAE':<10} {'RMSE':<10} {'MAPE(%)':<10} {'R2':<10}")
print("-"*65)
for nm,p in models_eval.items():
    m,r,mp,r2 = metrics(yt,p)
    print(f"{nm:<25} {m:<10.4f} {r:<10.4f} {mp:<10.2f} {r2:<10.4f}")
sys.stdout.flush()

# ==========================================
# 1.6: Stress Test
# ==========================================
print("\n" + "="*60); print("EXPERIMENT 1.6: Synthetic Disruption Stress Test (3x Shock)"); print("="*60, flush=True)

set_seed(42)
n = len(y_val)
shock_idx = np.random.choice(n, size=int(n*0.1), replace=False)

X_sh = X_dl_val.copy(); y_sh = y_val.copy()
X_sh[shock_idx, -4:, 0] *= 3.0; y_sh[shock_idx] *= 3.0

print(f"Total test: {n}, Shocked (10%): {len(shock_idx)}, Magnitude: 3x\n", flush=True)

full_model.eval()
with torch.no_grad():
    pc_s = full_model(torch.tensor(X_dl_val,dtype=torch.float32)).numpy()
    ps_s = full_model(torch.tensor(X_sh,dtype=torch.float32)).numpy()

pc = target_scaler.inverse_transform(pc_s.reshape(-1,1)).flatten()
psh = target_scaler.inverse_transform(ps_s.reshape(-1,1)).flatten()

mae_c = np.mean(np.abs(pc - y_val)); mae_s = np.mean(np.abs(psh - y_sh))
mae_sh_sub = np.mean(np.abs(psh[shock_idx] - y_sh[shock_idx]))
mae_c_sub = np.mean(np.abs(pc[shock_idx] - y_val[shock_idx]))
non_sh = np.setdiff1d(np.arange(n), shock_idx)
mae_ns = np.mean(np.abs(psh[non_sh] - y_sh[non_sh]))
mae_cn = np.mean(np.abs(pc[non_sh] - y_val[non_sh]))

arima_c = pd.Series(y_val).shift(1).rolling(window=2,min_periods=1).mean().fillna(np.mean(y_val)).values
arima_s = pd.Series(y_sh).shift(1).rolling(window=2,min_periods=1).mean().fillna(np.mean(y_sh)).values
a_mc = np.mean(np.abs(arima_c - y_val)); a_ms = np.mean(np.abs(arima_s - y_sh))

print(f"--- 1.6 RESULTS: Disruption Stress Test ---\n")
print(f"{'Metric':<35} {'SUREcast':<15} {'ARIMA Proxy':<15}")
print("-"*65)
print(f"{'MAE (Clean Data)':<35} {mae_c:<15.4f} {a_mc:<15.4f}")
print(f"{'MAE (With 3x Shock)':<35} {mae_s:<15.4f} {a_ms:<15.4f}")
d1 = ((mae_s-mae_c)/mae_c)*100; d2 = ((a_ms-a_mc)/a_mc)*100
print(f"{'Degradation %':<35} {d1:<14.2f}% {d2:<14.2f}%")
print(f"\nSubset Analysis:")
print(f"{'MAE on shocked seqs only':<35} {mae_sh_sub:<15.4f}")
print(f"{'MAE on non-shocked seqs':<35} {mae_ns:<15.4f}")
print(f"{'MAE baseline (non-shocked,clean)':<35} {mae_cn:<15.4f}")
ns_drift = ((mae_ns-mae_cn)/mae_cn)*100
print(f"{'Non-shocked drift %':<35} {ns_drift:<14.2f}%")
sys.stdout.flush()

print("\n" + "="*60); print("ALL EXPERIMENTS COMPLETE"); print("="*60)
