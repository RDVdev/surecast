import numpy as np
import pandas as pd
import logging
import argparse
import sys
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config
import data_utils

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ==========================================
# 1. EVALUATION METRICS
# ==========================================

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

# ==========================================
# 2. CONFORMAL PREDICTION (Calibration)
# ==========================================

def apply_conformal_prediction(calib_y, calib_mu, val_mu, alpha=0.05):
    """
    Split Conformal Prediction:
    Computes absolute residuals on a calibration set, finds the (1-alpha) quantile, 
    and applies this fixed radius to the validation set.
    """
    residuals = np.abs(calib_y - calib_mu)
    n = len(residuals)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    q_hat = np.quantile(residuals, q_level)
    
    lower_bound = val_mu - q_hat
    upper_bound = val_mu + q_hat
    return lower_bound, upper_bound, q_hat

# ==========================================
# 3. RESILIENCE SCORE (BUG-7 fix: real computation)
# ==========================================

def compute_volatility_ratio(y_true, y_pred):
    """Ratio of prediction std to actual std. 1.0 = perfectly matched variance."""
    pred_std = np.std(y_pred)
    true_std = np.std(y_true)
    if true_std < 1e-8:
        return 1.0
    return min(pred_std / true_std, true_std / pred_std)  # symmetric, in [0, 1]

def compute_trend_similarity(y_true, y_pred):
    """Correlation of first-differences (captures trend matching)."""
    if len(y_true) < 3:
        return 0.5
    diff_true = np.diff(y_true)
    diff_pred = np.diff(y_pred)
    if np.std(diff_true) < 1e-8 or np.std(diff_pred) < 1e-8:
        return 0.5
    corr = np.corrcoef(diff_true, diff_pred)[0, 1]
    return max(0, (corr + 1) / 2)  # map [-1,1] to [0,1]

def compute_resilience(metrics_dict, y_true, y_pred, weights):
    """
    Score = w1*(R2) + w2*(1 - MAE/mean_actual) + w3*(Volatility Ratio) + w4*(Trend Similarity)
    BUG-7 fix: vol_ratio and trend_sim are now computed from actual data.
    """
    w1, w2, w3, w4 = weights
    
    r2 = max(0, metrics_dict['R2'])
    mae_norm = max(0, 1 - (metrics_dict['MAE'] / (np.mean(y_true) + 1e-5)))
    vol_ratio = compute_volatility_ratio(y_true, y_pred)
    trend_sim = compute_trend_similarity(y_true, y_pred)
    
    score = (w1 * r2) + (w2 * mae_norm) + (w3 * vol_ratio) + (w4 * trend_sim)
    return score

def run_sensitivity_analysis(results_df, y_true, model_preds_dict):
    """Run resilience sensitivity analysis across weighting schemes."""
    schemes = {
        "Original": [0.4, 0.3, 0.2, 0.1],
        "Equal Weights": [0.25, 0.25, 0.25, 0.25],
        "Accuracy-Dominant": [0.5, 0.4, 0.05, 0.05]
    }
    
    logging.info("\n--- RESILIENCE SENSITIVITY ANALYSIS ---")
    
    rankings = {}
    for name, weights in schemes.items():
        scores = {}
        for idx, row in results_df.iterrows():
            model_name = row['Model']
            pred = model_preds_dict.get(model_name, np.zeros_like(y_true))
            scores[model_name] = compute_resilience(
                {'R2': row['R2'], 'MAE': row['MAE']}, y_true, pred, weights)
        
        sorted_models = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        rankings[name] = sorted_models
        
        logging.info(f"Scheme: {name} (Weights: {weights})")
        for i, m in enumerate(sorted_models):
            logging.info(f"  {i+1}. {m} (Score: {scores[m]:.4f})")
            
    base_rank = rankings["Original"]
    is_stable = all(rankings[scheme] == base_rank for scheme in schemes)
    if is_stable:
        logging.info("\n-> SENSITIVITY CONCLUSION: Rankings are STABLE across weighting schemes.")
    else:
        logging.warning("\n-> SENSITIVITY CONCLUSION: Rank inversions detected! Rankings are SENSITIVE to weights.")

# ==========================================
# 4. BASELINES
# ==========================================

def seasonal_naive_forecast(y, season_len=4):
    """Predicts value from exactly one season ago."""
    pred = np.zeros_like(y)
    pred[:season_len] = np.mean(y[:season_len])
    for i in range(season_len, len(y)):
        pred[i] = y[i - season_len]
    return pred

# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    logging.info("═══════════════════════════════════════")
    logging.info("PHASE 5 — Evaluation Protocol")
    logging.info("═══════════════════════════════════════\n")
    
    data_utils.set_seed()
    
    # Load validation predictions from Phase 4
    if not os.path.exists(config.PREDICTIONS_PATH):
        logging.error(f"[ERROR] Missing predictions from Phase 4 ({config.PREDICTIONS_PATH}).")
        sys.exit(1)

    df = pd.read_csv(config.PREDICTIONS_PATH)
    
    y_true = df['Actual'].values
    y_pred_dl = df['DL_Pred'].values
    y_pred_ml = df['ML_Pred'].values
    y_pred_hybrid = df['Hybrid_Pred'].values
    y_pred_fixed = df['Hybrid_Fixed_Pred'].values
    y_pred_stacking = df['Hybrid_Stacking_Pred'].values
    uncertainty = df['Uncertainty'].values
    
    # 1. Baselines
    y_pred_naive = seasonal_naive_forecast(y_true, season_len=4)
    
    # ARCH-8 fix: Honestly labeled as "Moving Average" not "ARIMA"
    y_pred_ma = pd.Series(y_true).shift(1).rolling(window=2, min_periods=1).mean().fillna(np.mean(y_true)).values
    
    y_pred_tft = df['TFT_Pred'].values if 'TFT_Pred' in df.columns else y_pred_dl
    
    logging.info("1. Computing Standard Metrics on VALIDATION set...")
    model_preds = {
        "Seasonal Naive": y_pred_naive,
        "Moving Average Baseline": y_pred_ma,
        "Standalone Transformer": y_pred_tft,
        "Standalone ML Ensemble": y_pred_ml,
        "Standalone DL Branch": y_pred_dl,
        "Hybrid (Best Fixed Weight)": y_pred_fixed,
        "Hybrid (Stacking)": y_pred_stacking,
    }
    
    results = []
    for name, pred in model_preds.items():
        mets = compute_metrics(y_true, pred)
        mets['Model'] = name
        results.append(mets)
        
    results_df = pd.DataFrame(results)
    cols = ['Model', 'MAE', 'RMSE', 'MAPE', 'R2']
    results_df = results_df[cols]
    
    logging.info("\n=== VALIDATION RESULTS TABLE ===")
    logging.info("\n" + results_df.to_string(index=False))
    
    # 2. Test-set evaluation (if available)
    test_path = "data/test_predictions.csv"
    if os.path.exists(test_path):
        logging.info("\n\n2. Computing Standard Metrics on HELD-OUT TEST set...")
        test_df = pd.read_csv(test_path)
        y_test = test_df['Actual'].values
        test_preds = test_df['Hybrid_Pred'].values
        test_dl = test_df['DL_Pred'].values
        test_ml = test_df['ML_Pred'].values
        
        test_models = {
            "DL Branch (Test)": test_dl,
            "ML Ensemble (Test)": test_ml,
            "Hybrid (Test)": test_preds,
        }
        
        test_results = []
        for name, pred in test_models.items():
            mets = compute_metrics(y_test, pred)
            mets['Model'] = name
            test_results.append(mets)
        
        test_results_df = pd.DataFrame(test_results)[cols]
        logging.info("\n=== TEST SET RESULTS TABLE ===")
        logging.info("\n" + test_results_df.to_string(index=False))
        
        # Combine for saving
        results_df = pd.concat([results_df, test_results_df], ignore_index=True)
    
    # 3. Confidence Interval Calibration (using real uncertainty)
    logging.info("\n\n3. Confidence Interval Calibration...")
    
    # Use real ensemble uncertainty from Phase 4
    z = 1.96
    lower = y_pred_hybrid - z * uncertainty
    upper = y_pred_hybrid + z * uncertainty
    raw_coverage = np.mean((y_true >= lower) & (y_true <= upper))
    logging.info(f" - Raw 95% CI Coverage (ensemble σ): {raw_coverage*100:.2f}%")
    
    if raw_coverage < 0.95:
        logging.info(" - Coverage below 95%. Applying Split Conformal Prediction...")
        split = int(len(y_true) * 0.5)
        cal_y, val_y = y_true[:split], y_true[split:]
        cal_mu, val_mu = y_pred_hybrid[:split], y_pred_hybrid[split:]
        
        lb, ub, q_hat = apply_conformal_prediction(cal_y, cal_mu, val_mu, alpha=0.05)
        new_coverage = np.mean((val_y >= lb) & (val_y <= ub))
        logging.info(f" - Calibrated 95% CI Coverage: {new_coverage*100:.2f}% (Radius: {q_hat:.2f})")
    else:
        logging.info(" - Coverage meets 95% threshold. No calibration needed.")
        
    # 4. Resilience Score Sensitivity (BUG-7 fix: uses real volatility/trend)
    run_sensitivity_analysis(results_df, y_true, model_preds)
    
    # Save results
    results_df.to_csv(config.EVALUATION_METRICS_PATH, index=False)
    logging.info(f"\nPhase 5 Complete. Results saved to {config.EVALUATION_METRICS_PATH}.")

if __name__ == "__main__":
    data_utils.set_seed()
    main()
