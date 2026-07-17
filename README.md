# SUREcast: Supply Chain Forecasting Pipeline

## Original Implementation Overview

SUREcast is an end-to-end machine learning and deep learning pipeline designed to forecast supply chain dynamics (such as sales and demand) while aligning model outputs with continuous human-like preference data. 

The original pipeline was structured into seven distinct phases:
1. **Phase 1 (Data Profiling)**: Generates automated statistical summaries and quality reports from raw supply chain data.
2. **Phase 2 (Sequence Construction)**: Transforms raw temporal data into sequence-based inputs suitable for time-series forecasting.
3. **Phase 3 (Feature Engineering)**: Constructs derived features (e.g., Sales-Quantity ratios, shipping risks) and applies imputation and scaling.
4. **Phase 4 (Model Architecture)**: Defines and trains the core forecasting model.
5. **Phase 5 (Evaluation)**: Computes predictive performance metrics (MSE, MAE, R-squared) and calibration scores.
6. **Phase 6 (Preference Alignment)**: Fine-tunes the forecasting model using a contrastive loss function inspired by Direct Preference Optimization (DPO).
7. **Phase 7 (Reporting Layer)**: Generates a final markdown report summarizing the model's performance and feature attributions.

While the original implementation established this 7-phase structure, it suffered from hardcoded configurations, architectural limitations, data leakage issues, and platform-specific bugs.

---

## Major Updates & Fixes (Changelog)

We recently performed a complete synchronization with a robust, cross-platform working implementation. Below are the major architectural, algorithmic, and bug fixes applied to the repository:

### 1. Centralized Configuration & Shared Utilities
- **Added `config.py`**: Centralized all hyperparameters, window lengths, and file paths. Replaced hardcoded values (like window length choices `[4, 8, 12, 26]`) with dynamic config lookups.
- **Added `data_utils.py`**: Removed hundreds of lines of duplicated data loading and sequence building logic across `predict.py`, `robustness_audit.py`, and `phase5_evaluation.py`.

### 2. Core Architectural Upgrades
- **Advanced Model Integration (Phase 4)**: Completely overhauled the deep learning architecture. Replaced the basic model with the `MultiBranchSequenceModel`, introducing proper channel attention mechanisms and comprehensive ablation study capabilities.

### 3. Algorithmic and Data Integrity Fixes
- **Data Leakage Fix (Phase 2)**: Stopped normalizing `days_since_epoch` across the entire dataset in Phase 2 (which caused look-ahead bias). Normalization is now correctly deferred to the `StandardScaler` in Phase 3.
- **Ratio Clipping (Phase 3)**: Added `np.clip(..., -1000.0, 1000.0)` to all division-based feature engineering (e.g., `Sales_Quantity_Ratio`) to prevent extreme infinite outliers caused by division by near-zero denominators.
- **Dataframe Reconstruction (BUG-8 / Phase 3)**: Fixed a double-transformation bug by correctly reconstructing the full dataset via `pd.concat([train_df, val_df])` after splitting.
- **Resilience Scoring (BUG-7 / Phase 5)**: Implemented the actual underlying math for the Resilience Score (`compute_volatility_ratio` and `compute_trend_similarity`), replacing dummy implementations.
- **Synthetic Preference Generation (Phase 6)**: Added the `generate_synthetic_preference_pairs()` logic, allowing the continuous DPO pipeline to train end-to-end without requiring manual expert annotations.

### 4. Dynamic Reporting 
- **Faithful Reporter (Phase 7)**: Scrapped the outdated hardcoded template system. Introduced the `ConstrainedReporter` which dynamically parses evaluation metrics and predictions directly from the Phase 4/5 CSV outputs, ensuring 100% faithfulness in the generated markdown reports.

### 5. Security & Cross-Platform Compatibility
- **Windows Execution (BUG-10 / `app.py`)**: Replaced hardcoded `python3` subprocess commands with `sys.executable`, making the Flask backend natively compatible with Windows environments.
- **Secure Model Loading (BUG-14 / `predict.py`)**: Added `weights_only=True` to all `torch.load()` calls to prevent arbitrary code execution vulnerabilities.
- **Device Support (ARCH-9)**: Added explicit `.to(device)` mapping for PyTorch tensors across `predict.py` and `robustness_audit.py` to seamlessly support both CPU and GPU execution.
- **Unified Error Handling (BUG-4 / `run_pipeline.py`)**: Upgraded the pipeline orchestrator to safely catch errors in any phase, halt execution, and cleanly report execution times.
- **CORS Support**: Added explicit `@cross_origin()` decorators in `app.py` to ensure the React frontend can reliably communicate with the Flask API.
