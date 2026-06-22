import numpy as np
from phase4_model_architecture import train_ml_ensemble

X_train = np.random.rand(100, 10)
y_train = np.random.rand(100)
X_val = np.random.rand(20, 10)
y_val = np.random.rand(20)

try:
    mae, pred, preds = train_ml_ensemble(X_train, y_train, X_val, y_val)
    print("FINISHED", mae)
except Exception as e:
    print("CRASHED", e)
