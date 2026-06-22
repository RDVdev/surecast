from sklearn.linear_model import BayesianRidge
import numpy as np

model = BayesianRidge()
X_train = np.random.rand(100, 10)
y_train = np.random.rand(100)
X_val = np.random.rand(20, 10)

model.fit(X_train, y_train)
pred = model.predict(X_val)
print("Predicted")

preds = {'a': pred, 'b': pred}
ensemble_pred = np.mean(list(preds.values()), axis=0)
print("Averaged")
