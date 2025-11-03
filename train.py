import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib, json

data = np.load("data/embeddings.npz", allow_pickle=True)
X, y = data['X'], data['y']

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

Xtr, Xval, ytr, yval = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=42)

clf = LogisticRegression(max_iter=500)
clf.fit(Xtr, ytr)

p = clf.predict_proba(Xval)[:,1]
preds = (p >= 0.75).astype(int)

metrics = {
    "accuracy": float(accuracy_score(yval, preds)),
    "roc_auc": float(roc_auc_score(yval, p)),
}
with open("reports/metrics.json","w") as f:
    json.dump(metrics, f, indent=2)

joblib.dump(clf, "models/model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
print("Saved model + scaler. Metrics:", metrics)