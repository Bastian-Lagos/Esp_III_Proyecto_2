import numpy as np, json
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score
import matplotlib.pyplot as plt
import joblib

data = np.load("data/embeddings.npz", allow_pickle=True)
X,y = data['X'], data['y']

clf = joblib.load("models/model.joblib")
scaler = joblib.load("models/scaler.joblib")

Xs = scaler.transform(X)
probs = clf.predict_proba(Xs)[:,1]

precision, recall, thresholds = precision_recall_curve(y, probs)
f1s = 2*precision*recall/(precision+recall+1e-9)
best_idx = f1s.argmax()
best_tau = thresholds[best_idx]

plt.plot(recall, precision)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR curve, best τ={best_tau:.2f}")
plt.savefig("reports/pr_curve.png")

roc_auc = auc(*roc_curve(y, probs)[:2])
with open("reports/eval.json", "w") as f:
    json.dump({"best_tau":float(best_tau), "roc_auc":float(roc_auc)}, f)
print("Best τ", best_tau, "ROC AUC", roc_auc)
