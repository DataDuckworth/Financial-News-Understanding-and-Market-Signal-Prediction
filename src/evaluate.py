import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

def classification_metrics(y, prob, threshold=0.5):
    pred = (np.asarray(prob) >= threshold).astype(int)
    return {"accuracy": float(accuracy_score(y,pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y,pred)),
            "precision": float(precision_score(y,pred,zero_division=0)),
            "recall": float(recall_score(y,pred,zero_division=0)),
            "f1": float(f1_score(y,pred,zero_division=0)),
            "roc_auc": float(roc_auc_score(y,prob)) if len(set(y)) == 2 else None,
            "confusion_matrix": confusion_matrix(y,pred,labels=[0,1]).tolist()}

def simple_directional_backtest(predictions, column, threshold=0.5):
    p = predictions[column].to_numpy(float)
    r = predictions.next_return.fillna(0).to_numpy(float)
    strat = (p >= threshold).astype(float)*r
    equity = np.cumprod(1+strat)
    return {"observations": int(len(r)),
            "strategy_total_return_no_costs": float(equity[-1]-1) if len(r) else 0.0,
            "strategy_mean_period_return": float(np.mean(strat)) if len(r) else 0.0,
            "buy_and_hold_total_return_same_rows": float(np.prod(1+r)-1) if len(r) else 0.0,
            "warning": "Illustrative only; excludes fees, slippage, spread, latency and capital constraints."}

def save_metrics(metrics, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
