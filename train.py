from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import build_dataset

HERE = Path(__file__).parent
FIG = HERE / "figures"
MODELS = HERE / "models"


def main():
    FIG.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    X, y, names, meta = build_dataset(n_emp=400, neg_per_pos=3, seed=11)
    X, y = np.array(X, float), np.array(y, int)
    print(f"Пар: {len(y)} | positive: {y.sum()} ({y.mean():.1%}) | фич: {X.shape[1]}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              stratify=y, random_state=42)

    models = {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000, C=1.0)),
        "gboost": GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                             learning_rate=0.1, random_state=42),
    }

    results = {}
    proba = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        proba[name] = p
        pred = (p >= 0.5).astype(int)
        results[name] = {
            "roc_auc": roc_auc_score(y_te, p),
            "pr_auc": average_precision_score(y_te, p),
            "precision@0.5": precision_score(y_te, pred),
            "recall@0.5": recall_score(y_te, pred),
            "f1@0.5": f1_score(y_te, pred),
        }

    print("\nСравнение моделей (test):")
    for name, m in results.items():
        print(f"{name:8s} ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
              f"P={m['precision@0.5']:.3f} R={m['recall@0.5']:.3f} F1={m['f1@0.5']:.3f}")

    best = max(results, key=lambda k: results[k]["pr_auc"])
    print(f"\nЛучшая по PR-AUC: {best}")

    # калибруем лучшую модель (isotonic)
    base = models[best]
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
    calibrated.fit(X_tr, y_tr)
    p_cal = calibrated.predict_proba(X_te)[:, 1]

    # порог под высокую точность (precision >= 0.99)
    prec, rec, thr = precision_recall_curve(y_te, p_cal)
    ok = np.where(prec[:-1] >= 0.99)[0]
    chosen_thr = float(thr[ok[0]]) if len(ok) else 0.5
    pred_thr = (p_cal >= chosen_thr).astype(int)
    print(f"Порог под precision>=0.99: thr={chosen_thr:.3f} "
          f"-> P={precision_score(y_te, pred_thr):.3f} "
          f"R={recall_score(y_te, pred_thr):.3f}")

    # порог под максимум F1
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    f1_thr = float(thr[int(np.argmax(f1s))])
    pred_f1 = (p_cal >= f1_thr).astype(int)
    print(f"Порог под max F1:           thr={f1_thr:.3f} "
          f"-> P={precision_score(y_te, pred_f1):.3f} "
          f"R={recall_score(y_te, pred_f1):.3f} F1={f1_score(y_te, pred_f1):.3f}")

    # графики
    # ROC
    plt.figure(figsize=(5, 4))
    for name, p in proba.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.7)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG / "roc.png", dpi=120); plt.close()

    # PR
    plt.figure(figsize=(5, 4))
    for name, p in proba.items():
        pr, rc, _ = precision_recall_curve(y_te, p)
        plt.plot(rc, pr, label=f"{name} (AP={results[name]['pr_auc']:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall")
    plt.legend(); plt.tight_layout(); plt.savefig(FIG / "pr.png", dpi=120); plt.close()

    # Калибровка
    plt.figure(figsize=(5, 4))
    for label, p in [("uncalibrated", proba[best]), ("isotonic", p_cal)]:
        fr_pos, mean_pred = calibration_curve(y_te, p, n_bins=10)
        plt.plot(mean_pred, fr_pos, "o-", label=label)
    plt.plot([0, 1], [0, 1], "k--", lw=0.7)
    plt.xlabel("Предсказанная вероятность"); plt.ylabel("Доля положительных")
    plt.title(f"Калибровка ({best})"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG / "calibration.png", dpi=120); plt.close()

    # Feature importance
    perm = permutation_importance(models[best], X_te, y_te, n_repeats=20,
                                  random_state=42, scoring="average_precision")
    order = np.argsort(perm.importances_mean)
    plt.figure(figsize=(6, 4))
    plt.barh([names[i] for i in order], perm.importances_mean[order],
             xerr=perm.importances_std[order])
    plt.xlabel("Падение PR-AUC при перемешивании фичи")
    plt.title(f"Permutation importance ({best})")
    plt.tight_layout(); plt.savefig(FIG / "feature_importance.png", dpi=120); plt.close()

    # сохраняем модель и метаданные
    joblib.dump(calibrated, MODELS / "matcher_model.joblib")
    (MODELS / "model_card.json").write_text(json.dumps({
        "best_model": best, "calibration": "isotonic",
        "threshold_precision_099": chosen_thr,
        "threshold_max_f1": f1_thr,
        "feature_names": names,
        "metrics_test": results,
        "permutation_importance": {names[i]: float(perm.importances_mean[i])
                                   for i in range(len(names))},
        "dataset": meta,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nМодель: {MODELS/'matcher_model.joblib'}")
    print(f"Графики: {FIG}/*.png")
    return results


if __name__ == "__main__":
    main()
