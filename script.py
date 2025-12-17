# ============================================================
# SISSO symbolic regression runner
# ============================================================

from pathlib import Path
import argparse
import yaml
import re
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split, KFold

from src.TorchSisso import SissoModel

torch.set_num_threads(1)


# ------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SISSO symbolic regression")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots (overrides config)"
    )
    return parser.parse_args()


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(np.abs(y_pred - y_true)))


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# ------------------------------------------------------------
# Equation handling
# ------------------------------------------------------------
def _transform_equation(eq: str) -> str:
    expr = eq.strip()
    expr = re.sub(r'pow\(\s*2\s*\)\s*\(\s*([^)]+?)\s*\)', r'(\1)**2', expr)
    expr = re.sub(r'pow\(\s*3\s*\)\s*\(\s*([^)]+?)\s*\)', r'(\1)**3', expr)
    expr = re.sub(r'\^([23])', r'**\1', expr)
    return expr


def predict_from_equation(equation: str, X_df: pd.DataFrame) -> np.ndarray:
    expr = _transform_equation(equation)
    ns = {c: X_df[c].to_numpy() for c in X_df.columns}
    ns.update({'abs': np.abs, 'np': np})
    return np.asarray(eval(expr, {"__builtins__": {}}, ns), dtype=float).ravel()


# ------------------------------------------------------------
# SISSO helper
# ------------------------------------------------------------
def make_sisso_model(df: pd.DataFrame, **kwargs):
    for key in ("dataframe", "df", "data", "dataset"):
        try:
            return SissoModel(**{key: df}, **kwargs)
        except TypeError:
            pass
    return SissoModel(df, **kwargs)


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
def load_dataset(cfg):
    path = Path(cfg["data"]["path"])
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    df = pd.read_csv(path, delim_whitespace=True, comment="#")
    df = df.rename(columns={cfg["data"]["target"]: "Eads"})

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df = df[["Eads"] + [c for c in num_cols if c != "Eads"]]
    df = df.dropna().astype("float32")

    return df


# ------------------------------------------------------------
# Cross-validation
# ------------------------------------------------------------
def cv_rmse_for_nterm(df, n_term, cfg, model_cfg):
    kf = KFold(
        n_splits=cfg["cv"]["folds"],
        shuffle=True,
        random_state=cfg["data"]["seed"]
    )

    ys = df["Eads"].to_numpy()
    Xs = df.drop(columns=["Eads"])
    cols = Xs.columns.tolist()
    Xs = Xs.to_numpy()

    rmses = []

    for tr, te in kf.split(Xs):
        df_tr = pd.DataFrame(
            np.column_stack([ys[tr], Xs[tr]]),
            columns=["Eads"] + cols
        )
        df_te = pd.DataFrame(
            np.column_stack([ys[te], Xs[te]]),
            columns=["Eads"] + cols
        )

        sm = make_sisso_model(
            df_tr,
            operators=model_cfg["operators"],
            n_expansion=model_cfg["n_expansion"],
            n_term=n_term,
            k=model_cfg["k_subspace"],
            use_gpu=model_cfg["use_gpu"]
        )

        _, eq, _, _ = sm.fit()
        y_pred = predict_from_equation(eq, df_te.drop(columns=["Eads"]))
        rmses.append(rmse(df_te["Eads"], y_pred))

        del sm
        gc.collect()

    return np.array(rmses)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.save_plots:
        cfg["output"]["save_plots"] = True

    out_dir = Path(cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg)

    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["seed"],
        shuffle=True
    )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    mode = "fast" if cfg["run_mode"]["fast"] else "full"
    model_cfg = cfg["model_configs"][mode]

    # CV5 на TRAIN для выбора n_term
    cv_results = []
    for n in model_cfg["n_terms"]:
        scores = cv_rmse_for_nterm(df_train, n, cfg, model_cfg)
        cv_results.append((n, scores.mean(), scores.std(), scores))

    cv_results.sort(key=lambda x: x[1])
    best_n_term, cv_mean, cv_std, cv_all = cv_results[0]

    # Full-fit на ВСЕХ данных
    sm_full = make_sisso_model(
        df,
        operators=model_cfg["operators"],
        n_expansion=model_cfg["n_expansion"],
        n_term=best_n_term,
        k=model_cfg["k_subspace"],
        use_gpu=model_cfg["use_gpu"]
    )
    rmse_full_fit, eq_full, r2_full_rep, _ = sm_full.fit()
    y_full_pred = predict_from_equation(
        eq_full, df.drop(columns=["Eads"]))
    overall_rmse = rmse(df["Eads"].to_numpy(), y_full_pred)
    overall_mae = mae(df["Eads"].to_numpy(), y_full_pred)
    overall_r2 = r2_score(df["Eads"].to_numpy(), y_full_pred)

    print("\n=== Full-fit on ALL (AMO) ===")
    print("Equation:", eq_full)
    print(
        f"RMSE={overall_rmse:.5f} | MAE={overall_mae:.5f} | R²={overall_r2:.5f}")

    # Hold-out TEST (20%)
    sm_hold = make_sisso_model(
        df_train,
        operators=model_cfg["operators"],
        n_expansion=model_cfg["n_expansion"],
        n_term=best_n_term,
        k=model_cfg["k_subspace"],
        use_gpu=model_cfg["use_gpu"]
    )
    rmse_tr_hold, eq_hold, r2_tr_hold, _ = sm_hold.fit()

    X_val = df_val.drop(columns=["Eads"])
    y_val_true = df_val["Eads"].to_numpy()
    y_val_pred = predict_from_equation(eq_hold, X_val)

    val_rmse = rmse(y_val_true, y_val_pred)
    val_mae = mae(y_val_true, y_val_pred)
    val_r2 = r2_score(y_val_true, y_val_pred)

    print("\n=== Hold-out (20%) — AMO ===")
    print("Equation (trained on 80%):", eq_hold)
    print(f"Train RMSE (80%)={rmse_tr_hold:.5f}")
    print(
        f"Val  RMSE (20%)={val_rmse:.5f} | MAE={val_mae:.5f} | R²={val_r2:.5f}")

    if cfg["output"]["save_predictions"]:
        holdout_df = pd.DataFrame({
            "Eads_true": y_val_true,
            "Eads_pred": y_val_pred,
        })
        holdout_df["residual"] = holdout_df["Eads_pred"] - \
            holdout_df["Eads_true"]
        holdout_df["abs_error"] = np.abs(holdout_df["residual"])
        holdout_df.to_csv(out_dir / "predictions.csv", index=False)

    if cfg["output"]["save_plots"]:
        labels = ["Overall (train full)",
                  f"CV{cfg["cv"]["folds"]} mean (train)", "Validation (20%)"]
        values = [overall_rmse, cv_mean, val_rmse]
        plt.figure(figsize=(5, 4))
        plt.bar(labels, values)
        plt.ylabel("RMSE")
        plt.title("AMO — RMSE comparison")
        plt.tight_layout()
        plt.savefig("amo_rmse_comparison.png", dpi=150)

        # Parity plot (validation)
        plt.figure(figsize=(5, 5))
        plt.scatter(y_val_true, y_val_pred, s=28)
        mn, mx = float(np.min(y_val_true)), float(np.max(y_val_true))
        plt.plot([mn, mx], [mn, mx], linestyle="--")
        plt.xlabel("True Eads (validation)")
        plt.ylabel("Predicted Eads (validation)")
        plt.title("AMO — Parity (validation)")
        plt.tight_layout()
        plt.savefig("amo_parity_validation.png", dpi=150)


if __name__ == "__main__":
    main()
