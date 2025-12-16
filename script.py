# ============================================================
# SISSO for /content/dataset_new_amo.dat  (amo + charge)
# CV5 on TRAIN (80%) -> best n_term  |  Hold-out TEST (20%)
# Saves: amo_sisso_report.txt, amo_holdout_predictions.csv,
#        amo_rmse_comparison.png, amo_parity_validation.png
# ============================================================
from src.TorchSisso import SissoModel
import sys, subprocess, importlib, re, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
torch.set_num_threads(1)
from sklearn.model_selection import train_test_split, KFold


# ---------- 1) Load AMO dataset ----------
DATA_PATH = Path("train.dat")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Файл не найден: {DATA_PATH}")

# пробельный разделитель, игнорируем строки с '#'
new_amo_ch = pd.read_csv(DATA_PATH, delim_whitespace=True, comment="#")
print("Загружено:", new_amo_ch.shape)
#display(new_amo_ch.head(3))


# ---------- 2) Target rename + numeric-only ----------
new_amo_ch = new_amo_ch.rename(columns={'Energy_Adsorption': 'Eads'})
if "Eads" not in new_amo_ch.columns:
    raise ValueError("Не найден целевой столбец 'Eads'.")

num_cols = new_amo_ch.select_dtypes(include=["number"]).columns.tolist()
if "Eads" not in num_cols:
    new_amo_ch["Eads"] = pd.to_numeric(new_amo_ch["Eads"], errors="coerce")
    num_cols = new_amo_ch.select_dtypes(include=["number"]).columns.tolist()
    if "Eads" not in num_cols:
        raise ValueError("'Eads' не числовая и не приводится к числу.")

feature_cols = [c for c in num_cols if c != "Eads"]
df_amo_num = new_amo_ch[["Eads"] + feature_cols].copy().dropna().astype("float32")
print(f"Численных фич: {len(feature_cols)} | строк: {len(df_amo_num)}")
#display(df_amo_num.head(3))


# ---------- 3) Hyperparams ----------
FAST = False            # первый стабильный прогон; потом можно False
SEED = 42
FOLDS = 5
FOLD_TIMEOUT_SEC = 300

safe_ops = ['+','-','*','/','pow(2)','abs']   # устойчивые операторы

if FAST:
    operators   = safe_ops
    N_EXPANSION = 2         # Tier=2 (быстрее)
    K_SUBSPACE  = 8
    N_TERMS_GRID = (1, 2)   # перебор размерности дескриптора
else:
    operators   = safe_ops  # можно добавить 'sqrt','ln','exp' позже
    N_EXPANSION = 3
    K_SUBSPACE  = 10
    N_TERMS_GRID = (1, 2, 3)

USE_GPU = False


# ---------- 4) Metrics & utilities ----------
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
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot

# парсинг строки уравнения в выражение (в этой версии нет sm.predict)
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
    y = eval(expr, {"__builtins__": {}}, ns)
    return np.asarray(y, dtype=float).ravel()

# подстройщик конструктора для разных версий пакета
def make_sisso_model(df: pd.DataFrame, **kwargs):
    for key in ("dataframe","df","data","dataset"):
        try:
            return SissoModel(**{key: df}, **kwargs)
        except TypeError:
            pass
    return SissoModel(df, **kwargs)


# ---------- 5) Split: CV только на TRAIN (80%), TEST = 20% ----------
train_idx, val_idx = train_test_split(
    np.arange(len(df_amo_num)), test_size=0.20, random_state=SEED, shuffle=True
)
df_train = df_amo_num.iloc[train_idx].reset_index(drop=True)
df_val   = df_amo_num.iloc[val_idx].reset_index(drop=True)

class TimeoutException(Exception): pass

def cv_rmse_for_nterm_on_train(df_tr_all: pd.DataFrame, n_term: int, folds: int = 5, seed: int = 42):
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    ys = df_tr_all["Eads"].to_numpy()
    Xs = df_tr_all.drop(columns=["Eads"])
    cols = Xs.columns.tolist()
    Xs = Xs.to_numpy()
    rmses = []
    for i, (tr, te) in enumerate(kf.split(Xs), 1):
        df_tr = pd.DataFrame(np.column_stack([ys[tr], Xs[tr]]), columns=["Eads"] + cols)
        df_te = pd.DataFrame(np.column_stack([ys[te],  Xs[te]]), columns=["Eads"] + cols)

        try:
            sm = make_sisso_model(
                df_tr,
                operators=operators,
                n_expansion=N_EXPANSION,
                n_term=n_term,
                k=K_SUBSPACE,
                use_gpu=USE_GPU
            )
            rmse_tr, eq, r2_tr, _ = sm.fit()
            y_pred = predict_from_equation(eq, df_te.drop(columns=["Eads"]))
            rmses.append(rmse(df_te["Eads"].to_numpy(), y_pred))
        except TimeoutException:
            print(f"[CV{folds} n_term={n_term}] Fold {i} — TIMEOUT ({FOLD_TIMEOUT_SEC}s)")
            rmses.append(np.nan)
        finally:
            del sm; gc.collect()
    rmses = np.array(rmses, dtype=float)
    rmses = rmses[~np.isnan(rmses)]
    if len(rmses) == 0: return np.array([np.inf])
    return rmses

def main():
    # --- CV5 на TRAIN для выбора n_term ---
    cv_summary = []
    for n_term in N_TERMS_GRID:
        scores = cv_rmse_for_nterm_on_train(df_train, n_term, folds=FOLDS, seed=SEED)
        cv_summary.append((n_term, scores.mean(), scores.std(), scores))
        print(f"CV{FOLDS} (TRAIN) n_term={n_term}: RMSE {scores.mean():.5f} ± {scores.std():.5f} | folds={np.round(scores,5)}")

    cv_summary.sort(key=lambda x: x[1])
    best_n_term, cv_mean, cv_std, cv_all = cv_summary[0]
    print(f"\nЛучший n_term по CV{FOLDS} на TRAIN: {best_n_term} (mean RMSE={cv_mean:.5f} ± {cv_std:.5f})")


    # ---------- 6) Full-fit на ВСЕХ данных ----------
    sm_full = make_sisso_model(
        df_amo_num,
        operators=operators,
        n_expansion=N_EXPANSION,
        n_term=best_n_term,
        k=K_SUBSPACE,
        use_gpu=USE_GPU
    )
    rmse_full_fit, eq_full, r2_full_rep, _ = sm_full.fit()
    y_full_pred = predict_from_equation(eq_full, df_amo_num.drop(columns=["Eads"]))
    overall_rmse = rmse(df_amo_num["Eads"].to_numpy(), y_full_pred)
    overall_mae  = mae(df_amo_num["Eads"].to_numpy(), y_full_pred)
    overall_r2   = r2_score(df_amo_num["Eads"].to_numpy(), y_full_pred)

    print("\n=== Full-fit on ALL (AMO) ===")
    print("Equation:", eq_full)
    print(f"RMSE={overall_rmse:.5f} | MAE={overall_mae:.5f} | R²={overall_r2:.5f}")


    # ---------- 7) Hold-out TEST (20%) ----------
    sm_hold = make_sisso_model(
        df_train,
        operators=operators,
        n_expansion=N_EXPANSION,
        n_term=best_n_term,
        k=K_SUBSPACE,
        use_gpu=USE_GPU
    )
    rmse_tr_hold, eq_hold, r2_tr_hold, _ = sm_hold.fit()

    X_val = df_val.drop(columns=["Eads"])
    y_val_true = df_val["Eads"].to_numpy()
    y_val_pred = predict_from_equation(eq_hold, X_val)

    val_rmse = rmse(y_val_true, y_val_pred)
    val_mae  = mae(y_val_true, y_val_pred)
    val_r2   = r2_score(y_val_true, y_val_pred)

    print("\n=== Hold-out (20%) — AMO ===")
    print("Equation (trained on 80%):", eq_hold)
    print(f"Train RMSE (80%)={rmse_tr_hold:.5f}")
    print(f"Val  RMSE (20%)={val_rmse:.5f} | MAE={val_mae:.5f} | R²={val_r2:.5f}")


    # # ---------- 8) Save predictions & plots ----------
    # # CSV с тестовыми предсказаниями
    # holdout_df = pd.DataFrame({
    #     "Eads_true": y_val_true,
    #     "Eads_pred": y_val_pred,
    # })
    # holdout_df["residual"]  = holdout_df["Eads_pred"] - holdout_df["Eads_true"]
    # holdout_df["abs_error"] = np.abs(holdout_df["residual"])
    # holdout_df.to_csv("amo_holdout_predictions.csv", index=False)
    # print("Saved: amo_holdout_predictions.csv")

    # # RMSE bar
    # labels = ["Overall (train full)", f"CV{FOLDS} mean (train)", "Validation (20%)"]
    # values = [overall_rmse, cv_mean, val_rmse]
    # plt.figure(figsize=(5,4))
    # plt.bar(labels, values)
    # plt.ylabel("RMSE")
    # plt.title("AMO — RMSE comparison")
    # plt.tight_layout()
    # plt.savefig("amo_rmse_comparison.png", dpi=150)
    # plt.show()

    # # Parity plot (validation)
    # plt.figure(figsize=(5,5))
    # plt.scatter(y_val_true, y_val_pred, s=28)
    # mn, mx = float(np.min(y_val_true)), float(np.max(y_val_true))
    # plt.plot([mn, mx], [mn, mx], linestyle="--")
    # plt.xlabel("True Eads (validation)")
    # plt.ylabel("Predicted Eads (validation)")
    # plt.title("AMO — Parity (validation)")
    # plt.tight_layout()
    # plt.savefig("amo_parity_validation.png", dpi=150)
    # plt.show()

if __name__ == "__main__":
    main()
