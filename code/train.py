import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import fisher_exact

# =========================
# 配置
# =========================
TRAIN_PATH = r"D:\work\论文\Experiment\train\papers_train.csv"
TEST_PATH  = r"D:\work\论文\Experiment\train\_papers_test.csv"

DISC_COL = "discriminant"
LABEL_COL = "award"

N_SPLITS = 5
TOP_K = 0.20

RANDOM_STATE = 42

# Logistic 兼容写法：用弱正则近似无正则
LOGIT_KW = dict(penalty="l2", C=1e6, solver="lbfgs", max_iter=2000)

# =========================
# 工具函数
# =========================
def evaluate_threshold(df, t):
    low = df[df[DISC_COL] <= t]
    high = df[df[DISC_COL] > t]
    p_low = low[LABEL_COL].mean() if len(low) else 0.0
    p_high = high[LABEL_COL].mean() if len(high) else 0.0
    rr = p_low / max(p_high, 1e-12)
    rd = p_low - p_high
    return rr, rd, p_low, p_high

def lift_at_k(df, k):
    df_sorted = df.sort_values(DISC_COL)
    top = df_sorted.head(max(int(len(df) * k), 1))
    base = df[LABEL_COL].mean()
    return (top[LABEL_COL].mean() / max(base, 1e-12))

def recall_at_k(df, k):
    df_sorted = df.sort_values(DISC_COL)
    top = df_sorted.head(max(int(len(df) * k), 1))
    total_pos = df[LABEL_COL].sum()
    return (top[LABEL_COL].sum() / max(total_pos, 1e-12))

def get_threshold_from_model(name, model, df_tr):
    # 返回阈值 t
    if name == "stump":
        # sklearn 的树：root split 在 threshold[0]
        return float(model.tree_.threshold[0])
    elif name == "logistic":
        a = float(model.coef_[0][0])
        b = float(model.intercept_[0])
        # 防止 a 过小
        if abs(a) < 1e-12:
            return float(df_tr[DISC_COL].median())
        return -b / a
    else:
        raise ValueError("unknown model name")

# =========================
# 1. 读取训练集 / 测试集
# =========================
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 基本检查
for col in [DISC_COL, LABEL_COL]:
    if col not in train_df.columns or col not in test_df.columns:
        raise ValueError(f"缺少列 {col}，请检查列名。")

# 只保留需要的列并转数值
train_df = train_df.copy()
test_df = test_df.copy()

train_df[DISC_COL] = pd.to_numeric(train_df[DISC_COL], errors="coerce")
test_df[DISC_COL] = pd.to_numeric(test_df[DISC_COL], errors="coerce")
train_df[LABEL_COL] = pd.to_numeric(train_df[LABEL_COL], errors="coerce").astype(int)
test_df[LABEL_COL] = pd.to_numeric(test_df[LABEL_COL], errors="coerce").astype(int)

train_df = train_df.dropna(subset=[DISC_COL, LABEL_COL])
test_df = test_df.dropna(subset=[DISC_COL, LABEL_COL])

# =========================
# 2. 定义候选方法
# =========================
methods = {
    "stump": lambda: DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
    "logistic": lambda: LogisticRegression(**LOGIT_KW),
    "quantile": None,  # baseline
}

# =========================
# 3. 训练集内 K 折 CV：评估每种方法的阈值稳定性与阈值效应
# =========================
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

results = {}

for name, model_fn in methods.items():
    ts, rrs, rds, lifts, recalls = [], [], [], [], []

    for tr_idx, va_idx in kf.split(train_df):
        df_tr = train_df.iloc[tr_idx]
        df_va = train_df.iloc[va_idx]

        # --- 学阈值（只用 df_tr）---
        if name == "quantile":
            t = float(df_tr[DISC_COL].quantile(0.30))  # 你也可以改 0.2/0.4
        else:
            model = model_fn()
            model.fit(df_tr[[DISC_COL]], df_tr[LABEL_COL])
            t = get_threshold_from_model(name, model, df_tr)

        # --- 在 df_va 验证阈值效应 ---
        rr, rd, _, _ = evaluate_threshold(df_va, t)
        lift = lift_at_k(df_va, TOP_K)
        rec = recall_at_k(df_va, TOP_K)

        ts.append(t)
        rrs.append(rr)
        rds.append(rd)
        lifts.append(lift)
        recalls.append(rec)

    results[name] = {
        "t_mean": float(np.mean(ts)),
        "t_std": float(np.std(ts)),
        "RR_mean": float(np.mean(rrs)),
        "RD_mean": float(np.mean(rds)),
        f"Lift@{int(TOP_K*100)}%": float(np.mean(lifts)),
        f"Recall@{int(TOP_K*100)}%": float(np.mean(recalls)),
    }

print("\n===== Training-set CV Results =====")
for k, v in results.items():
    print(k, v)

# =========================
# 4. 选最优方法（优先 RR，其次阈值稳定性）
# =========================
# 只在 supervised 方法中选（排除 quantile）
supervised_methods = {
    k: v for k, v in results.items() if k != "quantile"
}

best_method = max(
    supervised_methods.items(),
    key=lambda x: (x[1]["RD_mean"], -x[1]["t_std"])
)[0]


print("\nSelected method:", best_method)

# =========================
# 5. 用全训练集重训并得到最终阈值
# =========================
if best_method == "quantile":
    final_t = float(train_df[DISC_COL].quantile(0.30))
elif best_method == "stump":
    m = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)
    m.fit(train_df[[DISC_COL]], train_df[LABEL_COL])
    final_t = get_threshold_from_model("stump", m, train_df)
else:
    m = LogisticRegression(**LOGIT_KW)
    m.fit(train_df[[DISC_COL]], train_df[LABEL_COL])
    final_t = get_threshold_from_model("logistic", m, train_df)

print("\nFinal threshold Δ* =", final_t)

# =========================
# 6. 在测试集上做一次性验证（风险分层 + 显著性 + Lift/Recall）
# =========================
rr, rd, p_low, p_high = evaluate_threshold(test_df, final_t)

# 2x2 Fisher
a = int(((test_df[DISC_COL] <= final_t) & (test_df[LABEL_COL] == 1)).sum())
b = int(((test_df[DISC_COL] <= final_t) & (test_df[LABEL_COL] == 0)).sum())
c = int(((test_df[DISC_COL] > final_t) & (test_df[LABEL_COL] == 1)).sum())
d = int(((test_df[DISC_COL] > final_t) & (test_df[LABEL_COL] == 0)).sum())

odds, pval = fisher_exact([[a, b], [c, d]])

lift = lift_at_k(test_df, TOP_K)
rec = recall_at_k(test_df, TOP_K)

print("\n===== Test-set Validation =====")
print("Award rate (Δ ≤ t):", p_low)
print("Award rate (Δ > t):", p_high)
print("Risk Difference (RD):", rd)
print("Risk Ratio (RR):", rr)
print("Contingency [[a,b],[c,d]] =", [[a, b], [c, d]])
print("Odds Ratio:", odds)
print("Fisher p-value:", pval)
print(f"Lift@{int(TOP_K*100)}%:", lift)
print(f"Recall@{int(TOP_K*100)}%:", rec)

# =========================
# 7. Δ 阈值分类指标输出 (Accuracy / Precision / Recall / F1)
# =========================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Δ ≤ final_t 预测为获奖（1），否则为非获奖（0）
test_df['pred'] = (test_df[DISC_COL] <= final_t).astype(int)

accuracy = accuracy_score(test_df[LABEL_COL], test_df['pred'])
precision = precision_score(test_df[LABEL_COL], test_df['pred'], zero_division=0)
recall_val = recall_score(test_df[LABEL_COL], test_df['pred'])
f1 = f1_score(test_df[LABEL_COL], test_df['pred'])

print("\n===== Δ Threshold Classification Metrics =====")
print(f"Threshold Δ* = {final_t}")
print(f"Accuracy  = {accuracy:.4f}")
print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall_val:.4f}")
print(f"F1-score  = {f1:.4f}")
