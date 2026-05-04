# =========================================================
# 0. 全局设置
# =========================================================

import os
import re
import traceback
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline as ImbPipeline

import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams

try:
    from pygam import LinearGAM, s
except Exception:
    LinearGAM = None


try:
    from tabicl import TabICLClassifier
except Exception:
    TabICLClassifier = None


# =========================================================
# 1. 参数区
# =========================================================

RANDOM_STATE = 42

FILE_PATH = r"NEV-ResearchData-Total-R1.xlsx"
OUTPUT_DIR = "R1"
OUTPUT_BASE_SUBDIR = "NEV"

ENTITY_COL_REF = "股票代码"
YEAR_COL_REF = "年份"

# =========================================================
# 多目标循环设置
# 每个任务会分别输出到：R1/NEV/Zscore, R1/NEV/Oscore,
# R1/NEV/ViolationRisk, R1/NEV/KMV
# =========================================================
TARGET_TASKS = [
    {
        "task_name": "Zscore",
        "target_col_ref": "FinancialDistress-Zscore-Lead1",
        "output_subdir": os.path.join(OUTPUT_BASE_SUBDIR, "Zscore"),
    },
    {
        "task_name": "Oscore",
        "target_col_ref": "FinancialDistress-Oscore-Lead1",
        "output_subdir": os.path.join(OUTPUT_BASE_SUBDIR, "Oscore"),
    },
    {
        "task_name": "ViolationRisk",
        "target_col_ref": "ViolationRisk",
        "output_subdir": os.path.join(OUTPUT_BASE_SUBDIR, "ViolationRisk"),
    },
    {
        "task_name": "KMV",
        "target_col_ref": "Risk_DD_KMV_LowQ10",
        "output_subdir": os.path.join(OUTPUT_BASE_SUBDIR, "KMV"),
    },
]

# 循环某个目标时，把全部目标列都从特征中剔除，避免标签泄漏
TARGET_COL_REFS_TO_EXCLUDE = [task["target_col_ref"] for task in TARGET_TASKS]

# 普通原始年度特征列
FEATURE_COL_REFS = list(range(33, 55)) 

# 需要做 PCA 的财务特征列
PCA_COL_REFS = list(range(3, 33))

# PCA 保留信息量：0.95 表示保留 95% 累计解释方差
PCA_N_COMPONENTS = 0.95

# 训练 / 测试年份
TRAIN_START_YEAR = 2008
TRAIN_END_YEAR = 2020
TEST_START_YEAR = 2021
TEST_END_YEAR = 2023

SAVE_EXCEL = False

# 5折交叉验证
CV_FOLDS = 5
ENABLE_TUNING = False 

# 调参指标
TUNING_N_ITER = 12
TUNING_SCORING = "average_precision"

# =========================================================
# 主评估指标：风险排序 + Top-k 预警
# =========================================================

MAIN_METRIC_COLS = [
    "ROC-AUC",
    "PR-AUC",
    "KS",
    "Recall@Top10%",
    "Recall@Top20%",
    "Lift@Top10%",
    "Lift@Top20%",
    "NDCG@Top10%",
    "NDCG@Top20%",
    "Precision@Top10%",
    "Precision@Top20%",
]

AUX_METRIC_COLS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "TN",
    "FP",
    "FN",
    "TP",
    "Confusion Matrix",
]

# 推荐模型依据
MODEL_SELECTION_METRIC = "OOF_PR_AUC"

# =========================================================
# 阈值选择设置
# =========================================================

THRESHOLD_SELECTION_METHOD = "PrecisionConstraint_MaxRecall"
MIN_PRECISION = 0.65
FBETA_BETA = 0.5

# =========================================================
# 类别不平衡处理方式
# =========================================================

IMBALANCE_MODE = "smote_except_balanced_models"
SAMPLER_TYPE = "smote"

# 这些模型不外接 SMOTE
NO_EXTERNAL_SMOTE_MODELS = [
]

NO_EXTERNAL_SMOTE_MODELS_FOR_BASELINE = []

NO_TUNING_MODELS = []

TUNE_ONLY_BOOSTING_MODELS = False
BOOSTING_TUNING_MODELS = ["XGBoost", "LightGBM", "CatBoost"]

# TabICLv2 本地 ckpt 路径
LOCAL_TABICL_MODEL_PATH = r"tabicl-classifier-v2-20260212.ckpt"
ENABLE_TABICL = (
    TabICLClassifier is not None
    and bool(LOCAL_TABICL_MODEL_PATH.strip())
    and os.path.exists(LOCAL_TABICL_MODEL_PATH)
)

# =========================================================
# 本版只运行 TabICLv2
# =========================================================
if not ENABLE_TABICL:
    raise RuntimeError(
        "TabICLv2 当前不可用。请检查：1）tabicl 是否安装；"
        "2）LOCAL_TABICL_MODEL_PATH 是否正确；3）ckpt 文件是否存在。"
    )

MODEL_NAMES = ["TabICLv2"]

def print_global_settings():
    print("===== 当前实验设置：TabICLv2 主模型 + PCA First + Ranking Metrics + Multi Targets =====")
    print("任务模式：X_t -> y_{t+1} / 风险事件目标")
    print("TRAIN YEARS =", f"{TRAIN_START_YEAR}-{TRAIN_END_YEAR}")
    print("TEST YEARS =", f"{TEST_START_YEAR}-{TEST_END_YEAR}")
    print("TARGET TASKS =", [(t["task_name"], t["target_col_ref"]) for t in TARGET_TASKS])
    print("MODELS =", MODEL_NAMES)
    print("CV =", f"Stratified {CV_FOLDS}-fold CV within training set")
    print("Tuning metric =", TUNING_SCORING)
    print("TUNING_N_ITER =", TUNING_N_ITER)
    print("MODEL_SELECTION_METRIC =", MODEL_SELECTION_METRIC)
    print("主评估指标 =", MAIN_METRIC_COLS)
    print("辅助指标 =", AUX_METRIC_COLS)
    print("PCA_N_COMPONENTS =", PCA_N_COMPONENTS)
    print("Imbalance mode =", IMBALANCE_MODE)
    print("Sampler type =", SAMPLER_TYPE)
    print("ENABLE_TABICL =", ENABLE_TABICL)


# =========================================================
# 2. 工具函数
# =========================================================

def resolve_col_name(df: pd.DataFrame, col_ref):
    if isinstance(col_ref, int):
        return df.columns[col_ref]
    return col_ref


def deduplicate_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def summarize_yearly_positive_ratio(df: pd.DataFrame, year_col: str, target_col: str, tag: str):
    out = (
        df.groupby(year_col)[target_col]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={
            "count": "N",
            "sum": "Positives",
            "mean": "Positive_Ratio",
        })
    )
    out["Source"] = tag
    return out


def compute_class_weights(y):
    y = pd.Series(y)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())

    if pos == 0:
        scale_pos_weight = 1.0
        class_weight_dict = {0: 1.0, 1: 1.0}
    else:
        scale_pos_weight = neg / pos
        class_weight_dict = {0: 1.0, 1: neg / pos}

    return scale_pos_weight, class_weight_dict


def make_sampler(y_train, random_state=42):
    """
    TabICLv2 主模型版本不再外接采样器。
    保留该函数仅为兼容旧流程接口。
    """
    return None


def should_use_external_sampler(model_name):
    """TabICLv2 主模型版本不使用外部 SMOTE/ROS。"""
    return False


def should_tune_model(model_name):
    """TabICLv2 主模型版本不做随机寻参。"""
    return False


def safe_sheet_name(base_name, used_names):
    base_name = str(base_name)
    base_name = re.sub(r"[\[\]\*\?\/\\:]", "_", base_name)
    base_name = base_name[:31]

    if base_name not in used_names:
        used_names.add(base_name)
        return base_name

    i = 1
    while True:
        suffix = f"_{i}"
        candidate = base_name[:31 - len(suffix)] + suffix
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        i += 1


def get_pred_prob(model, X):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)

        if isinstance(prob, np.ndarray):
            if prob.ndim == 2 and prob.shape[1] >= 2:
                classes = getattr(model, "classes_", None)

                if classes is not None and 1 in list(classes):
                    pos_idx = list(classes).index(1)
                    return prob[:, pos_idx]

                return prob[:, 1]

            elif prob.ndim == 1:
                return prob

    return model.predict(X).astype(float)


# =========================================================
# 科研风格绘图 + TabICLv2 SHAP 解释工具
# =========================================================

plt.style.use("default")

rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 13,
    "axes.labelsize": 17,
    "axes.titlesize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 13,
    "lines.linewidth": 2.2,
    "axes.linewidth": 1.8,
    "grid.linewidth": 0.8,
    "lines.markersize": 7,
})

COLOR_PALETTE = {
    "main": "#3b5b92",
    "ci": "#8395b1",
    "positive": "#36a168",
    "negative": "#e05263",
    "zero_line": "#666666",
    "tipping_point": "#f0746e",
    "background": "#f9f9f9",
}

# =========================================================
# 强制使用离散型 SHAP dependence 的特征
# 说明：EF-KMS 与 EF-NE 的取值分布较离散/偏态，
# 不适合强行使用 GAM 平滑曲线。
# =========================================================
FORCE_DISCRETE_DEPENDENCE_FEATURES = {
    "EF-KMS",
    "EF-NE",
}


def safe_file_name(name):
    
    return re.sub(r'[\\/:*?"<>|]+', "_", str(name))


def get_regular_feature_cols(feature_cols):
    return [c for c in feature_cols if str(c).startswith("REG_")]


def get_feature_display_name(feature_name):
    feature_name = str(feature_name)
    if feature_name.startswith("REG_"):
        return feature_name.replace("REG_", "", 1)
    return feature_name


def plot_scientific_style(ax, XX, y_pred, ci, feature_name, tipping_points=None, p_value=None):
    ax.set_facecolor(COLOR_PALETTE["background"])
    fig = plt.gcf()
    fig.patch.set_facecolor("white")

    ax.axhline(
        0,
        color=COLOR_PALETTE["zero_line"],
        linestyle=(0, (5, 2)),
        linewidth=1.2,
        alpha=0.9,
    )

    ax.fill_between(
        XX.flatten(),
        ci[:, 0],
        ci[:, 1],
        color=COLOR_PALETTE["ci"],
        alpha=0.35,
        edgecolor="none",
    )

    ax.plot(XX, y_pred, color=COLOR_PALETTE["main"], linewidth=2.5)

    if tipping_points:
        for x, y in tipping_points:
            ax.axvline(
                x,
                color=COLOR_PALETTE["tipping_point"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
            )
            ax.scatter(
                x,
                y,
                color=COLOR_PALETTE["tipping_point"],
                s=70,
                edgecolor="white",
                linewidth=1,
            )
            ax.annotate(
                f"{x:.2f}",
                xy=(x, y),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor=COLOR_PALETTE["tipping_point"],
                ),
            )

    pos_mask = y_pred > 0
    neg_mask = y_pred <= 0

    if np.any(pos_mask):
        ax.fill_between(
            XX.flatten(),
            0,
            y_pred,
            where=pos_mask,
            color=COLOR_PALETTE["positive"],
            alpha=0.15,
            label="Positive",
        )

    if np.any(neg_mask):
        ax.fill_between(
            XX.flatten(),
            0,
            y_pred,
            where=neg_mask,
            color=COLOR_PALETTE["negative"],
            alpha=0.15,
            label="Negative",
        )

    ax.set_xlabel(feature_name, fontweight="bold")
    ax.set_ylabel("SHAP value", fontweight="bold")

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    if p_value is not None and np.isfinite(p_value):
        p_text = "p < 0.01" if p_value < 0.01 else f"p = {p_value:.3e}"
        ax.text(
            0.95,
            0.95,
            p_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3),
        )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", frameon=True, framealpha=0.95)



def plot_discrete_shap_dependence(x_feature, y_shap, feature_display_name, save_path):
    """
    离散型 SHAP dependence 图。
    """
    x = np.asarray(x_feature).flatten()
    y = np.asarray(y_shap).flatten()

    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]

    if len(x) == 0:
        print(f"[跳过] {feature_display_name}: 无有效样本，无法绘制离散型依赖图。")
        return

    df_plot = pd.DataFrame({"x": x, "shap": y}).copy()

    mean_df = (
        df_plot
        .groupby("x", as_index=False)
        .agg(
            mean_shap=("shap", "mean"),
            std_shap=("shap", "std"),
            n=("shap", "size"),
        )
        .sort_values("x")
    )
    mean_df["std_shap"] = mean_df["std_shap"].fillna(0.0)
    mean_df["se"] = mean_df["std_shap"] / np.sqrt(mean_df["n"].clip(lower=1))
    mean_df["ci_low"] = mean_df["mean_shap"] - 1.96 * mean_df["se"]
    mean_df["ci_high"] = mean_df["mean_shap"] + 1.96 * mean_df["se"]

    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=600)
    ax.set_facecolor(COLOR_PALETTE["background"])
    fig.patch.set_facecolor("white")

    ax.axhline(
        0,
        color=COLOR_PALETTE["zero_line"],
        linestyle=(0, (5, 2)),
        linewidth=1.2,
        alpha=0.9,
        zorder=1,
    )

    y_min = min(0, float(np.nanmin(y)), float(np.nanmin(mean_df["ci_low"])))
    y_max = max(0, float(np.nanmax(y)), float(np.nanmax(mean_df["ci_high"])))
    if y_max > 0:
        ax.axhspan(
            0,
            y_max,
            color=COLOR_PALETTE["positive"],
            alpha=0.08,
            label="Positive",
            zorder=0,
        )
    if y_min < 0:
        ax.axhspan(
            y_min,
            0,
            color=COLOR_PALETTE["negative"],
            alpha=0.08,
            label="Negative",
            zorder=0,
        )

    rng = np.random.default_rng(RANDOM_STATE)
    unique_values = np.sort(df_plot["x"].unique())
    unique_count = len(unique_values)

    if unique_count <= 1:
        jitter_scale = 0.02
    else:
        x_range = float(np.nanmax(x) - np.nanmin(x))
        min_gap = np.min(np.diff(unique_values)) if unique_count > 1 else 1.0
        jitter_scale = max(0.006 * max(x_range, 1.0), 0.02 * max(min_gap, 1e-6))
        jitter_scale = min(jitter_scale, 0.08 * max(min_gap, 1e-6))

    x_jitter = x + rng.normal(0, jitter_scale, size=len(x))

    ax.scatter(
        x_jitter,
        y,
        s=26,
        alpha=0.45,
        color=COLOR_PALETTE["main"],
        edgecolor="white",
        linewidth=0.25,
        label="Sample SHAP",
        zorder=2,
    )

    if len(mean_df) >= 2:
        ax.fill_between(
            mean_df["x"].values,
            mean_df["ci_low"].values,
            mean_df["ci_high"].values,
            color=COLOR_PALETTE["ci"],
            alpha=0.35,
            edgecolor="none",
            label="95% confidence band",
            zorder=1,
        )

    ax.plot(
        mean_df["x"].values,
        mean_df["mean_shap"].values,
        color=COLOR_PALETTE["main"],
        linewidth=2.5,
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgecolor=COLOR_PALETTE["main"],
        markeredgewidth=1.2,
        label="Mean SHAP",
        zorder=3,
    )

    for xv in unique_values:
        ax.axvline(
            xv,
            color=COLOR_PALETTE["tipping_point"],
            linestyle="--",
            linewidth=0.8,
            alpha=0.25,
            zorder=0,
        )

    mean_y = mean_df["mean_shap"].values
    mean_x = mean_df["x"].values
    crossing_idx = np.where(np.diff(np.sign(mean_y)))[0]
    for idx in crossing_idx:
        cand = [idx, idx + 1]
        j = cand[int(np.argmin(np.abs(mean_y[cand])))]
        ax.axvline(
            mean_x[j],
            color=COLOR_PALETTE["tipping_point"],
            linestyle="--",
            linewidth=1.3,
            alpha=0.75,
            zorder=1,
        )
        ax.annotate(
            f"{mean_x[j]:.2f}",
            xy=(mean_x[j], mean_y[j]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.85,
                edgecolor=COLOR_PALETTE["tipping_point"],
            ),
            zorder=4,
        )

    if unique_count <= 12:
        ax.set_xticks(unique_values)
        ax.set_xticklabels([f"{v:g}" for v in unique_values])

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if x_min == x_max:
        pad = 0.2 if abs(x_min) < 1 else abs(x_min) * 0.05
    else:
        pad = 0.08 * (x_max - x_min)
    ax.set_xlim(x_min - pad, x_max + pad)

    y_pad = 0.08 * (y_max - y_min) if y_max > y_min else 0.01
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_xlabel(feature_display_name, fontweight="bold")
    ax.set_ylabel("SHAP value", fontweight="bold")

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="best", frameon=True, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(save_path, dpi=600, format="jpg", bbox_inches="tight")
    plt.close(fig)


def save_combined_shap_summary_plot(
    shap_values_df,
    X_explain,
    output_path,
    dataset_label=None,
    panel_label=None,
    max_display=20,
):
    """
    保存 SHAP 复合图
    """

    if shap_values_df.empty or X_explain.empty:
        print("[警告] SHAP 或 X_explain 为空，跳过复合 SHAP 图。")
        return

    common_cols = [c for c in X_explain.columns if c in shap_values_df.columns]
    if len(common_cols) == 0:
        print("[警告] SHAP 与 X_explain 没有共同特征列，跳过复合 SHAP 图。")
        return

    shap_values_df = shap_values_df[common_cols].copy()
    X_explain = X_explain[common_cols].copy()

    mean_abs_shap = shap_values_df.abs().mean(axis=0).sort_values(ascending=False)

    if max_display is None:
        display_cols = mean_abs_shap.index.tolist()
    else:
        display_cols = mean_abs_shap.head(min(max_display, len(mean_abs_shap))).index.tolist()

    shap_plot_df = shap_values_df[display_cols].copy()
    X_plot_df = X_explain[display_cols].copy()
    shap_values_numpy = shap_plot_df.values

    fig, ax1 = plt.subplots(figsize=(11.5, 8.8), dpi=600)

    # 1. 绘制 dot summary plot
    plt.sca(ax1)
    shap.summary_plot(
        shap_values_numpy,
        X_plot_df,
        feature_names=X_plot_df.columns.tolist(),
        plot_type="dot",
        show=False,
        color_bar=True,
        max_display=len(display_cols),
        sort=True,
    )

    ax1 = plt.gca()
    ax1.set_position([0.25, 0.13, 0.58, 0.76])

    ax1.tick_params(axis="x", labelsize=15, colors="black", width=1.2)
    ax1.tick_params(axis="y", labelsize=17, colors="black", width=1.2)

    # 取 dot 图实际生成的 y 轴刻度和标签，保证 bar 与 dot 完全对齐。
    y_ticks = ax1.get_yticks()
    y_labels = [tick.get_text() for tick in ax1.get_yticklabels()]

    mean_abs_display = mean_abs_shap.to_dict()
    bar_values = [mean_abs_display.get(label, np.nan) for label in y_labels]

    if np.any(pd.isna(bar_values)):
        available_values = [mean_abs_display[c] for c in display_cols]
        if len(available_values) == len(y_ticks):
            bar_values = list(reversed(available_values))
        else:
            bar_values = [0.0 for _ in y_ticks]

    # 2. 叠加 mean(|SHAP|) bar plot
    ax2 = ax1.twiny()
    ax2.set_position(ax1.get_position())
    ax2.set_ylim(ax1.get_ylim())

    ax2.barh(
        y_ticks,
        bar_values,
        height=0.72,
        alpha=0.22,
        color="#8ecae6",
        edgecolor="none",
        zorder=0,
    )

    ax1.set_zorder(2)
    ax1.patch.set_alpha(0)
    ax2.set_zorder(1)

    # 保留 ax1 左侧特征名；隐藏 ax2 的 y 轴，避免重复或覆盖。
    ax2.yaxis.set_visible(False)
    ax2.grid(False)

    # 3. 坐标轴与样式
    ax1.set_xlabel("Shapley Value Contribution", fontsize=18, fontweight="bold", labelpad=8)
    ax2.set_xlabel("Mean Shapley Value", fontsize=18, fontweight="bold", labelpad=10)
    ax2.xaxis.set_label_position("top")
    ax2.xaxis.tick_top()

    ax2.tick_params(axis="x", labelsize=15, colors="black", width=1.2)

    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.3)

    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.1)

    # 放大 colorbar 字体和标签。
    for other_ax in fig.axes:
        if other_ax not in [ax1, ax2]:
            other_ax.tick_params(labelsize=14)
            other_ax.set_ylabel("Feature value", fontsize=17, fontweight="bold", labelpad=12)
    
    if dataset_label is not None:
        ax1.text(
            0.98,
            0.04,
            str(dataset_label),
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=18,
            fontweight="bold",
        )

   
    if panel_label is not None:
        fig.text(
            0.035,
            0.965,
            str(panel_label),
            fontsize=24,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.savefig(output_path, dpi=600, format="jpg", bbox_inches="tight")
    plt.close(fig)
    print(f"[已保存] 复合 SHAP 图: {output_path}")


def run_tabicl_shap_xai(
    model,
    X_train,
    X_test,
    feature_cols,
    output_dir,
    task_name,
    max_background=80,
    max_explain=200,
    top_n_display=None,
    top_n_gam=None,
):
    """
    TabICLv2 的可解释机器学习分析。
    """

    os.makedirs(output_dir, exist_ok=True)

    X_train = X_train[feature_cols].copy()
    X_test = X_test[feature_cols].copy()

    regular_feature_cols = get_regular_feature_cols(feature_cols)

    if len(regular_feature_cols) == 0:
        raise ValueError(
            "没有检测到 REG_ 开头的非 PCA 特征列。"
            "请检查 build_pca_once_dataset() 中 regular_out_cols 的命名。"
        )

    if len(X_train) > max_background:
        X_background = X_train.sample(max_background, random_state=RANDOM_STATE)
    else:
        X_background = X_train.copy()

    if len(X_test) > max_explain:
        X_explain = X_test.sample(max_explain, random_state=RANDOM_STATE)
    else:
        X_explain = X_test.copy()

    print("\n===== TabICLv2 SHAP 解释开始 =====")
    print("Background samples =", X_background.shape)
    print("Explain samples =", X_explain.shape)
    print("All model feature count =", len(feature_cols))
    print("Regular non-PC feature count =", len(regular_feature_cols))

    def predict_positive_proba(X_input):
        if isinstance(X_input, np.ndarray):
            X_input = pd.DataFrame(X_input, columns=feature_cols)
        else:
            X_input = pd.DataFrame(X_input, columns=feature_cols)

        return get_pred_prob(model, X_input)

    # =====================================================
    # 1. Model-agnostic SHAP
    # =====================================================
    try:
        masker = shap.maskers.Independent(X_background)

        explainer = shap.Explainer(
            predict_positive_proba,
            masker,
            algorithm="permutation",
            feature_names=feature_cols,
        )

        shap_exp = explainer(
            X_explain,
            max_evals=2 * len(feature_cols) + 1,
            batch_size=32,
        )

        shap_values_numpy = np.asarray(shap_exp.values)

    except Exception as e:
        print("[警告] shap.Explainer 失败，切换到 KernelExplainer。错误：", e)

        explainer = shap.KernelExplainer(
            predict_positive_proba,
            X_background,
        )

        shap_values_numpy = np.asarray(
            explainer.shap_values(
                X_explain,
                nsamples=2 * len(feature_cols) + 1,
            )
        )

    if shap_values_numpy.ndim == 3:
        shap_values_numpy = shap_values_numpy[:, :, -1]

    shap_values_all_df = pd.DataFrame(
        shap_values_numpy,
        columns=feature_cols,
        index=X_explain.index,
    )

    shap_values_regular_df = shap_values_all_df[regular_feature_cols].copy()
    shap_values_regular_df_display = shap_values_regular_df.rename(
        columns={c: get_feature_display_name(c) for c in regular_feature_cols}
    )
    shap_values_regular_df_display.insert(0, "sample_index", X_explain.index)

    shap_values_path = os.path.join(
        output_dir,
        f"{task_name}_TabICLv2_SHAP_values_regular_nonPC.xlsx",
    )
    shap_values_regular_df_display.to_excel(shap_values_path, index=False)

    X_explain_regular = X_explain[regular_feature_cols].copy()
    X_explain_regular_display = X_explain_regular.rename(
        columns={c: get_feature_display_name(c) for c in regular_feature_cols}
    )

    X_explain_path = os.path.join(
        output_dir,
        f"{task_name}_TabICLv2_X_explain_regular_nonPC.xlsx",
    )
    X_explain_regular_display.to_excel(X_explain_path, index=True)

    # =====================================================
    # 2. 输出 SHAP importance
    # =====================================================
    importance_df = pd.DataFrame({
        "Feature": [get_feature_display_name(c) for c in regular_feature_cols],
        "Model_Feature": regular_feature_cols,
        "MeanAbsSHAP": np.abs(shap_values_regular_df.values).mean(axis=0),
        "MeanSHAP": shap_values_regular_df.values.mean(axis=0),
    }).sort_values("MeanAbsSHAP", ascending=False)

    importance_path = os.path.join(
        output_dir,
        f"{task_name}_TabICLv2_SHAP_importance_regular_nonPC.xlsx",
    )
    importance_df.to_excel(importance_path, index=False)

    print("\n===== Top SHAP Features: Regular Non-PC Features Only =====")
    print(importance_df.head(20))

    # =====================================================
    # 3. 复合 SHAP summary 图：dot + bar overlay
    # =====================================================
    if top_n_display is None:
        top_n_display = len(regular_feature_cols)
    else:
        top_n_display = min(top_n_display, len(regular_feature_cols))

    combined_plot_path = os.path.join(
        output_dir,
        f"{task_name}_TabICLv2_SHAP_summary_combined_regular_nonPC.jpg",
    )

    shap_values_for_plot = shap_values_regular_df.rename(
        columns={c: get_feature_display_name(c) for c in regular_feature_cols}
    )

    save_combined_shap_summary_plot(
        shap_values_df=shap_values_for_plot,
        X_explain=X_explain_regular_display,
        output_path=combined_plot_path,
        dataset_label=task_name,
        panel_label=None,
        max_display=top_n_display,
    )

    # =====================================================
    # 4. GAM-smoothed dependence plots
    # =====================================================
    if LinearGAM is None:
        print("[提示] 未安装 pygam，跳过 GAM 依赖图。可运行：pip install pygam")
        return importance_df, shap_values_regular_df_display

    gam_dir = os.path.join(output_dir, "GAM_dependence_regular_nonPC")
    os.makedirs(gam_dir, exist_ok=True)

    if top_n_gam is None:
        top_features_model_names = importance_df["Model_Feature"].tolist()
    else:
        top_features_model_names = importance_df.head(top_n_gam)["Model_Feature"].tolist()

    print(f"[INFO] GAM/离散依赖图计划绘制特征数：{len(top_features_model_names)}")

    for feature_model_name in top_features_model_names:
        try:
            feature_display_name = get_feature_display_name(feature_model_name)

            x_feature = X_explain[feature_model_name].values.reshape(-1, 1)
            y_shap = shap_values_all_df[feature_model_name].values

            valid_mask = np.isfinite(x_feature.flatten()) & np.isfinite(y_shap)
            x_feature = x_feature[valid_mask]
            y_shap = y_shap[valid_mask]

            if len(x_feature) < 20:
                print(f"[跳过] {feature_display_name}: 有效样本少于 20。")
                continue

            unique_values = np.unique(x_feature.flatten())
            n_unique = len(unique_values)


            force_discrete = (
                feature_display_name in FORCE_DISCRETE_DEPENDENCE_FEATURES
                or feature_model_name in FORCE_DISCRETE_DEPENDENCE_FEATURES
            )

            if force_discrete or n_unique < 5:
                reason = "强制离散特征" if force_discrete else f"唯一值数量={n_unique}"
                print(f"[离散图] {feature_display_name}: {reason}，使用离散型 SHAP dependence，不做 GAM 拟合。")
                save_name = safe_file_name(feature_display_name)
                plot_discrete_shap_dependence(
                    x_feature=x_feature,
                    y_shap=y_shap,
                    feature_display_name=feature_display_name,
                    save_path=os.path.join(gam_dir, f"{save_name}_SHAP_discrete.jpg"),
                )
                continue

            n_splines = min(20, max(5, n_unique - 1))

            gam = LinearGAM(
                s(0, n_splines=n_splines, spline_order=3, lam=3)
            )

            try:
                gam.gridsearch(x_feature, y_shap, progress=False)
            except TypeError:
                gam.gridsearch(x_feature, y_shap)

            XX = gam.generate_X_grid(term=0, n=300)
            y_pred = gam.predict(XX)
            ci = gam.prediction_intervals(XX, width=0.95)

            zero_crossings = np.where(np.diff(np.sign(y_pred)))[0]
            tipping_points = [(XX[i][0], y_pred[i]) for i in zero_crossings]

            p_value = None
            try:
                p_value = gam.statistics_["p_values"][0]
            except Exception:
                pass

            fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=600)

            plot_scientific_style(
                ax,
                XX,
                y_pred,
                ci,
                feature_name=feature_display_name,
                tipping_points=tipping_points,
                p_value=p_value,
            )

            try:
                r_squared = gam.statistics_["pseudo_r2"]["explained_deviance"] * 100
                ax.text(
                    0.05,
                    0.95,
                    f"$R^2$ = {r_squared:.1f}%",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=11,
                    bbox=dict(
                        facecolor="white",
                        alpha=0.9,
                        edgecolor="none",
                        pad=3,
                    ),
                )
            except Exception:
                pass

            fig.tight_layout()

            save_name = safe_file_name(feature_display_name)
            fig.savefig(
                os.path.join(gam_dir, f"{save_name}_SHAP_GAM.jpg"),
                dpi=600,
                format="jpg",
                bbox_inches="tight",
            )

            plt.close(fig)

        except Exception as e:
            print(f"[警告] {feature_model_name} 的 GAM-SHAP 图绘制失败：{e}")
            continue

    print("\n===== TabICLv2 SHAP 解释完成：仅输出非 PC 特征 =====")
    print("输出目录：", output_dir)

    return importance_df, shap_values_regular_df_display


# =========================================================
# 3. 评估指标
# =========================================================

def calculate_ks(y_true, y_score):
    df = pd.DataFrame({
        "y_true": np.asarray(y_true),
        "y_score": np.asarray(y_score),
    }).dropna()

    if df.empty or df["y_true"].nunique() < 2:
        return np.nan

    df = df.sort_values("y_score", ascending=False).reset_index(drop=True)

    total_bad = (df["y_true"] == 1).sum()
    total_good = (df["y_true"] == 0).sum()

    if total_bad == 0 or total_good == 0:
        return np.nan

    df["cum_bad_rate"] = (df["y_true"] == 1).cumsum() / total_bad
    df["cum_good_rate"] = (df["y_true"] == 0).cumsum() / total_good
    df["ks"] = np.abs(df["cum_bad_rate"] - df["cum_good_rate"])

    return float(df["ks"].max())


def calculate_ndcg_at_k(y_true, y_score, k_ratio=0.1):
    df = pd.DataFrame({
        "y_true": np.asarray(y_true),
        "y_score": np.asarray(y_score),
    }).dropna()

    if df.empty:
        return np.nan

    n = len(df)
    k = max(1, int(np.ceil(n * k_ratio)))

    df = df.sort_values("y_score", ascending=False).reset_index(drop=True)

    rel = df["y_true"].astype(float).values[:k]
    ranks = np.arange(1, len(rel) + 1)
    dcg = np.sum(rel / np.log2(ranks + 1))

    total_pos = int((df["y_true"] == 1).sum())
    ideal_pos = min(total_pos, k)

    if ideal_pos == 0:
        return np.nan

    ideal_rel = np.ones(ideal_pos)
    ideal_ranks = np.arange(1, ideal_pos + 1)
    idcg = np.sum(ideal_rel / np.log2(ideal_ranks + 1))

    if idcg == 0:
        return np.nan

    return float(dcg / idcg)


def calculate_topk_metrics(y_true, y_score, k_ratio=0.1):
    df = pd.DataFrame({
        "y_true": np.asarray(y_true),
        "y_score": np.asarray(y_score),
    }).dropna()

    if df.empty:
        return {
            "Precision": np.nan,
            "Recall": np.nan,
            "Lift": np.nan,
            "NDCG": np.nan,
            "K": 0,
            "Top_Positive_Count": 0,
            "Total_Positive_Count": 0,
            "Base_Rate": np.nan,
        }

    n = len(df)
    k = max(1, int(np.ceil(n * k_ratio)))

    df = df.sort_values("y_score", ascending=False).reset_index(drop=True)

    total_pos = int((df["y_true"] == 1).sum())
    base_rate = total_pos / n if n > 0 else np.nan

    top_df = df.head(k)
    top_pos = int((top_df["y_true"] == 1).sum())

    precision_at_k = top_pos / k if k > 0 else np.nan
    recall_at_k = top_pos / total_pos if total_pos > 0 else np.nan
    lift_at_k = precision_at_k / base_rate if base_rate and base_rate > 0 else np.nan
    ndcg_at_k = calculate_ndcg_at_k(y_true, y_score, k_ratio=k_ratio)

    return {
        "Precision": float(precision_at_k),
        "Recall": float(recall_at_k) if not pd.isna(recall_at_k) else np.nan,
        "Lift": float(lift_at_k) if not pd.isna(lift_at_k) else np.nan,
        "NDCG": float(ndcg_at_k) if not pd.isna(ndcg_at_k) else np.nan,
        "K": int(k),
        "Top_Positive_Count": int(top_pos),
        "Total_Positive_Count": int(total_pos),
        "Base_Rate": float(base_rate) if not pd.isna(base_rate) else np.nan,
    }


def calc_ranking_metrics(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        ks = calculate_ks(y_true, y_prob)
    else:
        roc_auc = np.nan
        pr_auc = np.nan
        ks = np.nan

    top10 = calculate_topk_metrics(y_true, y_prob, k_ratio=0.10)
    top20 = calculate_topk_metrics(y_true, y_prob, k_ratio=0.20)

    return {
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "KS": ks,

        "Recall@Top10%": top10["Recall"],
        "Recall@Top20%": top20["Recall"],

        "Lift@Top10%": top10["Lift"],
        "Lift@Top20%": top20["Lift"],

        "NDCG@Top10%": top10["NDCG"],
        "NDCG@Top20%": top20["NDCG"],

        "Precision@Top10%": top10["Precision"],
        "Precision@Top20%": top20["Precision"],

        "K@Top10%": top10["K"],
        "K@Top20%": top20["K"],

        "Top_Positive_Count@Top10%": top10["Top_Positive_Count"],
        "Top_Positive_Count@Top20%": top20["Top_Positive_Count"],

        "Total_Positive_Count": top10["Total_Positive_Count"],
        "Base_Rate": top10["Base_Rate"],
    }


def calc_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    ranking_metrics = calc_ranking_metrics(y_true, y_prob)

    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        aux_metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "Confusion Matrix": cm.tolist(),
        }

    except Exception:
        aux_metrics = {
            "Accuracy": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "TN": np.nan,
            "FP": np.nan,
            "FN": np.nan,
            "TP": np.nan,
            "Confusion Matrix": None,
        }

    out = {}
    out.update(ranking_metrics)
    out.update(aux_metrics)
    return out


def print_main_metrics(metrics_dict, model_name=None, dataset_name=None):
    display_items = {}

    for col in MAIN_METRIC_COLS:
        if col in metrics_dict:
            display_items[col] = metrics_dict[col]

    for col in ["K@Top10%", "K@Top20%", "Top_Positive_Count@Top10%", "Top_Positive_Count@Top20%", "Total_Positive_Count", "Base_Rate"]:
        if col in metrics_dict:
            display_items[col] = metrics_dict[col]

    for col in ["Accuracy", "Precision", "Recall", "F1", "TP", "FP", "FN", "TN"]:
        if col in metrics_dict:
            display_items[f"Aux_{col}"] = metrics_dict[col]

    if "Threshold" in metrics_dict:
        display_items["Aux_Threshold"] = metrics_dict["Threshold"]

    title = ""
    if model_name is not None:
        title += f"[{model_name}] "
    if dataset_name is not None:
        title += f"{dataset_name}"

    if title:
        print("\n" + title)

    print(pd.Series(display_items))


def add_topk_flags(df_out, score_col="y_prob"):
    df_out = df_out.copy()
    n = len(df_out)

    if n == 0:
        df_out["risk_rank"] = np.nan
        df_out["Top10_Flag"] = np.nan
        df_out["Top20_Flag"] = np.nan
        return df_out

    k10 = max(1, int(np.ceil(n * 0.10)))
    k20 = max(1, int(np.ceil(n * 0.20)))

    df_out["risk_rank"] = df_out[score_col].rank(
        ascending=False,
        method="first"
    ).astype(int)

    df_out["Top10_Flag"] = (df_out["risk_rank"] <= k10).astype(int)
    df_out["Top20_Flag"] = (df_out["risk_rank"] <= k20).astype(int)

    return df_out


# =========================================================
# 4. PCA
# =========================================================

def build_pca_once_dataset(
    train_df,
    test_df,
    regular_feature_cols,
    pca_feature_cols,
    entity_col,
    year_col,
    target_col,
    pca_n_components=0.95,
):
    X_train_regular_raw = train_df[regular_feature_cols].copy()
    X_test_regular_raw = test_df[regular_feature_cols].copy()

    regular_imputer = SimpleImputer(strategy="median")
    regular_scaler = StandardScaler()

    X_train_regular = regular_imputer.fit_transform(X_train_regular_raw)
    X_train_regular = regular_scaler.fit_transform(X_train_regular)

    X_test_regular = regular_imputer.transform(X_test_regular_raw)
    X_test_regular = regular_scaler.transform(X_test_regular)

    regular_out_cols = [f"REG_{c}" for c in regular_feature_cols]

    train_regular_df = pd.DataFrame(
        X_train_regular,
        columns=regular_out_cols,
        index=train_df.index,
    )

    test_regular_df = pd.DataFrame(
        X_test_regular,
        columns=regular_out_cols,
        index=test_df.index,
    )

    X_train_pca_raw = train_df[pca_feature_cols].copy()
    X_test_pca_raw = test_df[pca_feature_cols].copy()

    pca_imputer = SimpleImputer(strategy="median")
    pca_scaler = StandardScaler()
    pca_model = PCA(
        n_components=pca_n_components,
        random_state=RANDOM_STATE,
    )

    X_train_pca_base = pca_imputer.fit_transform(X_train_pca_raw)
    X_train_pca_base = pca_scaler.fit_transform(X_train_pca_base)
    X_train_pca = pca_model.fit_transform(X_train_pca_base)

    X_test_pca_base = pca_imputer.transform(X_test_pca_raw)
    X_test_pca_base = pca_scaler.transform(X_test_pca_base)
    X_test_pca = pca_model.transform(X_test_pca_base)

    n_components = int(pca_model.n_components_)
    pc_cols = [f"PC{i+1}" for i in range(n_components)]

    train_pca_df = pd.DataFrame(
        X_train_pca,
        columns=pc_cols,
        index=train_df.index,
    )

    test_pca_df = pd.DataFrame(
        X_test_pca,
        columns=pc_cols,
        index=test_df.index,
    )

    train_model_df = train_df[[entity_col, year_col, target_col]].copy()
    test_model_df = test_df[[entity_col, year_col, target_col]].copy()

    train_model_df = pd.concat(
        [train_model_df, train_regular_df, train_pca_df],
        axis=1,
    )

    test_model_df = pd.concat(
        [test_model_df, test_regular_df, test_pca_df],
        axis=1,
    )

    model_feature_cols = regular_out_cols + pc_cols

    pca_explained_df = pd.DataFrame({
        "Component": pc_cols,
        "Explained_Variance_Ratio": pca_model.explained_variance_ratio_,
        "Cumulative_Explained_Variance": np.cumsum(pca_model.explained_variance_ratio_),
        "Eigenvalue": pca_model.explained_variance_,
    })

    pca_component_weights_df = pd.DataFrame(
        pca_model.components_.T,
        columns=pc_cols,
    )
    pca_component_weights_df.insert(0, "Feature", pca_feature_cols)

    top_rows = []
    for pc in pc_cols:
        temp = pca_component_weights_df[["Feature", pc]].copy()
        temp["Abs_Loading"] = temp[pc].abs()
        temp = temp.sort_values("Abs_Loading", ascending=False).head(10)
        temp.insert(0, "Component", pc)
        temp = temp.rename(columns={pc: "Loading"})
        top_rows.append(temp)

    pca_top_loadings_df = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()

    pca_train_scores_df = train_df[[entity_col, year_col, target_col]].copy()
    for pc in pc_cols:
        pca_train_scores_df[pc] = train_pca_df[pc].values
    pca_train_scores_df["Dataset"] = "Train"

    pca_test_scores_df = test_df[[entity_col, year_col, target_col]].copy()
    for pc in pc_cols:
        pca_test_scores_df[pc] = test_pca_df[pc].values
    pca_test_scores_df["Dataset"] = "Test"

    pca_summary_df = pd.DataFrame({
        "Item": [
            "PCA original feature count",
            "PCA retained component count",
            "PCA cumulative explained variance",
            "Regular feature count",
            "Final model feature count",
        ],
        "Value": [
            len(pca_feature_cols),
            n_components,
            float(pca_explained_df["Cumulative_Explained_Variance"].iloc[-1]),
            len(regular_feature_cols),
            len(model_feature_cols),
        ],
    })

    print("\n===== PCA 预处理完成 =====")
    print("PCA 原始特征数 =", len(pca_feature_cols))
    print("PCA 主成分数量 =", n_components)
    print("PCA 累计解释方差 =", pca_explained_df["Cumulative_Explained_Variance"].iloc[-1])
    print("普通特征数 =", len(regular_feature_cols))
    print("最终模型特征数 =", len(model_feature_cols))

    return (
        train_model_df,
        test_model_df,
        model_feature_cols,
        pca_summary_df,
        pca_explained_df,
        pca_component_weights_df,
        pca_top_loadings_df,
        pca_train_scores_df,
        pca_test_scores_df,
    )


# =========================================================
# 5. CV
# =========================================================

def make_stratified_cv_splits(X, y, n_splits=5):
    y = pd.Series(y).reset_index(drop=True)
    min_class_count = int(y.value_counts().min())

    if min_class_count < n_splits:
        actual_splits = max(2, min_class_count)
        print(
            f"[警告] 少数类样本数 {min_class_count} 小于设定折数 {n_splits}，"
            f"CV_FOLDS 自动调整为 {actual_splits}"
        )
    else:
        actual_splits = n_splits

    cv = StratifiedKFold(
        n_splits=actual_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    splits = list(cv.split(X, y))

    print("\n===== Stratified CV folds =====")
    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        y_train_fold = y.iloc[train_idx]
        y_valid_fold = y.iloc[valid_idx]
        print(
            f"fold={fold_id} | "
            f"train_n={len(train_idx)}, valid_n={len(valid_idx)}, "
            f"train_pos={int(y_train_fold.sum())}, valid_pos={int(y_valid_fold.sum())}"
        )

    return splits


# =========================================================
# 6. 模型定义
# =========================================================

def make_model_pipeline(model_name, random_state, y_train):
    """
    主模型版本：只保留 TabICLv2。
    """
    if model_name != "TabICLv2":
        raise ValueError(f"当前脚本只支持主模型 TabICLv2，不再运行其他模型：{model_name}")

    return ImbPipeline(steps=[
        ("model", TabICLClassifier(
            random_state=random_state,
            device=None,
            kv_cache=False,
            n_estimators=8,
            verbose=False,
            model_path=LOCAL_TABICL_MODEL_PATH,
            allow_auto_download=False,
        )),
    ])


def get_param_distributions(model_name: str):
    
    if model_name != "TabICLv2":
        raise ValueError(f"当前脚本只支持主模型 TabICLv2，不再提供其他模型调参空间：{model_name}")
    return None


# =========================================================
# 7. OOF 概率、阈值和调参
# =========================================================

def choose_threshold_from_pr_curve(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    oof_ranking_metrics = calc_ranking_metrics(y_true, y_prob)

    def add_oof_metrics(out_dict):
        for k, v in oof_ranking_metrics.items():
            out_dict[f"OOF_{k}"] = v

        # 为了兼容推荐模型选择，额外保留下划线版本
        out_dict["OOF_PR_AUC"] = oof_ranking_metrics.get("PR-AUC", np.nan)
        out_dict["OOF_ROC_AUC"] = oof_ranking_metrics.get("ROC-AUC", np.nan)
        out_dict["OOF_KS"] = oof_ranking_metrics.get("KS", np.nan)

        return out_dict

    if len(np.unique(y_true)) < 2:
        return add_oof_metrics({
            "best_threshold": 0.5,
            "best_f1": np.nan,
            "best_precision": np.nan,
            "best_recall": np.nan,
            "oof_pr_auc": np.nan,
            "threshold_note": "Only one class in OOF labels",
        })

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    oof_pr_auc = average_precision_score(y_true, y_prob)

    if len(thresholds) == 0:
        return add_oof_metrics({
            "best_threshold": 0.5,
            "best_f1": np.nan,
            "best_precision": np.nan,
            "best_recall": np.nan,
            "oof_pr_auc": float(oof_pr_auc),
            "threshold_note": "No valid threshold",
        })

    p = precision[:-1]
    r = recall[:-1]
    f1_values = 2 * p * r / (p + r + 1e-12)

    if THRESHOLD_SELECTION_METHOD == "PrecisionConstraint_MaxRecall":
        valid = p >= MIN_PRECISION

        if valid.sum() > 0:
            valid_indices = np.where(valid)[0]
            best_idx = valid_indices[np.argmax(r[valid_indices])]
            note = f"Max Recall under Precision >= {MIN_PRECISION}"
        else:
            best_idx = int(np.nanargmax(p))
            note = (
                f"No threshold reached Precision >= {MIN_PRECISION}; "
                f"fallback to max Precision"
            )

    elif THRESHOLD_SELECTION_METHOD == "FBeta_Max":
        beta = FBETA_BETA
        fbeta_values = (1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-12)
        best_idx = int(np.nanargmax(fbeta_values))
        note = f"Max F-beta, beta={beta}"

    elif THRESHOLD_SELECTION_METHOD == "PR_Curve_MaxF1":
        best_idx = int(np.nanargmax(f1_values))
        note = "Max F1 on OOF PR curve"

    else:
        raise ValueError(f"未知 THRESHOLD_SELECTION_METHOD: {THRESHOLD_SELECTION_METHOD}")

    return add_oof_metrics({
        "best_threshold": float(thresholds[best_idx]),
        "best_f1": float(f1_values[best_idx]),
        "best_precision": float(p[best_idx]),
        "best_recall": float(r[best_idx]),
        "oof_pr_auc": float(oof_pr_auc),
        "threshold_note": note,
    })


def get_oof_prob_for_threshold(X, y, model_name, cv_splits, best_params=None):
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    all_valid_y = []
    all_valid_prob = []
    all_valid_index = []

    for fold_id, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_va = X.iloc[valid_idx]
        y_va = y.iloc[valid_idx]

        fold_model = make_model_pipeline(model_name, RANDOM_STATE, y_tr)

        if best_params:
            try:
                fold_model.set_params(**best_params)
            except Exception as e:
                print(f"[{model_name}] fold {fold_id} 设置 best_params 失败：{e}")

        fold_model.fit(X_tr, y_tr)
        prob_va = get_pred_prob(fold_model, X_va)

        all_valid_y.extend(y_va.tolist())
        all_valid_prob.extend(prob_va.tolist())
        all_valid_index.extend(valid_idx.tolist())

    return np.asarray(all_valid_y), np.asarray(all_valid_prob), np.asarray(all_valid_index)


def tune_model(train_df, feature_cols, target_col, model_name):
    train_df = train_df.reset_index(drop=True).copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)

    cv_splits = make_stratified_cv_splits(
        X=X_train,
        y=y_train,
        n_splits=CV_FOLDS,
    )

    base_model = make_model_pipeline(model_name, RANDOM_STATE, y_train)
    best_params = {}

    param_distributions = get_param_distributions(model_name)

    if param_distributions is not None and len(cv_splits) >= 2:
        try:
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=TUNING_N_ITER,
                scoring=TUNING_SCORING,
                n_jobs=1,
                cv=cv_splits,
                refit=True,
                random_state=RANDOM_STATE,
                verbose=0,
                error_score="raise",
            )
            search.fit(X_train, y_train)

            best_model = search.best_estimator_
            best_params = search.best_params_

            print(f"[{model_name}] 调参成功。Best CV {TUNING_SCORING} = {search.best_score_:.6f}")
            print(f"[{model_name}] Best params = {best_params}")

        except Exception as e:
            print(f"[{model_name}] 调参失败，使用默认参数模型。错误：{e}")
            print(traceback.format_exc())
            base_model.fit(X_train, y_train)
            best_model = base_model

    else:
        if param_distributions is None:
            print(f"[{model_name}] 无调参空间，使用默认参数。")
        else:
            print(f"[{model_name}] CV folds 不足，跳过调参。")
        base_model.fit(X_train, y_train)
        best_model = base_model

    if len(cv_splits) >= 2:
        try:
            y_oof, oof_prob, _ = get_oof_prob_for_threshold(
                X=X_train,
                y=y_train,
                model_name=model_name,
                cv_splits=cv_splits,
                best_params=best_params,
            )
            threshold_info = choose_threshold_from_pr_curve(
                y_true=y_oof,
                y_prob=oof_prob,
            )

        except Exception as e:
            print(f"[{model_name}] OOF 指标或阈值搜索失败，使用 0.5。错误：{e}")
            print(traceback.format_exc())
            threshold_info = {
                "best_threshold": 0.5,
                "best_f1": np.nan,
                "best_precision": np.nan,
                "best_recall": np.nan,
                "oof_pr_auc": np.nan,
                "OOF_PR_AUC": np.nan,
                "OOF_KS": np.nan,
                "threshold_note": "Threshold search failed; use 0.5",
            }
    else:
        threshold_info = {
            "best_threshold": 0.5,
            "best_f1": np.nan,
            "best_precision": np.nan,
            "best_recall": np.nan,
            "oof_pr_auc": np.nan,
            "OOF_PR_AUC": np.nan,
            "OOF_KS": np.nan,
            "threshold_note": "No valid CV split; use 0.5",
        }

    return best_model, best_params, threshold_info


# =========================================================
# 8-15. 单个目标任务完整运行流程
# =========================================================

def run_one_target(task_config: dict):
    TASK_NAME = task_config["task_name"]
    TARGET_COL_REF = task_config["target_col_ref"]
    OUTPUT_SUBDIR = task_config["output_subdir"]

    print("\n" + "#" * 120)
    print(f"开始目标任务：{TASK_NAME}")
    print(f"目标列：{TARGET_COL_REF}")
    print(f"输出文件夹：{os.path.join(OUTPUT_DIR, OUTPUT_SUBDIR)}")
    print("#" * 120)

    # =========================================================
    # 8. 读取原始数据
    # =========================================================

    df_data = pd.read_excel(FILE_PATH)

    ENTITY_COL = resolve_col_name(df_data, ENTITY_COL_REF)
    YEAR_COL = resolve_col_name(df_data, YEAR_COL_REF)
    TARGET_COL = resolve_col_name(df_data, TARGET_COL_REF)

    if TARGET_COL not in df_data.columns:
        raise KeyError(f"目标列不存在：{TARGET_COL}")

    # 把四个预测目标列全部排除在特征之外，避免目标泄漏
    TARGET_EXCLUDE_COLS = [
        resolve_col_name(df_data, c)
        for c in TARGET_COL_REFS_TO_EXCLUDE
        if (isinstance(c, int) or c in df_data.columns)
    ]
    TARGET_EXCLUDE_COLS = deduplicate_keep_order(TARGET_EXCLUDE_COLS)
    EXCLUDE_COLS = deduplicate_keep_order([ENTITY_COL, YEAR_COL] + TARGET_EXCLUDE_COLS)

    REGULAR_FEATURE_COLS = [resolve_col_name(df_data, c) for c in FEATURE_COL_REFS]
    PCA_FEATURE_COLS = [resolve_col_name(df_data, c) for c in PCA_COL_REFS]

    REGULAR_FEATURE_COLS = deduplicate_keep_order(REGULAR_FEATURE_COLS)
    PCA_FEATURE_COLS = deduplicate_keep_order(PCA_FEATURE_COLS)

    REGULAR_FEATURE_COLS = [c for c in REGULAR_FEATURE_COLS if c not in EXCLUDE_COLS]
    PCA_FEATURE_COLS = [c for c in PCA_FEATURE_COLS if c not in EXCLUDE_COLS]

    REGULAR_FEATURE_COLS = [c for c in REGULAR_FEATURE_COLS if c not in PCA_FEATURE_COLS]
    RAW_FEATURE_COLS = REGULAR_FEATURE_COLS + PCA_FEATURE_COLS

    print("\n===== 原始字段信息 =====")
    print("ENTITY_COL =", ENTITY_COL)
    print("YEAR_COL =", YEAR_COL)
    print("TASK_NAME =", TASK_NAME)
    print("TARGET_COL =", TARGET_COL)
    print("目标列排除列表 =", TARGET_EXCLUDE_COLS)

    print("\n===== 原始特征列设置 =====")
    print("普通特征列数量 =", len(REGULAR_FEATURE_COLS))
    print("PCA 原始特征列数量 =", len(PCA_FEATURE_COLS))
    print("原始输入列总数 =", len(RAW_FEATURE_COLS))

    print("\n普通特征列：")
    for i, col in enumerate(REGULAR_FEATURE_COLS, start=1):
        print(f"{i}. {col}")

    print("\nPCA 特征列：")
    for i, col in enumerate(PCA_FEATURE_COLS, start=1):
        print(f"{i}. {col}")


    # =========================================================
    # 9. 数据清洗
    # =========================================================

    use_cols = [ENTITY_COL, YEAR_COL] + RAW_FEATURE_COLS + [TARGET_COL]
    use_cols = deduplicate_keep_order(use_cols)

    df_raw = df_data[use_cols].copy()

    df_raw[ENTITY_COL] = (
        df_raw[ENTITY_COL]
        .astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
        .str.zfill(6)
    )

    for col in RAW_FEATURE_COLS + [TARGET_COL]:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    df_raw[YEAR_COL] = pd.to_numeric(df_raw[YEAR_COL], errors="coerce")

    df_raw = df_raw.dropna(subset=[ENTITY_COL, YEAR_COL, TARGET_COL]).copy()
    df_raw[YEAR_COL] = df_raw[YEAR_COL].astype(int)
    df_raw[TARGET_COL] = df_raw[TARGET_COL].astype(int)

    target_unique = sorted(df_raw[TARGET_COL].dropna().unique().tolist())
    print("\nTARGET unique values =", target_unique)

    if not set(target_unique).issubset({0, 1}):
        raise ValueError(
            f"当前标签列 {TARGET_COL} 不是 0/1 二分类。"
            f"请先将标签转换为 0/1。"
        )

    df_raw = df_raw.sort_values([ENTITY_COL, YEAR_COL]).reset_index(drop=True)

    print("\n===== 原始全样本概况 =====")
    print("raw shape =", df_raw.shape)
    print("raw year range =", df_raw[YEAR_COL].min(), "-", df_raw[YEAR_COL].max())
    print("raw overall label ratio =", df_raw[TARGET_COL].mean())

    raw_year_ratio_df = summarize_yearly_positive_ratio(
        df_raw,
        YEAR_COL,
        TARGET_COL,
        "raw_sample",
    )

    print("\n===== 原始每年正类比例 =====")
    print(raw_year_ratio_df)


    # =========================================================
    # 10. 固定时间切分
    # =========================================================

    train_raw_df = df_raw[
        (df_raw[YEAR_COL] >= TRAIN_START_YEAR)
        & (df_raw[YEAR_COL] <= TRAIN_END_YEAR)
    ].copy()

    test_raw_df = df_raw[
        (df_raw[YEAR_COL] >= TEST_START_YEAR)
        & (df_raw[YEAR_COL] <= TEST_END_YEAR)
    ].copy()

    print("\n===== 原始切分结果 =====")
    print("train_raw shape =", train_raw_df.shape)
    print("test_raw shape =", test_raw_df.shape)

    if train_raw_df.empty:
        raise ValueError("训练集为空。请检查 TRAIN_START_YEAR / TRAIN_END_YEAR。")
    if test_raw_df.empty:
        raise ValueError("测试集为空。请检查 TEST_START_YEAR / TEST_END_YEAR。")

    print("train positive ratio =", train_raw_df[TARGET_COL].mean())
    print("test positive ratio =", test_raw_df[TARGET_COL].mean())

    train_ratio_df = summarize_yearly_positive_ratio(
        train_raw_df,
        YEAR_COL,
        TARGET_COL,
        "train",
    )

    test_ratio_df = summarize_yearly_positive_ratio(
        test_raw_df,
        YEAR_COL,
        TARGET_COL,
        "test",
    )

    print("\n===== 训练集每年正类比例 =====")
    print(train_ratio_df)

    print("\n===== 测试集每年正类比例 =====")
    print(test_ratio_df)


    # =========================================================
    # 11. PCA First
    # =========================================================

    (
        train_df,
        test_df,
        FEATURE_COLS,
        pca_summary_df,
        pca_explained_df,
        pca_component_weights_df,
        pca_top_loadings_df,
        pca_train_scores_df,
        pca_test_scores_df,
    ) = build_pca_once_dataset(
        train_df=train_raw_df,
        test_df=test_raw_df,
        regular_feature_cols=REGULAR_FEATURE_COLS,
        pca_feature_cols=PCA_FEATURE_COLS,
        entity_col=ENTITY_COL,
        year_col=YEAR_COL,
        target_col=TARGET_COL,
        pca_n_components=PCA_N_COMPONENTS,
    )

    print("\n===== PCA 后建模数据 =====")
    print("train_df shape =", train_df.shape)
    print("test_df shape =", test_df.shape)
    print("FEATURE_COLS count =", len(FEATURE_COLS))
    print("FEATURE_COLS preview =", FEATURE_COLS[:10])


    # =========================================================
    # 12. 模型训练与评估
    # =========================================================

    all_results = []
    sample_outputs = []
    threshold_summary = []
    fitted_models = {}

    for model_name in MODEL_NAMES:
        print("\n" + "=" * 100)
        print(f"开始模型：{model_name}")
        print("=" * 100)

        try:
            model, best_params, threshold_info = tune_model(
                train_df=train_df,
                feature_cols=FEATURE_COLS,
                target_col=TARGET_COL,
                model_name=model_name,
            )

            fitted_models[model_name] = model
            best_threshold = threshold_info["best_threshold"]

            threshold_row = {
                "Target_Task": TASK_NAME,
                "Target_Col": TARGET_COL,
                "Model": model_name,
                "Model_Selection_Metric": MODEL_SELECTION_METRIC,
                "CV_Method": f"Stratified_{CV_FOLDS}_Fold",
                "Threshold_Selection_Method": THRESHOLD_SELECTION_METHOD,
                "Min_Precision": MIN_PRECISION if THRESHOLD_SELECTION_METHOD == "PrecisionConstraint_MaxRecall" else np.nan,
                "FBeta_Beta": FBETA_BETA if THRESHOLD_SELECTION_METHOD == "FBeta_Max" else np.nan,
                "Best_Threshold": best_threshold,

                "OOF_PR_AUC": threshold_info.get("OOF_PR_AUC", threshold_info.get("oof_pr_auc", np.nan)),
                "OOF_ROC_AUC": threshold_info.get("OOF_ROC_AUC", np.nan),
                "OOF_KS": threshold_info.get("OOF_KS", np.nan),

                "OOF_Best_F1": threshold_info.get("best_f1", np.nan),
                "OOF_Best_Precision": threshold_info.get("best_precision", np.nan),
                "OOF_Best_Recall": threshold_info.get("best_recall", np.nan),

                "Threshold_Note": threshold_info.get("threshold_note", ""),
                "Best_Params": str(best_params),
            }

            for metric_name in MAIN_METRIC_COLS:
                threshold_row[f"OOF_{metric_name}"] = threshold_info.get(f"OOF_{metric_name}", np.nan)

            threshold_summary.append(threshold_row)

            print(f"\n[{model_name}] 最优阈值 = {best_threshold:.6f}")
            print(f"[{model_name}] OOF_PR_AUC = {threshold_row['OOF_PR_AUC']}")
            print(f"[{model_name}] OOF_KS = {threshold_row['OOF_KS']}")
            print(f"[{model_name}] OOF_Recall@Top10% = {threshold_row.get('OOF_Recall@Top10%', np.nan)}")
            print(f"[{model_name}] OOF_NDCG@Top10% = {threshold_row.get('OOF_NDCG@Top10%', np.nan)}")
            print(f"[{model_name}] Threshold note = {threshold_info.get('threshold_note', '')}")

            # -----------------------------
            # 样本内训练集评估
            # -----------------------------
            X_train = train_df[FEATURE_COLS]
            y_train = train_df[TARGET_COL].astype(int)

            train_prob = get_pred_prob(model, X_train)
            train_pred = (train_prob >= best_threshold).astype(int)

            train_metrics = calc_metrics(y_train, train_pred, train_prob)
            train_metrics.update({
                "Target_Task": TASK_NAME,
                "Target_Col": TARGET_COL,
                "Model": model_name,
                "Dataset": "In_Sample_Train",
                "Year": f"{TRAIN_START_YEAR}-{TRAIN_END_YEAR}",
                "Threshold": best_threshold,
                "CV_Method": f"Stratified_{CV_FOLDS}_Fold",
                "Threshold_Selection_Method": THRESHOLD_SELECTION_METHOD,
                "Min_Precision": MIN_PRECISION if THRESHOLD_SELECTION_METHOD == "PrecisionConstraint_MaxRecall" else np.nan,
                "OOF_PR_AUC": threshold_row["OOF_PR_AUC"],
                "OOF_ROC_AUC": threshold_row["OOF_ROC_AUC"],
                "OOF_KS": threshold_row["OOF_KS"],
                "OOF_Recall@Top10%": threshold_row.get("OOF_Recall@Top10%", np.nan),
                "OOF_Recall@Top20%": threshold_row.get("OOF_Recall@Top20%", np.nan),
                "OOF_Lift@Top10%": threshold_row.get("OOF_Lift@Top10%", np.nan),
                "OOF_Lift@Top20%": threshold_row.get("OOF_Lift@Top20%", np.nan),
                "OOF_NDCG@Top10%": threshold_row.get("OOF_NDCG@Top10%", np.nan),
                "OOF_NDCG@Top20%": threshold_row.get("OOF_NDCG@Top20%", np.nan),
                "Threshold_Note": threshold_info.get("threshold_note", ""),
                "Best Params": str(best_params),
            })
            all_results.append(train_metrics)

            print_main_metrics(
                metrics_dict=train_metrics,
                model_name=model_name,
                dataset_name="样本内训练集结果",
            )

            train_out = train_df[[ENTITY_COL, YEAR_COL, TARGET_COL]].copy()
            train_out["target_task"] = TASK_NAME
            train_out["target_col"] = TARGET_COL
            train_out["y_prob"] = train_prob
            train_out["y_pred"] = train_pred
            train_out["model"] = model_name
            train_out["dataset"] = "In_Sample_Train"
            train_out["threshold"] = best_threshold
            train_out = add_topk_flags(train_out)
            sample_outputs.append(train_out)

            # -----------------------------
            # 样本外整体测试集
            # -----------------------------
            X_test = test_df[FEATURE_COLS]
            y_test = test_df[TARGET_COL].astype(int)

            test_prob = get_pred_prob(model, X_test)
            test_pred = (test_prob >= best_threshold).astype(int)

            test_metrics = calc_metrics(y_test, test_pred, test_prob)
            test_metrics.update({
                "Target_Task": TASK_NAME,
                "Target_Col": TARGET_COL,
                "Model": model_name,
                "Dataset": "Out_of_Sample_Test_All",
                "Year": f"{TEST_START_YEAR}-{TEST_END_YEAR}",
                "Threshold": best_threshold,
                "CV_Method": f"Stratified_{CV_FOLDS}_Fold",
                "Threshold_Selection_Method": THRESHOLD_SELECTION_METHOD,
                "Min_Precision": MIN_PRECISION if THRESHOLD_SELECTION_METHOD == "PrecisionConstraint_MaxRecall" else np.nan,
                "OOF_PR_AUC": threshold_row["OOF_PR_AUC"],
                "OOF_ROC_AUC": threshold_row["OOF_ROC_AUC"],
                "OOF_KS": threshold_row["OOF_KS"],
                "OOF_Recall@Top10%": threshold_row.get("OOF_Recall@Top10%", np.nan),
                "OOF_Recall@Top20%": threshold_row.get("OOF_Recall@Top20%", np.nan),
                "OOF_Lift@Top10%": threshold_row.get("OOF_Lift@Top10%", np.nan),
                "OOF_Lift@Top20%": threshold_row.get("OOF_Lift@Top20%", np.nan),
                "OOF_NDCG@Top10%": threshold_row.get("OOF_NDCG@Top10%", np.nan),
                "OOF_NDCG@Top20%": threshold_row.get("OOF_NDCG@Top20%", np.nan),
                "Threshold_Note": threshold_info.get("threshold_note", ""),
                "Best Params": str(best_params),
            })
            all_results.append(test_metrics)

            print_main_metrics(
                metrics_dict=test_metrics,
                model_name=model_name,
                dataset_name="样本外整体测试集结果",
            )

            test_out = test_df[[ENTITY_COL, YEAR_COL, TARGET_COL]].copy()
            test_out["target_task"] = TASK_NAME
            test_out["target_col"] = TARGET_COL
            test_out["y_prob"] = test_prob
            test_out["y_pred"] = test_pred
            test_out["model"] = model_name
            test_out["dataset"] = "Out_of_Sample_Test_All"
            test_out["threshold"] = best_threshold
            test_out = add_topk_flags(test_out)
            sample_outputs.append(test_out)

            # -----------------------------
            # TabICLv2 可解释机器学习分析
            # 说明：模型预测使用完整特征 REG_* + PC*；
            #      但 SHAP 输出和绘图只保留不做 PCA 的 REG_* 特征。
            # -----------------------------
            if model_name == "TabICLv2":
                xai_output_dir = os.path.join(
                    OUTPUT_DIR,
                    OUTPUT_SUBDIR,
                    "XAI_TabICLv2_regular_nonPC",
                )

                run_tabicl_shap_xai(
                    model=model,
                    X_train=X_train,
                    X_test=X_test,
                    feature_cols=FEATURE_COLS,
                    output_dir=xai_output_dir,
                    task_name=TASK_NAME,
                    max_background=80,
                    max_explain=200,
                    top_n_display=None,
                    top_n_gam=None,
                )

            # -----------------------------
            # 样本外分年份测试
            # -----------------------------
            for test_year in range(TEST_START_YEAR, TEST_END_YEAR + 1):
                test_year_df = test_df[test_df[YEAR_COL] == test_year].copy()

                if test_year_df.empty:
                    continue

                X_ty = test_year_df[FEATURE_COLS]
                y_ty = test_year_df[TARGET_COL].astype(int)

                prob_ty = get_pred_prob(model, X_ty)
                pred_ty = (prob_ty >= best_threshold).astype(int)

                metrics_ty = calc_metrics(y_ty, pred_ty, prob_ty)
                metrics_ty.update({
                    "Target_Task": TASK_NAME,
                    "Target_Col": TARGET_COL,
                    "Model": model_name,
                    "Dataset": "Out_of_Sample_Test_Year",
                    "Year": test_year,
                    "Threshold": best_threshold,
                    "CV_Method": f"Stratified_{CV_FOLDS}_Fold",
                    "Threshold_Selection_Method": THRESHOLD_SELECTION_METHOD,
                    "Min_Precision": MIN_PRECISION if THRESHOLD_SELECTION_METHOD == "PrecisionConstraint_MaxRecall" else np.nan,
                    "OOF_PR_AUC": threshold_row["OOF_PR_AUC"],
                    "OOF_ROC_AUC": threshold_row["OOF_ROC_AUC"],
                    "OOF_KS": threshold_row["OOF_KS"],
                    "OOF_Recall@Top10%": threshold_row.get("OOF_Recall@Top10%", np.nan),
                    "OOF_Recall@Top20%": threshold_row.get("OOF_Recall@Top20%", np.nan),
                    "OOF_Lift@Top10%": threshold_row.get("OOF_Lift@Top10%", np.nan),
                    "OOF_Lift@Top20%": threshold_row.get("OOF_Lift@Top20%", np.nan),
                    "OOF_NDCG@Top10%": threshold_row.get("OOF_NDCG@Top10%", np.nan),
                    "OOF_NDCG@Top20%": threshold_row.get("OOF_NDCG@Top20%", np.nan),
                    "Threshold_Note": threshold_info.get("threshold_note", ""),
                    "Best Params": str(best_params),
                })
                all_results.append(metrics_ty)

                print_main_metrics(
                    metrics_dict=metrics_ty,
                    model_name=model_name,
                    dataset_name=f"样本外预测年份 {test_year} 结果",
                )

                test_year_out = test_year_df[[ENTITY_COL, YEAR_COL, TARGET_COL]].copy()
                test_year_out["target_task"] = TASK_NAME
                test_year_out["target_col"] = TARGET_COL
                test_year_out["y_prob"] = prob_ty
                test_year_out["y_pred"] = pred_ty
                test_year_out["model"] = model_name
                test_year_out["dataset"] = f"Out_of_Sample_Test_{test_year}"
                test_year_out["threshold"] = best_threshold
                test_year_out = add_topk_flags(test_year_out)
                sample_outputs.append(test_year_out)

        except Exception as e:
            print(f"[{model_name}] 运行失败：{e}")
            print(traceback.format_exc())
            continue


    # =========================================================
    # 13. 汇总结果
    # =========================================================

    results_df = pd.DataFrame(all_results)
    threshold_df = pd.DataFrame(threshold_summary)

    print("\n===== OOF 阈值与排序指标汇总 =====")
    print(threshold_df)

    print("\n===== 最终汇总结果 =====")
    if not results_df.empty:
        display_cols = [
            "Target_Task",
            "Target_Col",
            "Model",
            "Dataset",
            "Year",
        ] + MAIN_METRIC_COLS + [
            "K@Top10%",
            "K@Top20%",
            "Top_Positive_Count@Top10%",
            "Top_Positive_Count@Top20%",
            "Total_Positive_Count",
            "Base_Rate",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "TP",
            "FP",
            "FN",
            "TN",
            "Threshold",
            "OOF_PR_AUC",
            "OOF_ROC_AUC",
            "OOF_KS",
            "OOF_Recall@Top10%",
            "OOF_Recall@Top20%",
            "OOF_Lift@Top10%",
            "OOF_Lift@Top20%",
            "OOF_NDCG@Top10%",
            "OOF_NDCG@Top20%",
            "Threshold_Note",
        ]

        display_cols = [c for c in display_cols if c in results_df.columns]

        print(
            results_df[display_cols]
            .sort_values(["Dataset", "Model", "Year"], key=lambda s: s.astype(str))
        )
    else:
        print("无结果。")


    # =========================================================
    # 14. 推荐模型
    # =========================================================

    recommended_model_df = pd.DataFrame()

    if not threshold_df.empty and MODEL_SELECTION_METRIC in threshold_df.columns:
        threshold_df_valid = threshold_df.dropna(subset=[MODEL_SELECTION_METRIC]).copy()

        if not threshold_df_valid.empty:
            threshold_df_valid = threshold_df_valid.sort_values(
                by=MODEL_SELECTION_METRIC,
                ascending=False,
            ).reset_index(drop=True)

            best_model_by_metric = threshold_df_valid.loc[0, "Model"]

            print(f"\n===== 基于 {MODEL_SELECTION_METRIC} 的推荐模型 =====")
            print("Recommended Model =", best_model_by_metric)
            print(threshold_df_valid.head(10))

            recommended_model_df = results_df[
                results_df["Model"] == best_model_by_metric
            ].copy()

            print("\n===== 推荐模型的样本外结果 =====")
            rec_display_cols = [
                "Model",
                "Dataset",
                "Year",
            ] + MAIN_METRIC_COLS + [
                "K@Top10%",
                "K@Top20%",
                "Top_Positive_Count@Top10%",
                "Top_Positive_Count@Top20%",
                "Total_Positive_Count",
                "Base_Rate",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "TP",
                "FP",
                "FN",
                "TN",
                "Threshold",
                "OOF_PR_AUC",
                "OOF_ROC_AUC",
                "OOF_KS",
                "OOF_Recall@Top10%",
                "OOF_Recall@Top20%",
                "OOF_Lift@Top10%",
                "OOF_Lift@Top20%",
                "OOF_NDCG@Top10%",
                "OOF_NDCG@Top20%",
                "Threshold_Note",
            ]

            rec_display_cols = [c for c in rec_display_cols if c in recommended_model_df.columns]

            print(
                recommended_model_df[rec_display_cols]
                .sort_values(["Dataset", "Year"], key=lambda s: s.astype(str))
            )

        else:
            print(f"\n[警告] threshold_df 中没有有效 {MODEL_SELECTION_METRIC}，无法推荐模型。")
    else:
        print(f"\n[警告] threshold_df 为空或不存在 {MODEL_SELECTION_METRIC}，无法推荐模型。")


    # =========================================================
    # 15. 保存结果
    # =========================================================

    if SAVE_EXCEL:
        out_dir = os.path.join(OUTPUT_DIR, OUTPUT_SUBDIR)
        os.makedirs(out_dir, exist_ok=True)

        target_file_tag = re.sub(r"[^0-9A-Za-z_\\-]+", "_", f"{TASK_NAME}_{TARGET_COL}")
        out_path = os.path.join(
            out_dir,
            f"NEV_no_window_PCAFirst_train_{TRAIN_START_YEAR}_{TRAIN_END_YEAR}_"
            f"test_{TEST_START_YEAR}_{TEST_END_YEAR}_"
            f"RankingMetrics_{IMBALANCE_MODE}_{SAMPLER_TYPE}_"
            f"{target_file_tag}.xlsx",
        )

        used_sheet_names = set()

        with pd.ExcelWriter(out_path) as writer:
            raw_year_ratio_df.to_excel(
                writer,
                sheet_name=safe_sheet_name("raw_year_ratio", used_sheet_names),
                index=False,
            )

            train_ratio_df.to_excel(
                writer,
                sheet_name=safe_sheet_name("train_year_ratio", used_sheet_names),
                index=False,
            )

            test_ratio_df.to_excel(
                writer,
                sheet_name=safe_sheet_name("test_year_ratio", used_sheet_names),
                index=False,
            )

            feature_info_df = pd.DataFrame({
                "Feature_Type": (
                    ["regular"] * len(REGULAR_FEATURE_COLS)
                    + ["pca_original"] * len(PCA_FEATURE_COLS)
                    + ["model_feature"] * len(FEATURE_COLS)
                ),
                "Feature_Name": REGULAR_FEATURE_COLS + PCA_FEATURE_COLS + FEATURE_COLS,
            })

            feature_info_df.to_excel(
                writer,
                sheet_name=safe_sheet_name("feature_info", used_sheet_names),
                index=False,
            )

            pca_summary_df.to_excel(
                writer,
                sheet_name=safe_sheet_name("pca_summary", used_sheet_names),
                index=False,
            )

            pca_explained_df.to_excel(
                writer,
                sheet_name=safe_sheet_name("pca_explained_variance", used_sheet_names),
                index=False,
            )

            pca_component_weights_df.to_excel(
                writer,
                sheet_name=safe_sheet_name("pca_component_weights", used_sheet_names),
                index=False,
            )

            pca_top_loadings_df.to_excel(
                writer,
                sheet_name=safe_sheet_name("pca_top_loadings", used_sheet_names),
                index=False,
            )

            pca_train_scores_df.to_excel(
                writer,
                sheet_name=safe_sheet_name("pca_train_scores", used_sheet_names),
                index=False,
            )

            pca_test_scores_df.to_excel(
                writer,
                sheet_name=safe_sheet_name("pca_test_scores", used_sheet_names),
                index=False,
            )

            threshold_df.to_excel(
                writer,
                sheet_name=safe_sheet_name("oof_threshold_ranking", used_sheet_names),
                index=False,
            )

            results_df.to_excel(
                writer,
                sheet_name=safe_sheet_name("all_results", used_sheet_names),
                index=False,
            )

            if not recommended_model_df.empty:
                recommended_model_df.to_excel(
                    writer,
                    sheet_name=safe_sheet_name("recommended_model", used_sheet_names),
                    index=False,
                )

            for i, df_out in enumerate(sample_outputs):
                model_name = str(df_out["model"].iloc[0])
                dataset_name = str(df_out["dataset"].iloc[0])
                base_sheet_name = f"{model_name}_{dataset_name}_{i}"
                sheet_name = safe_sheet_name(base_sheet_name, used_sheet_names)
                df_out.to_excel(writer, sheet_name=sheet_name, index=False)

        print("\n文件已保存：")
        print(out_path)

# =========================================================
# 16. 多目标循环入口
# =========================================================

if __name__ == "__main__":
    print_global_settings()

    failed_tasks = []
    for task_config in TARGET_TASKS:
        try:
            run_one_target(task_config)
        except Exception as e:
            failed_tasks.append({
                "Target_Task": task_config.get("task_name"),
                "Target_Col": task_config.get("target_col_ref"),
                "Error": str(e),
            })
            print("\n" + "!" * 120)
            print(f"目标任务失败：{task_config.get('task_name')} | {task_config.get('target_col_ref')}")
            print(e)
            print(traceback.format_exc())
            print("!" * 120)
            continue

    print("\n" + "=" * 120)
    print("全部目标任务运行结束。")
    if failed_tasks:
        print("以下目标任务失败：")
        print(pd.DataFrame(failed_tasks))
    else:
        print("四个目标任务均已完成。")
    print("=" * 120)