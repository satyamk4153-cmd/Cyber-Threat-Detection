"""
Comprehensive visualization module for exploratory data analysis (EDA),
attack intelligence, model evaluation metrics, and SOC dashboard analytics.
Supports Matplotlib/Seaborn for publication-grade figures and Plotly for interactive web charts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for all environments
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import config

logger = logging.getLogger(__name__)

# Dark SOC Palette definition for consistent visual branding
SOC_PALETTE = {
    "navy_dark": "#0B132B",
    "charcoal": "#1C2541",
    "panel": "#1E293B",
    "electric_blue": "#3A86FF",
    "cyan": "#00F5D4",
    "alert_red": "#FF0054",
    "success_green": "#38B000",
    "warning_yellow": "#FFBE0B",
    "purple": "#7209B7",
    "text_light": "#F8F9FA",
    "text_muted": "#94A3B8",
}

# Set default Seaborn & Matplotlib visual style
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams.update({
    "figure.autolayout": True,
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "figure.dpi": 200,
})


# ------------------------------------------------------------------------------
# 1. Matplotlib / Seaborn Publication Figures
# ------------------------------------------------------------------------------

def plot_class_distribution_sns(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Plots and saves the full attack category frequency distribution."""
    col = target_col or config.target_column
    counts = df[col].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=counts.values,
        y=counts.index,
        hue=counts.index,
        legend=False,
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Network Traffic Class Distribution (CIC-IDS2017)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Flow Count")
    ax.set_ylabel("Traffic Category")

    # Add count labels to bars
    for idx, val in enumerate(counts.values):
        pct = (val / len(df)) * 100
        ax.text(val + max(counts.values) * 0.01, idx, f"{val:,} ({pct:.1f}%)", va="center", fontsize=9)

    out_file = Path(save_path) if save_path else config.figures_dir / "01_class_distribution.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved class distribution figure to {out_file}")
    return out_file


def plot_benign_vs_attack_sns(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Plots benign vs attack ratio as bar & donut comparison."""
    col = target_col or config.target_column
    benign_label = config.benign_label.upper()
    is_attack = df[col].astype(str).str.strip().str.upper() != benign_label

    benign_cnt = int((~is_attack).sum())
    attack_cnt = int(is_attack.sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    categories = ["BENIGN", "ATTACK"]
    values = [benign_cnt, attack_cnt]
    colors = [SOC_PALETTE["success_green"], SOC_PALETTE["alert_red"]]
    sns.barplot(x=categories, y=values, hue=categories, legend=False, palette=colors, ax=ax1)
    ax1.set_title("Traffic Volume: Benign vs Attack", fontweight="bold")
    ax1.set_ylabel("Total Flows")
    for idx, v in enumerate(values):
        ax1.text(idx, v + max(values) * 0.02, f"{v:,} ({(v/len(df))*100:.1f}%)", ha="center")

    # Donut chart
    ax2.pie(
        values,
        labels=categories,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor="w"),
    )
    ax2.set_title("Traffic Proportions", fontweight="bold")

    out_file = Path(save_path) if save_path else config.figures_dir / "02_benign_vs_attack.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    return out_file


def plot_flow_metrics_distribution_sns(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Plots log-scaled distributions of Flow Duration, Bytes/s, Packets/s, and Packet Length."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics = [
        ("Flow Duration", "Flow Duration (Microseconds)"),
        ("Flow Bytes/s", "Flow Bytes per Second"),
        ("Flow Packets/s", "Flow Packets per Second"),
        ("Packet Length Mean", "Mean Packet Length (Bytes)"),
    ]

    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        if col in df.columns:
            clean_series = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            clean_series = clean_series[clean_series > 0]
            if len(clean_series) > 0:
                sns.histplot(clean_series, bins=40, log_scale=True, kde=True, color=SOC_PALETTE["electric_blue"], ax=ax)
                ax.set_title(title, fontweight="bold")
                ax.set_xlabel("Log Scale")
                ax.set_ylabel("Frequency")

    out_file = Path(save_path) if save_path else config.figures_dir / "03_flow_metrics_distribution.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    return out_file


def plot_correlation_heatmap_sns(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Computes and renders correlation matrix heatmap for numeric traffic features."""
    if features:
        sub_df = df[[f for f in features if f in df.columns]]
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns[:20]
        sub_df = df[num_cols]

    corr = sub_df.corr().replace([np.inf, -np.inf], np.nan).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Network Flow Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    out_file = Path(save_path) if save_path else config.figures_dir / "04_correlation_heatmap.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    return out_file


def plot_confusion_matrix_sns(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Plots normalized confusion matrix with raw counts and percentages."""
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(8, 6))
    annot_matrix = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_matrix[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm_norm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")

    out_file = Path(save_path) if save_path else config.figures_dir / "confusion_matrix.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    return out_file


# ------------------------------------------------------------------------------
# 2. Interactive Plotly Figures (for Streamlit Dashboard)
# ------------------------------------------------------------------------------

def get_plotly_class_distribution(df: pd.DataFrame, target_col: Optional[str] = None):
    """Generates interactive Plotly bar chart for class frequencies."""
    import plotly.express as px

    col = target_col or config.target_column
    counts = df[col].value_counts().reset_index()
    counts.columns = ["Traffic_Class", "Count"]
    counts["Percentage"] = (counts["Count"] / len(df) * 100).round(2)

    fig = px.bar(
        counts,
        x="Count",
        y="Traffic_Class",
        orientation="h",
        color="Count",
        color_continuous_scale="Viridis",
        hover_data=["Percentage"],
        title="Network Traffic Class Breakdown",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8F9FA"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def get_plotly_benign_vs_attack(df: pd.DataFrame, target_col: Optional[str] = None):
    """Generates interactive Plotly donut chart comparing Benign vs Malicious flows."""
    import plotly.graph_objects as go

    col = target_col or config.target_column
    benign_label = config.benign_label.upper()
    is_attack = df[col].astype(str).str.strip().str.upper() != benign_label

    labels = ["BENIGN", "ATTACK"]
    values = [int((~is_attack).sum()), int(is_attack.sum())]
    colors = [SOC_PALETTE["success_green"], SOC_PALETTE["alert_red"]]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color="#0B132B", width=2)),
        textinfo="label+percent",
        hoverinfo="label+value+percent",
    )])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8F9FA"),
        title="Benign vs. Threat Traffic Ratio",
    )
    return fig


def get_plotly_model_comparison(comparison_df: pd.DataFrame):
    """Generates interactive multi-metric bar chart comparing evaluated algorithms."""
    import plotly.graph_objects as go

    fig = go.Figure()
    metrics = ["accuracy", "precision", "recall", "macro_f1"]
    colors = [SOC_PALETTE["cyan"], SOC_PALETTE["electric_blue"], SOC_PALETTE["purple"], SOC_PALETTE["warning_yellow"]]

    for idx, m in enumerate(metrics):
        if m in comparison_df.columns:
            fig.add_trace(go.Bar(
                name=m.replace("_", " ").title(),
                x=comparison_df["model_name"],
                y=comparison_df[m],
                marker_color=colors[idx % len(colors)],
            ))

    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8F9FA"),
        title="Model Benchmark: Accuracy, Precision, Recall, and Macro-F1",
        yaxis=dict(range=[0, 1.05], title="Score"),
        xaxis=dict(title="Algorithm"),
    )
    return fig


def get_plotly_feature_importance(importance_dict: Dict[str, float], top_n: int = 20):
    """Generates interactive horizontal bar chart for top N feature importances."""
    import plotly.express as px

    items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    df_imp = pd.DataFrame(items, columns=["Feature", "Importance"])

    fig = px.bar(
        df_imp,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Teal",
        title=f"Top {top_n} Most Influential Traffic Features",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8F9FA"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def get_plotly_roc_curves(roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]]):
    """
    Renders interactive ROC curves for multiple models.
    roc_data: Dict mapping model_name -> (fpr, tpr, roc_auc)
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    # Baseline diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Random Chance (AUC = 0.50)",
    ))

    colors = [
        SOC_PALETTE["cyan"],
        SOC_PALETTE["electric_blue"],
        SOC_PALETTE["purple"],
        SOC_PALETTE["warning_yellow"],
        SOC_PALETTE["success_green"],
    ]

    for idx, (m_name, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{m_name} (AUC = {auc_val:.3f})",
            line=dict(color=colors[idx % len(colors)], width=2.5),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8F9FA"),
        title="Receiver Operating Characteristic (ROC) Curves",
        xaxis=dict(title="False Positive Rate (FPR)", range=[0, 1]),
        yaxis=dict(title="True Positive Rate (TPR / Recall)", range=[0, 1.05]),
    )
    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_loader import load_dataset
    df, _ = load_dataset(use_sample_if_empty=True)
    f1 = plot_class_distribution_sns(df)
    f2 = plot_benign_vs_attack_sns(df)
    f3 = plot_flow_metrics_distribution_sns(df)
    print("Figures generated successfully at:", f1, f2, f3)
