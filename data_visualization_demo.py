"""
Data Visualization Program (Matplotlib + Seaborn)
-------------------------------------------------
Creates a set of polished visualizations from built-in seaborn datasets
and saves them to an ./outputs folder. Use this as a portfolio-ready
starter script and customize as needed.

Run:
    python data_visualization_demo.py

Outputs (PNG files):
    outputs/
        flights_line.png
        tips_bar.png
        tips_scatter_reg.png
        tips_box_violin.png
        tips_corr_heatmap.png
        iris_pairplot.png
        flights_story.png

Requirements:
    pip install matplotlib seaborn pandas numpy
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------
# Global style
# -------------------------
sns.set_theme(context="talk", style="whitegrid")
plt.rcParams.update({
    "figure.autolayout": True,
    "figure.figsize": (10, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def ensure_output_dir(path: str = "outputs") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_show(fig: plt.Figure, path: str, show: bool = True) -> None:
    fig.savefig(path, dpi=150)
    if show:
        plt.show()   # <-- This will display the chart in a window
    # Do not close immediately, let the user see it



# -------------------------
# Visual 1: Line chart (time series)
# -------------------------

def flights_line(outdir: str) -> None:
    flights = sns.load_dataset("flights")  # columns: year, month, passengers
    # Build a datetime index for clarity
    flights["date"] = pd.to_datetime(flights["year"].astype(str) + "-" + flights["month"].astype(str), format="%Y-%b")
    flights = flights.sort_values("date")

    fig, ax = plt.subplots()
    ax.plot(flights["date"], flights["passengers"], linewidth=2)
    ax.set_title("Airline passengers over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Passengers")

    # Annotate peak
    peak_row = flights.loc[flights["passengers"].idxmax()]
    ax.scatter([peak_row["date"]], [peak_row["passengers"]], s=80)
    ax.annotate(
        f"Peak: {peak_row['passengers']}",
        (peak_row["date"], peak_row["passengers"]),
        xytext=(15, 15), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=1.5)
    )

    save_show(fig, os.path.join(outdir, "flights_line.png"))
    plt.show()



# -------------------------
# Visual 2: Bar chart (categorical aggregation)
# -------------------------

def tips_bar(outdir: str) -> None:
    tips = sns.load_dataset("tips")
    grouped = tips.groupby("day")["tip"].mean().reset_index()
    grouped = grouped.sort_values("tip", ascending=False)

    fig, ax = plt.subplots()
    ax.bar(grouped["day"], grouped["tip"])
    ax.set_title("Average tip by day")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Tip ($)")

    # Highlight best day
    best_day = grouped.iloc[0]
    ax.annotate(
        f"Highest avg tip\n{best_day['day']}: ${best_day['tip']:.2f}",
        (best_day["day"], best_day["tip"]),
        xytext=(0, 20), textcoords="offset points",
        ha="center", arrowprops=dict(arrowstyle="->", lw=1.5)
    )

    save_show(fig, os.path.join(outdir, "tips_bar.png"))


# -------------------------
# Visual 3: Scatter + regression (relationship)
# -------------------------

def tips_scatter_reg(outdir: str) -> None:
    tips = sns.load_dataset("tips")

    fig, ax = plt.subplots()
    # Use seaborn's regplot for regression line and matplotlib for legend
    sns.regplot(data=tips, x="total_bill", y="tip", ax=ax, scatter_kws={"alpha": 0.6})

    # Add a second aesthetic via marker shape by smoker status
    for smoker, df in tips.groupby("smoker"):
        ax.scatter(df["total_bill"], df["tip"], alpha=0.5, label=f"Smoker: {smoker}")

    ax.set_title("Tips vs Total Bill with Regression")
    ax.set_xlabel("Total Bill ($)")
    ax.set_ylabel("Tip ($)")
    ax.legend(title="Group")

    save_show(fig, os.path.join(outdir, "tips_scatter_reg.png"))


# -------------------------
# Visual 4: Box + Violin (distribution + outliers)
# -------------------------

def tips_box_violin(outdir: str) -> None:
    tips = sns.load_dataset("tips")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    sns.boxplot(data=tips, x="day", y="total_bill", ax=axes[0])
    axes[0].set_title("Total Bill by Day (Boxplot)")

    sns.violinplot(data=tips, x="day", y="total_bill", inner="quartile", ax=axes[1])
    axes[1].set_title("Total Bill by Day (Violin)")

    for ax in axes:
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Total Bill ($)")

    fig.tight_layout()
    save_show(fig, os.path.join(outdir, "tips_box_violin.png"))


# -------------------------
# Visual 5: Correlation heatmap (feature relationships)
# -------------------------

def tips_corr_heatmap(outdir: str) -> None:
    tips = sns.load_dataset("tips").copy()
    # Encode categoricals to numeric for correlation (simple demo)
    tips["sex"] = tips["sex"].map({"Male": 1, "Female": 0})
    tips["smoker"] = tips["smoker"].map({"Yes": 1, "No": 0})
    tips["day"] = tips["day"].map({"Thur": 4, "Fri": 5, "Sat": 6, "Sun": 7})
    tips["time"] = tips["time"].map({"Lunch": 0, "Dinner": 1})

    corr = tips.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", square=True, cmap="vlag", ax=ax)
    ax.set_title("Correlation heatmap - Tips dataset (encoded)")

    save_show(fig, os.path.join(outdir, "tips_corr_heatmap.png"))


# -------------------------
# Visual 6: Pairplot (multivariate overview)
# -------------------------

def iris_pairplot(outdir: str) -> None:
    iris = sns.load_dataset("iris")
    g = sns.pairplot(iris, hue="species", corner=True)
    g.fig.suptitle("Iris Pairplot", y=1.02)
    g.savefig(os.path.join(outdir, "iris_pairplot.png"), dpi=150)
    plt.close(g.fig)


# -------------------------
# Visual 7: Storytelling figure with annotations
# -------------------------

def flights_story(outdir: str) -> None:
    flights = sns.load_dataset("flights")
    flights["date"] = pd.to_datetime(flights["year"].astype(str) + "-" + flights["month"].astype(str), format="%Y-%b")
    flights = flights.sort_values("date")

    # Compute rolling mean to reveal trend
    flights["roll_6"] = flights["passengers"].rolling(6, min_periods=1).mean()

    fig, ax = plt.subplots()
    ax.plot(flights["date"], flights["passengers"], alpha=0.4, label="Monthly passengers")
    ax.plot(flights["date"], flights["roll_6"], linewidth=3, label="6-month rolling avg")

    # Annotate a dip and a surge for narrative
    min_idx = flights["passengers"].idxmin()
    max_idx = flights["passengers"].idxmax()
    for idx, label in [(min_idx, "Early trough"), (max_idx, "Record high")]:
        row = flights.loc[idx]
        ax.scatter([row["date"]], [row["passengers"]], s=80)
        ax.annotate(
            f"{label}: {row['passengers']}",
            (row["date"], row["passengers"]),
            xytext=(10, 15), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", lw=1.5)
        )

    ax.set_title("Air travel growth with rolling trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Passengers")
    ax.legend()

    save_show(fig, os.path.join(outdir, "flights_story.png"))


def main() -> None:
    outdir = ensure_output_dir()
    flights_line(outdir)
    tips_bar(outdir)
    tips_scatter_reg(outdir)
    tips_box_violin(outdir)
    tips_corr_heatmap(outdir)
    iris_pairplot(outdir)
    flights_story(outdir)
    print(f"✅ Done. Images saved to: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()
