"""
Visualization script for FIFA World Cup 2026 predictions.

This script:
- Loads fifa2026_predictions.csv (created by your main model script)
- Plots:
    1) Top-10 teams by ENSEMBLE champion probability
    2) Comparison of RF vs LR vs Ensemble for top-10 teams

Requirements:
- Run your main model file first so fifa2026_predictions.csv exists:
    python fifa2026_full_history_model.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_predictions(csv_path="fifa2026_predictions.csv"):
    print(f"Loading predictions from {csv_path} ...")
    df = pd.read_csv(csv_path)
    # Ensure proper sorting by ensemble probability
    df = df.sort_values("prob_champion_ensemble", ascending=False).reset_index(drop=True)
    print("Loaded predictions shape:", df.shape)
    return df


def plot_top10_ensemble(df):
    """
    Bar chart of top-10 teams by ensemble champion probability.
    """
    top10 = df.head(10)

    teams = top10["Team"]
    probs = top10["prob_champion_ensemble"]

    plt.figure(figsize=(10, 6))
    plt.bar(teams, probs)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Ensemble Probability")
    plt.title("FIFA 2026 – Top 10 Teams by Ensemble Champion Probability")
    plt.tight_layout()
    plt.show()


def plot_rf_lr_ensemble_comparison(df):
    """
    Grouped bar chart for top-10 teams:
    - Random Forest normalized probability
    - Logistic Regression normalized probability
    - Ensemble probability
    """
    top10 = df.head(10).copy()

    x = np.arange(len(top10))  # positions
    width = 0.25  # bar width

    rf = top10["prob_champion_rf_norm"]
    lr = top10["prob_champion_lr_norm"]
    ens = top10["prob_champion_ensemble"]

    plt.figure(figsize=(12, 6))

    plt.bar(x - width, rf, width, label="Random Forest")
    plt.bar(x, lr, width, label="Logistic Regression")
    plt.bar(x + width, ens, width, label="Ensemble")

    plt.xticks(x, top10["Team"], rotation=45, ha="right")
    plt.ylabel("Normalized Probability")
    plt.title("FIFA 2026 – RF vs LR vs Ensemble (Top 10 Teams)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    df = load_predictions("fifa2026_predictions.csv")

    # Print top 10 to console as well
    print("\nTop 10 teams by ENSEMBLE probability:")
    print(
        df[["Team", "prob_champion_ensemble"]]
        .head(10)
        .to_string(index=False)
    )

    # Plot 1: Ensemble Top-10
    plot_top10_ensemble(df)

    # Plot 2: RF vs LR vs Ensemble comparison
    plot_rf_lr_ensemble_comparison(df)


if __name__ == "__main__":
    main()
