import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make plots cleaner
plt.style.use("ggplot")

# ============================================================
# LOAD PREDICTION RESULTS
# ============================================================

df = pd.read_csv("fifa2026_with_groups_predictions.csv")

print("Loaded prediction file with shape:", df.shape)
print(df.head())


# ============================================================
# 1. FEATURE IMPORTANCE (Random Forest)
# ============================================================

def plot_feature_importance(model_file="rf_feature_importances.csv"):
    """
    You need to save RF feature importances from your main ML code:
        pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_})
            .to_csv("rf_feature_importances.csv", index=False)
    """
    try:
        imp = pd.read_csv(model_file)
    except:
        print("\n Missing rf_feature_importances.csv. Please export RF importances in main script.")
        return

    imp = imp.sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(imp["feature"], imp["importance"], color="steelblue")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()


# ============================================================
# 2. TOP 15 ML PROBABILITIES (Bar Chart)
# ============================================================

def plot_ml_top15(df):
    top = df.sort_values("prob_champion_final", ascending=False).head(15)

    plt.figure(figsize=(12, 6))
    plt.barh(top["Team"], top["prob_champion_final"], color="darkgreen")
    plt.title("Top 15 Teams by ML Probability of Winning World Cup 2026")
    plt.xlabel("Probability")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# ============================================================
# 3. MONTE CARLO TOP 10 (Bar Chart)
# ============================================================

def plot_mc_top10(df):
    top = df.sort_values("mc_win_prob", ascending=False).head(10)

    plt.figure(figsize=(12, 6))
    plt.barh(top["Team"], top["mc_win_prob"], color="royalblue")
    plt.title("Top 10 Teams by Monte Carlo Win Probability")
    plt.xlabel("Probability")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. STRENGTH INDEX vs GROUP DIFFICULTY
# ============================================================

def plot_strength_vs_group(df):

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=df["group_difficulty_norm"],
        y=df["strength_index"],
        hue=df["Group"],
        s=120,
        palette="tab20",
        edgecolor="black"
    )

    for _, row in df.iterrows():
        plt.text(row["group_difficulty_norm"] + 0.005,
                 row["strength_index"] + 0.005,
                 row["Team"],
                 fontsize=7)

    plt.title("Team Strength vs Group Difficulty (2026 WC)")
    plt.xlabel("Group Difficulty (Normalized)")
    plt.ylabel("Strength Index")
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. CORRELATION HEATMAP OF FEATURES
# ============================================================

def plot_corr_heatmap(df):
    feature_cols = [
        "Elo", "strength_index", "group_difficulty_norm",
        "prob_champion_rf_norm", "prob_champion_lr_norm",
        "prob_champion_xgb_norm", "prob_champion_final",
        "mc_win_prob"
    ]

    plt.figure(figsize=(10, 8))
    corr = df[feature_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap of Model Features & Probabilities")
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN VISUALIZATION EXECUTION
# ============================================================

if __name__ == "__main__":

    print("\n Generating Visualization 1: RF Feature Importance")
    plot_feature_importance()

    print("\nGenerating Visualization 2: ML Top 15")
    plot_ml_top15(df)

    print("\n Generating Visualization 3: Monte Carlo Top 10")
    plot_mc_top10(df)

    print("\n Generating Visualization 4: Strength vs Group Difficulty")
    plot_strength_vs_group(df)

    print("\n Generating Visualization 5: Correlation Heatmap")
    plot_corr_heatmap(df)

    print("\nAll visualizations generated.")
