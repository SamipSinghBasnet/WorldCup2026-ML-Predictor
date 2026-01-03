"""
FIFA World Cup 2026 Prediction using FULL World Cup History (1930–2022)

Uses:
- matches_1930_2022.csv  : all World Cup matches
- world_cup.csv          : champion & runner-up per World Cup
- fifa_ranking_2022-10-06.csv : modern FIFA rankings (used as a strength proxy)
No manual teams_2026_features.csv needed.

We build:
- Historical pre-tournament features for each (Year, Team)
- Labels: 1=Champion, 2=Runner-up, 3=Third, 0=Others
- Train RandomForest + LogisticRegression
- Generate features for 2026 teams from history and ranking
- Predict top 3 for 2026.
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# ---------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------

# 48 teams from your list (qualified + playoff)
TEAMS_2026 = [
    "USA", "MEXICO", "CANADA", "AUSTRALIA", "IRAN", "JAPAN", "JORDAN",
    "KOREA REPUBLIC", "QATAR", "SAUDI ARABIA", "UZBEKISTAN",
    "ALGERIA", "CAPE VERDE", "IVORY COAST", "EGYPT", "GHANA",
    "MOROCCO", "SENEGAL", "SOUTH AFRICA", "TUNISIA", "CURACAO",
    "HAITI", "PANAMA", "ARGENTINA", "BRAZIL", "COLOMBIA", "ECUADOR",
    "PARAGUAY", "URUGUAY", "NEW ZEALAND", "AUSTRIA", "BELGIUM",
    "CROATIA", "ENGLAND", "FRANCE", "GERMANY", "NETHERLANDS",
    "NORWAY", "PORTUGAL", "SCOTLAND", "SPAIN", "SWITZERLAND",
    "BOLIVIA", "DR CONGO", "IRAQ", "JAMAICA", "NEW CALEDONIA",
    "SURINAME",
]

# RandomForest parameters
RF_TREES = 400
RANDOM_STATE = 42

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

print("Loading datasets...")

matches_all = pd.read_csv("matches_1930_2022.csv")
world_cup_info = pd.read_csv("world_cup.csv")
ranking = pd.read_csv("fifa_ranking_2022-10-06.csv")

print("Loaded matches:", matches_all.shape)
print("Loaded world_cup:", world_cup_info.shape)
print("Loaded ranking:", ranking.shape, "\n")

# Normalize team names to uppercase
ranking["team"] = ranking["team"].str.upper()
matches_all["home_team"] = matches_all["home_team"].str.upper()
matches_all["away_team"] = matches_all["away_team"].str.upper()
world_cup_info["Champion"] = world_cup_info["Champion"].str.upper()
world_cup_info["Runner-Up"] = world_cup_info["Runner-Up"].str.upper()

# ---------------------------------------------------------
# 2. BUILD PER-MATCH LONG FORMAT (ONE ROW PER TEAM PER MATCH)
# ---------------------------------------------------------

print("Building long-format match records...")

# outcome flags helper
def outcome_flags(gf, ga):
    if gf > ga:
        return 1, 0, 0, 3  # win, draw, loss, points
    elif gf == ga:
        return 0, 1, 0, 1
    else:
        return 0, 0, 1, 0

home_rows = []
away_rows = []

for _, row in matches_all.iterrows():
    year = row["Year"]
    ht = row["home_team"]
    at = row["away_team"]
    hs = row["home_score"]
    a_s = row["away_score"]

    w, d, l, pts = outcome_flags(hs, a_s)
    home_rows.append(
        {
            "Year": year,
            "Team": ht,
            "gf": hs,
            "ga": a_s,
            "win": w,
            "draw": d,
            "loss": l,
            "points": pts,
        }
    )

    w2, d2, l2, pts2 = outcome_flags(a_s, hs)
    away_rows.append(
        {
            "Year": year,
            "Team": at,
            "gf": a_s,
            "ga": hs,
            "win": w2,
            "draw": d2,
            "loss": l2,
            "points": pts2,
        }
    )

team_match_long = pd.DataFrame(home_rows + away_rows)

print("Long-format rows:", team_match_long.shape, "\n")

# ---------------------------------------------------------
# 3. AGGREGATE TO YEARLY PERFORMANCE PER TEAM
# ---------------------------------------------------------

print("Aggregating to per-year team performance...")

yearly_perf = (
    team_match_long.groupby(["Team", "Year"])
    .agg(
        gf_year=("gf", "sum"),
        ga_year=("ga", "sum"),
        wins_year=("win", "sum"),
        draws_year=("draw", "sum"),
        losses_year=("loss", "sum"),
        points_year=("points", "sum"),
        matches_year=("gf", "count"),
    )
    .reset_index()
)

# sort for cumulative history
yearly_perf = yearly_perf.sort_values(["Team", "Year"])

# cumulative totals (up to and including that year)
for col in ["gf_year", "ga_year", "wins_year", "draws_year", "losses_year", "points_year", "matches_year"]:
    yearly_perf[f"cum_{col}"] = yearly_perf.groupby("Team")[col].cumsum()

# historical totals BEFORE the given year (used for training features)
for col in ["gf_year", "ga_year", "wins_year", "draws_year", "losses_year", "points_year", "matches_year"]:
    yearly_perf[f"hist_{col}"] = (
        yearly_perf.groupby("Team")[col].cumsum().shift(1).fillna(0)
    )

# derived historical ratios
def safe_div(num, den):
    return np.where(den > 0, num / den, 0.0)

yearly_perf["hist_goal_diff"] = yearly_perf["hist_gf_year"] - yearly_perf["hist_ga_year"]
yearly_perf["hist_goals_per_match"] = safe_div(
    yearly_perf["hist_gf_year"], yearly_perf["hist_matches_year"]
)
yearly_perf["hist_goals_conceded_per_match"] = safe_div(
    yearly_perf["hist_ga_year"], yearly_perf["hist_matches_year"]
)
yearly_perf["hist_win_rate"] = safe_div(
    yearly_perf["hist_wins_year"], yearly_perf["hist_matches_year"]
)
yearly_perf["hist_points_per_match"] = safe_div(
    yearly_perf["hist_points_year"], yearly_perf["hist_matches_year"]
)

print("Yearly performance table:", yearly_perf.shape, "\n")

# ---------------------------------------------------------
# 4. LABELS: 1=CHAMPION, 2=RUNNER-UP, 3=THIRD, 0=OTHERS
# ---------------------------------------------------------

print("Creating labels for each (Year, Team)...")

# build champion and runner-up labels
label_df = yearly_perf[["Team", "Year"]].copy()
label_df["label"] = 0  # default others

# champion / runner-up
for _, row in world_cup_info.iterrows():
    y = row["Year"]
    champ = row["Champion"]
    runner_up = row["Runner-Up"]

    # champion
    mask_ch = (label_df["Year"] == y) & (label_df["Team"] == champ)
    label_df.loc[mask_ch, "label"] = 1

    # runner-up
    mask_ru = (label_df["Year"] == y) & (label_df["Team"] == runner_up)
    label_df.loc[mask_ru, "label"] = 2

# third place via "Third place" match in matches_1930_2022
third_labels = []
for y in sorted(matches_all["Year"].unique()):
    third_match = matches_all[
        (matches_all["Year"] == y)
        & (matches_all["Round"].str.contains("Third", case=False, na=False))
    ]
    if not third_match.empty:
        r = third_match.iloc[0]
        if r["home_score"] > r["away_score"]:
            third_team = r["home_team"]
        else:
            third_team = r["away_team"]
        third_labels.append({"Year": y, "Team": third_team})

third_df = pd.DataFrame(third_labels)
third_df["Team"] = third_df["Team"].str.upper()

for _, row in third_df.iterrows():
    mask_3 = (label_df["Year"] == row["Year"]) & (label_df["Team"] == row["Team"])
    label_df.loc[mask_3, "label"] = 3

print("Label distribution:")
print(label_df["label"].value_counts(), "\n")

# merge labels with yearly_perf
train_df = yearly_perf.merge(label_df, on=["Team", "Year"], how="inner")

# ---------------------------------------------------------
# 5. ADD RANKING-BASED STRENGTH FEATURES
# ---------------------------------------------------------

print("Merging FIFA ranking strength...")

ranking_2022 = ranking[["team", "rank", "points"]].rename(
    columns={"team": "Team", "rank": "fifa_rank", "points": "fifa_points"}
)

train_df = train_df.merge(ranking_2022, on="Team", how="left")

# teams missing ranking → use neutral values
median_rank = ranking_2022["fifa_rank"].median()
median_points = ranking_2022["fifa_points"].median()

train_df["fifa_rank"] = train_df["fifa_rank"].fillna(median_rank)
train_df["fifa_points"] = train_df["fifa_points"].fillna(median_points)

print("Training dataframe shape (all tournaments):", train_df.shape, "\n")

# ---------------------------------------------------------
# 6. SELECT FEATURES & TRAIN MODELS
# ---------------------------------------------------------

feature_cols = [
    "hist_gf_year",
    "hist_ga_year",
    "hist_goal_diff",
    "hist_goals_per_match",
    "hist_goals_conceded_per_match",
    "hist_win_rate",
    "hist_points_year",
    "hist_points_per_match",
    "hist_matches_year",
    "fifa_rank",
    "fifa_points",
]

X = train_df[feature_cols]
y = train_df["label"]

print("Label counts (0=others,1=champ,2=runner-up,3=third):")
print(y.value_counts(), "\n")

# Random Forest
rf = RandomForestClassifier(
    n_estimators=RF_TREES, random_state=RANDOM_STATE, class_weight="balanced"
)
rf.fit(X, y)
y_pred_rf = rf.predict(X)
print("=== Random Forest – Training Performance ===")
print(classification_report(y, y_pred_rf))

# Logistic Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

logreg = LogisticRegression(
    multi_class="multinomial",
    max_iter=1000,
    solver="lbfgs",
    class_weight="balanced",
)
logreg.fit(X_scaled, y)
y_pred_lr = logreg.predict(X_scaled)
print("\n=== Logistic Regression – Training Performance ===")
print(classification_report(y, y_pred_lr))

# ---------------------------------------------------------
# 7. BUILD 2026 FEATURE ROWS FROM HISTORY
# ---------------------------------------------------------

print("\nBuilding features for 2026 teams from historical data...")

# latest year present in matches (should be 2022)
max_year = yearly_perf["Year"].max()

# cumulative totals at last known year (history up to 2022)
latest_hist = (
    yearly_perf[yearly_perf["Year"] == max_year]
    .set_index("Team")
    [[
        "cum_gf_year",
        "cum_ga_year",
        "cum_wins_year",
        "cum_draws_year",
        "cum_losses_year",
        "cum_points_year",
        "cum_matches_year",
    ]]
)

# construct 2026 dataframe
rows_2026 = []
for team in TEAMS_2026:
    t = team.upper()
    if t in latest_hist.index:
        row = latest_hist.loc[t]
        gf_h = row["cum_gf_year"]
        ga_h = row["cum_ga_year"]
        wins_h = row["cum_wins_year"]
        draws_h = row["cum_draws_year"]
        losses_h = row["cum_losses_year"]
        pts_h = row["cum_points_year"]
        matches_h = row["cum_matches_year"]
    else:
        gf_h = ga_h = wins_h = draws_h = losses_h = pts_h = matches_h = 0

    goal_diff_h = gf_h - ga_h
    goals_per_match_h = gf_h / matches_h if matches_h > 0 else 0.0
    goals_conceded_per_match_h = ga_h / matches_h if matches_h > 0 else 0.0
    win_rate_h = wins_h / matches_h if matches_h > 0 else 0.0
    points_per_match_h = pts_h / matches_h if matches_h > 0 else 0.0

    rows_2026.append(
        {
            "Team": t,
            "hist_gf_year": gf_h,
            "hist_ga_year": ga_h,
            "hist_goal_diff": goal_diff_h,
            "hist_goals_per_match": goals_per_match_h,
            "hist_goals_conceded_per_match": goals_conceded_per_match_h,
            "hist_win_rate": win_rate_h,
            "hist_points_year": pts_h,
            "hist_points_per_match": points_per_match_h,
            "hist_matches_year": matches_h,
        }
    )

features_2026 = pd.DataFrame(rows_2026)

# add ranking strength
features_2026 = features_2026.merge(ranking_2022, on="Team", how="left")
features_2026["fifa_rank"] = features_2026["fifa_rank"].fillna(median_rank)
features_2026["fifa_points"] = features_2026["fifa_points"].fillna(median_points)

print("Features for 2026 teams:", features_2026.shape, "\n")

# ---------------------------------------------------------
# 8. PREDICT 2026 CHAMPION PROBABILITIES
# ---------------------------------------------------------

X_2026 = features_2026[feature_cols]
X_2026_scaled = scaler.transform(X_2026)

probs_rf = rf.predict_proba(X_2026)
probs_lr = logreg.predict_proba(X_2026_scaled)

champ_idx_rf = list(rf.classes_).index(1)
champ_idx_lr = list(logreg.classes_).index(1)

features_2026["prob_champion_rf"] = probs_rf[:, champ_idx_rf]
features_2026["prob_champion_lr"] = probs_lr[:, champ_idx_lr]

# Normalize probabilities so they sum to 1 (makes them easier to interpret)
features_2026["prob_champion_rf_norm"] = (
    features_2026["prob_champion_rf"] / features_2026["prob_champion_rf"].sum()
)
features_2026["prob_champion_lr_norm"] = (
    features_2026["prob_champion_lr"] / features_2026["prob_champion_lr"].sum()
)

# simple ensemble: average normalized probs from RF + LR
features_2026["prob_champion_ensemble"] = (
    features_2026["prob_champion_rf_norm"] +
    features_2026["prob_champion_lr_norm"]
) / 2.0

# ---------------------------------------------------------
# SAVE 2026 PREDICTIONS TO CSV FOR VISUALIZATION
# ---------------------------------------------------------
pred_cols = [
    "Team",
    "prob_champion_rf_norm",
    "prob_champion_lr_norm",
    "prob_champion_ensemble",
]

features_2026[pred_cols].to_csv("fifa2026_predictions.csv", index=False)
print("Saved 2026 predictions to fifa2026_predictions.csv\n")


# ---------------------------------------------------------
# 9. SHOW RANKINGS & TOP 3
# ---------------------------------------------------------

rf_sorted = features_2026[["Team", "prob_champion_rf_norm"]].sort_values(
    "prob_champion_rf_norm", ascending=False
).reset_index(drop=True)

lr_sorted = features_2026[["Team", "prob_champion_lr_norm"]].sort_values(
    "prob_champion_lr_norm", ascending=False
).reset_index(drop=True)

ens_sorted = features_2026[["Team", "prob_champion_ensemble"]].sort_values(
    "prob_champion_ensemble", ascending=False
).reset_index(drop=True)

print("=== FIFA 2026 – Random Forest Normalized Champion Probabilities ===")
print(rf_sorted.head(15), "\n")

print("=== FIFA 2026 – Logistic Regression Normalized Champion Probabilities ===")
print(lr_sorted.head(15), "\n")

print("=== FIFA 2026 – Ensemble (RF + LR) Champion Probabilities ===")
print(ens_sorted.head(15), "\n")

# Top 3 from ensemble
print("🏆 ENSEMBLE MODEL — TOP 3 PREDICTIONS (FIFA 2026)")
print(f"🥇 1st (Champion): {ens_sorted.loc[0, 'Team']}  ({ens_sorted.loc[0, 'prob_champion_ensemble']:.3f})")
print(f"🥈 2nd:           {ens_sorted.loc[1, 'Team']}  ({ens_sorted.loc[1, 'prob_champion_ensemble']:.3f})")
print(f"🥉 3rd:           {ens_sorted.loc[2, 'Team']}  ({ens_sorted.loc[2, 'prob_champion_ensemble']:.3f})")

print("\nDone.")
