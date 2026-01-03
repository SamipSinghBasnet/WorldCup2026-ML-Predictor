import pandas as pd
import numpy as np
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

RANDOM_STATE = 42

# ============================================================
# Utility Helpers
# ============================================================

def safe_div(num, den):
    return np.where(den > 0, num / den, 0.0)

def normalize_series(s):
    s_min, s_max = s.min(), s.max()
    if s_max > s_min:
        return (s - s_min) / (s_max - s_min)
    return pd.Series(0.0, index=s.index)

# ============================================================
# 1. Load Data
# ============================================================

def load_data():
    matches = pd.read_csv("matches_1930_2022.csv")
    world_cup = pd.read_csv("world_cup.csv")
    ranking = pd.read_csv("fifa_ranking_2022-10-06.csv")
    groups = pd.read_csv("Group_matches.csv")
    elo_df = pd.read_csv("Elo_ranking.csv")

    matches["home_team"] = matches["home_team"].str.upper()
    matches["away_team"] = matches["away_team"].str.upper()

    world_cup["Champion"] = world_cup["Champion"].str.upper()
    world_cup["Runner-Up"] = world_cup["Runner-Up"].str.upper()

    ranking["team"] = ranking["team"].str.upper()
    ranking["team_code"] = ranking["team_code"].str.upper()

    groups["Team"] = groups["Team"].str.upper()
    groups["Group"] = groups["Group"].str.upper()

    # Elo cleaning
    elo_df.columns = elo_df.columns.str.strip()
    elo_df["Team"] = elo_df["Team"].str.upper()
    elo_df["Elo"] = pd.to_numeric(elo_df["Elo"], errors="coerce")

    return matches, world_cup, ranking, groups, elo_df

# ============================================================
# 2. Build Yearly Performance Features
# ============================================================

def build_yearly_performance(matches):
    def outcome(gf, ga):
        if gf > ga: return 1,0,0,3
        if gf == ga: return 0,1,0,1
        return 0,0,1,0

    rows = []
    for _, r in matches.iterrows():
        ht, at = r["home_team"], r["away_team"]
        hs, as_ = r["home_score"], r["away_score"]
        y = r["Year"]

        # home entry
        w,d,l,p = outcome(hs, as_)
        rows.append({"Year":y,"Team":ht,"gf":hs,"ga":as_,"win":w,"draw":d,"loss":l,"points":p})

        # away entry
        w,d,l,p = outcome(as_, hs)
        rows.append({"Year":y,"Team":at,"gf":as_,"ga":hs,"win":w,"draw":d,"loss":l,"points":p})

    df = pd.DataFrame(rows)

    yearly = df.groupby(["Team","Year"]).agg(
        gf_year=("gf","sum"),
        ga_year=("ga","sum"),
        wins_year=("win","sum"),
        draws_year=("draw","sum"),
        losses_year=("loss","sum"),
        points_year=("points","sum"),
        matches_year=("gf","count")
    ).reset_index()

    yearly = yearly.sort_values(["Team","Year"])

    # rolling sums
    for c in ["gf_year","ga_year","wins_year","draws_year","losses_year","points_year","matches_year"]:
        yearly[f"cum_{c}"] = yearly.groupby("Team")[c].cumsum()
        yearly[f"hist_{c}"] = yearly.groupby("Team")[c].cumsum().shift(1).fillna(0)

    # Numerics
    yearly["hist_goal_diff"] = yearly["hist_gf_year"] - yearly["hist_ga_year"]
    yearly["hist_goals_per_match"] = safe_div(yearly["hist_gf_year"], yearly["hist_matches_year"])
    yearly["hist_goals_conceded_per_match"] = safe_div(yearly["hist_ga_year"], yearly["hist_matches_year"])
    yearly["hist_win_rate"] = safe_div(yearly["hist_wins_year"], yearly["hist_matches_year"])
    yearly["hist_points_per_match"] = safe_div(yearly["hist_points_year"], yearly["hist_matches_year"])

    return yearly

# ============================================================
# 3. Build Labels from World Cup
# ============================================================

def build_labels(yearly, wc, matches):
    label_df = yearly[["Team","Year"]].copy()
    label_df["label"] = 0

    # champions + runner-up
    for _, r in wc.iterrows():
        y, ch, ru = r["Year"], r["Champion"], r["Runner-Up"]
        label_df.loc[(label_df["Year"]==y)&(label_df["Team"]==ch),"label"]=1
        label_df.loc[(label_df["Year"]==y)&(label_df["Team"]==ru),"label"]=2

    # third place via matches
    third_rows=[]
    for y in sorted(matches["Year"].unique()):
        m = matches[(matches["Year"]==y)&(matches["Round"].str.contains("Third",case=False,na=False))]
        if not m.empty:
            r = m.iloc[0]
            t3 = r["home_team"] if r["home_score"]>r["away_score"] else r["away_team"]
            third_rows.append({"Year":y,"Team":t3})

    third_df = pd.DataFrame(third_rows)
    third_df["Team"] = third_df["Team"].str.upper()

    for _, r in third_df.iterrows():
        label_df.loc[(label_df["Year"]==r["Year"])&(label_df["Team"]==r["Team"]),"label"]=3

    return label_df

# ============================================================
# 4. Merge FIFA Ranking into Training Data
# ============================================================

def merge_ranking(train_df, ranking):
    r22 = ranking.rename(columns={"team":"Team","rank":"fifa_rank","points":"fifa_points"})
    r22 = r22[["Team","team_code","fifa_rank","fifa_points"]]

    train_df = train_df.merge(r22[["Team","fifa_rank","fifa_points"]],on="Team",how="left")

    med_rank = r22["fifa_rank"].median()
    med_pts = r22["fifa_points"].median()

    train_df["fifa_rank"]=train_df["fifa_rank"].fillna(med_rank)
    train_df["fifa_points"]=train_df["fifa_points"].fillna(med_pts)

    return train_df, r22, med_rank, med_pts

# ============================================================
# 5. Train Models: RF + LR + XGB
# ============================================================

def train_models(train_df):
    feature_cols=[
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

    print("Training label distribution:")
    print(y.value_counts(),"\n")

    # RF
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    rf.fit(X,y)
    print("\n=== Random Forest ===\n",classification_report(y,rf.predict(X)))

    # LR
    scaler=StandardScaler()
    Xs = scaler.fit_transform(X)

    lr = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial",
        class_weight="balanced"
    )
    lr.fit(Xs,y)
    print("\n=== Logistic Regression ===\n",classification_report(y,lr.predict(Xs)))

    # XGB
    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE
    )
    xgb.fit(X,y)
    print("\n=== XGBoost ===\n",classification_report(y,xgb.predict(X)))

    return rf, lr, xgb, scaler, feature_cols

# ============================================================
# 6. Build 2026 Feature Matrix (WC History + FIFA + Elo + Groups)
# ============================================================

def build_2026_features(yearly, rank22, med_rank, med_pts, groups, elo):
    teams_2026 = sorted(groups["Team"].unique())
    print(f"2026 teams loaded: {len(teams_2026)}")

    max_year = yearly["Year"].max()
    hist_latest = yearly[yearly["Year"]==max_year].set_index("Team")

    code_map = dict(zip(rank22["team_code"], rank22["Team"]))

    def fetch_real_team_data(team):
        if team in hist_latest.index:
            row=hist_latest.loc[team]
            gf,ga,w,d,l,p,m = row[[
                "cum_gf_year","cum_ga_year","cum_wins_year",
                "cum_draws_year","cum_losses_year","cum_points_year",
                "cum_matches_year"]]
        else:
            gf=ga=w=d=l=p=m=0.0

        r=rank22[rank22["Team"]==team]
        if not r.empty:
            fr=float(r.iloc[0]["fifa_rank"])
            fp=float(r.iloc[0]["fifa_points"])
        else:
            fr=float(med_rank)
            fp=float(med_pts)
        return gf,ga,w,d,l,p,m,fr,fp

    def fetch_placeholder(team):
        m=re.search(r"FIFA\s*\((.*?)\)",team)
        if not m:
            return 0,0,0,0,0,0,0,float(med_rank),float(med_pts)

        codes=[c.strip().upper() for c in m.group(1).split("/")]

        agg={"gf":0,"ga":0,"w":0,"d":0,"l":0,"p":0,"m":0,"fr":0,"fp":0}
        cnt=0
        for c in codes:
            if c not in code_map: 
                continue
            rt = code_map[c]
            gf,ga,w,d,l,p,mm,fr,fp = fetch_real_team_data(rt)
            agg["gf"]+=gf; agg["ga"]+=ga; agg["w"]+=w
            agg["d"]+=d; agg["l"]+=l; agg["p"]+=p
            agg["m"]+=mm; agg["fr"]+=fr; agg["fp"]+=fp
            cnt+=1

        if cnt==0:
            return 0,0,0,0,0,0,0,float(med_rank),float(med_pts)

        return (agg["gf"]/cnt, agg["ga"]/cnt, agg["w"]/cnt, agg["d"]/cnt,
                agg["l"]/cnt, agg["p"]/cnt, agg["m"]/cnt,
                agg["fr"]/cnt, agg["fp"]/cnt)

    rows=[]
    for t in teams_2026:
        t_up = t.upper()
        if t_up.startswith("FIFA(") or "FIFA (" in t_up:
            gf,ga,w,d,l,p,m,fr,fp = fetch_placeholder(t_up)
        else:
            gf,ga,w,d,l,p,m,fr,fp = fetch_real_team_data(t_up)

        goal_diff=gf-ga
        gf_pm = gf/m if m>0 else 0
        ga_pm = ga/m if m>0 else 0
        win_rate = w/m if m>0 else 0
        pts_pm = p/m if m>0 else 0

        rows.append({
            "Team":t_up,
            "hist_gf_year":gf,
            "hist_ga_year":ga,
            "hist_goal_diff":goal_diff,
            "hist_goals_per_match":gf_pm,
            "hist_goals_conceded_per_match":ga_pm,
            "hist_win_rate":win_rate,
            "hist_points_year":p,
            "hist_points_per_match":pts_pm,
            "hist_matches_year":m,
            "fifa_rank":fr,
            "fifa_points":fp
        })

    df=pd.DataFrame(rows)

    elo_med=elo["Elo"].median()
    df=df.merge(elo[["Team","Elo"]],on="Team",how="left")
    df["Elo"]=df["Elo"].fillna(elo_med)

    df=df.merge(groups[["Group","Team"]],on="Team",how="left")

    return df

# ============================================================
# 7. Strength Index + Group Difficulty
# ============================================================

def compute_strength(features, rank22):
    max_rank = rank22["fifa_rank"].max()
    features["rank_strength"] = (max_rank+1) - features["fifa_rank"]
    features["momentum_raw"] = features["hist_points_per_match"] * features["fifa_points"]

    # Normalize
    for col,new in [
        ("Elo","elo_norm"),
        ("fifa_points","fifa_points_norm"),
        ("hist_goal_diff","hist_goal_diff_norm"),
        ("hist_win_rate","hist_win_rate_norm"),
        ("hist_matches_year","hist_matches_norm"),
        ("rank_strength","rank_strength_norm"),
        ("momentum_raw","momentum_norm"),
    ]:
        features[new] = normalize_series(features[col])

    # Composite strength
    features["strength_index"] = (
        0.45*features["elo_norm"] +
        0.20*features["fifa_points_norm"] +
        0.15*features["hist_goal_diff_norm"] +
        0.10*features["hist_win_rate_norm"] +
        0.05*features["rank_strength_norm"] +
        0.05*features["momentum_norm"]
    )

    # Group difficulty
    gd=[]
    for _,r in features.iterrows():
        g=r["Group"]
        t=r["Team"]
        group = features[features["Group"]==g]
        opp = group[group["Team"]!=t]
        gd.append(opp["strength_index"].mean() if not opp.empty else features["strength_index"].mean())

    features["group_difficulty"]=gd
    features["group_difficulty_norm"]=normalize_series(features["group_difficulty"])

    return features

# ============================================================
# 8. Predict 2026 (RF + LR + XGB + Group Adjustment)
# ============================================================

def predict_2026(features, rf, lr, xgb, scaler, feature_cols):

    X=features[feature_cols]
    Xs=scaler.transform(X)

    prf=rf.predict_proba(X)
    plr=lr.predict_proba(Xs)
    pxgb=xgb.predict_proba(X)

    ci_rf=list(rf.classes_).index(1)
    ci_lr=list(lr.classes_).index(1)
    ci_xgb=list(xgb.classes_).index(1)

    features["prob_champion_rf"]=prf[:,ci_rf]
    features["prob_champion_lr"]=plr[:,ci_lr]
    features["prob_champion_xgb"]=pxgb[:,ci_xgb]

    for model in ["rf","lr","xgb"]:
        col=f"prob_champion_{model}"
        features[col+"_norm"]=features[col]/features[col].sum()

    features["prob_champion_ensemble"]=(
        features["prob_champion_rf_norm"]+
        features["prob_champion_lr_norm"]+
        features["prob_champion_xgb_norm"]
    )/3

    # group adjustment
    gd=features["group_difficulty_norm"]
    std=gd.std() if gd.std()>0 else 1
    gd_z=(gd-gd.mean())/std
    beta=0.25

    adj=1-beta*gd_z
    adj=adj.clip(lower=0.1)

    features["prob_champion_final_raw"]=features["prob_champion_ensemble"]*adj
    features["prob_champion_final"]=(
        features["prob_champion_final_raw"]/features["prob_champion_final_raw"].sum()
    )

    return features

# ============================================================
# 9. Monte Carlo World Cup Simulation (48 → 32 Knockout)
# ============================================================

def simulate_match(team_a, team_b, strength_map, draw_prob=0.25, k=4.0):
    sa=strength_map[team_a]
    sb=strength_map[team_b]
    diff=sa-sb

    p_a_no_draw=1/(1+np.exp(-k*diff))
    p_b_no_draw=1-p_a_no_draw

    p_d=draw_prob
    p_a=p_a_no_draw*(1-p_d)
    p_b=p_b_no_draw*(1-p_d)

    return np.random.choice(["A","D","B"],p=[p_a,p_d,p_b])

def simulate_group_table(teams, strength_map):
    table={t:{"pts":0,"gf":0,"ga":0} for t in teams}

    for i in range(len(teams)):
        for j in range(i+1,len(teams)):
            a,b = teams[i], teams[j]
            res=simulate_match(a,b,strength_map)

            sa=strength_map[a]
            sb=strength_map[b]
            lam_a=1.4*(sa/(sa+sb))
            lam_b=1.4*(sb/(sa+sb))

            gfa=np.random.poisson(lam_a)
            gfb=np.random.poisson(lam_b)

            table[a]["gf"]+=gfa; table[a]["ga"]+=gfb
            table[b]["gf"]+=gfb; table[b]["ga"]+=gfa

            if res=="A": table[a]["pts"]+=3
            elif res=="B": table[b]["pts"]+=3
            else: table[a]["pts"]+=1; table[b]["pts"]+=1

    df=pd.DataFrame(
        [(t,v["pts"],v["gf"],v["ga"]) for t,v in table.items()],
        columns=["Team","pts","gf","ga"]
    )
    df["gd"]=df["gf"]-df["ga"]
    df=df.sort_values(["pts","gd","gf"],ascending=[False,False,False]).reset_index(drop=True)
    return df

def simulate_knockout(teams, strength_map):
    teams=list(teams)
    np.random.shuffle(teams)
    if len(teams)%2==1:
        # drop weakest if odd
        weakest=min(teams,key=lambda t:strength_map[t])
        teams.remove(weakest)

    while len(teams)>1:
        nxt=[]
        for i in range(0,len(teams),2):
            a,b = teams[i], teams[i+1]
            while True:
                res=simulate_match(a,b,strength_map,draw_prob=0.0)
                if res=="A": nxt.append(a); break
                if res=="B": nxt.append(b); break
        teams=nxt

    return teams[0]

def monte_carlo(features, n_sims=2000, seed=42):
    np.random.seed(seed)
    strength_map=features.set_index("Team")["strength_index"].to_dict()

    groups=sorted(features["Group"].unique())
    team_groups={g:list(features[features["Group"]==g]["Team"]) for g in groups}

    win_count={t:0 for t in features["Team"]}

    for _ in range(n_sims):
        top2=[]
        third_candidates=[]

        # group stage
        for g in groups:
            teams=team_groups[g]
            tab=simulate_group_table(teams,strength_map)
            if len(tab)>=2:
                top2.extend(tab.iloc[:2]["Team"].tolist())
            if len(tab)>=3:
                t3=tab.iloc[2]
                third_candidates.append({
                    "Team":t3["Team"],
                    "pts":t3["pts"],
                    "gd":t3["gd"],
                    "gf":t3["gf"],
                    "strength":strength_map[t3["Team"]]
                })

        third_df=pd.DataFrame(third_candidates)
        if not third_df.empty:
            third_df=third_df.sort_values(
                ["pts","gd","gf","strength"],
                ascending=[False,False,False,False]
            )
            best8=third_df.head(8)["Team"].tolist()
        else:
            best8=[]

        ko=top2+best8
        if len(ko)<2:
            continue

        champ=simulate_knockout(ko,strength_map)
        win_count[champ]+=1

    total=float(n_sims)
    mc=pd.DataFrame(
        [(t,win_count[t]/total) for t in win_count],
        columns=["Team","mc_win_prob"]
    ).sort_values("mc_win_prob",ascending=False)

    return mc

# ============================================================
# 10. MAIN PIPELINE
# ============================================================

def main():
    print("Loading data...")
    matches, wc, ranking, groups, elo = load_data()

    print("Computing yearly performance...")
    yearly = build_yearly_performance(matches)

    print("Building labels...")
    labels = build_labels(yearly, wc, matches)
    train_df = yearly.merge(labels, on=["Team","Year"], how="inner")

    print("Merging ranking...")
    train_df, rank22, med_rank, med_pts = merge_ranking(train_df, ranking)

    print("Training models...")
    rf, lr, xgb, scaler, feats = train_models(train_df)

    print("Building 2026 feature set...")
    f2026 = build_2026_features(yearly, rank22, med_rank, med_pts, groups, elo)

    print("Computing strength & group difficulty...")
    f2026 = compute_strength(f2026, rank22)

    print("Predicting ML probabilities...")
    f2026 = predict_2026(f2026, rf, lr, xgb, scaler, feats)

    print("Running Monte Carlo simulation...")
    mc = monte_carlo(f2026, n_sims=2000, seed=42)
    f2026 = f2026.merge(mc, on="Team", how="left")

    # Save
    cols=[
        "Team","Group","Elo","strength_index","group_difficulty_norm",
        "prob_champion_rf_norm","prob_champion_lr_norm","prob_champion_xgb_norm",
        "prob_champion_ensemble","prob_champion_final","mc_win_prob"
    ]
    f2026[cols].to_csv("fifa2026_with_groups_predictions.csv", index=False)
    print("\nSaved fifa2026_with_groups_predictions.csv")

    # Print results
    top=f2026.sort_values("prob_champion_final",ascending=False)[["Team","Group","prob_champion_final"]].head(15)
    print("\n=== TOP 15 ML Predictions ===")
    print(top.to_string(index=False))

    topmc=mc.head(10)
    print("\n=== TOP 10 Monte Carlo Predictions ===")
    print(topmc.to_string(index=False))

    print("\n FINAL TOP 3 (ML Probability)")
    print(f"1️ {top.iloc[0]['Team']} ({top.iloc[0]['prob_champion_final']:.3f})")
    print(f"2️ {top.iloc[1]['Team']} ({top.iloc[1]['prob_champion_final']:.3f})")
    print(f"3️ {top.iloc[2]['Team']} ({top.iloc[2]['prob_champion_final']:.3f})")

    print("\nDone.")

if __name__ == "__main__":
    main()
