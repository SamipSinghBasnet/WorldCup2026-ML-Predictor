# WorldCup2026-ML-Predictor
 FIFA World Cup 2026 Winner Prediction Using Machine Learning & Monte Carlo Simulation
Author

Samip Singh Basnet
Computer Science | Machine Learning & Data Analytics

1. Project Overview

This project develops a data-driven predictive system to forecast the top contenders (1st, 2nd, 3rd) for the FIFA World Cup 2026 using historical football data, advanced machine learning models, and Monte Carlo tournament simulations.

The system integrates:

Historical World Cup performance (1930–2022)

FIFA rankings and Elo ratings

Group difficulty effects

Ensemble ML models

Probabilistic tournament simulation

The goal is not only to predict outcomes, but to explain why certain teams emerge as favorites.

2. Datasets Used
Dataset	Description
matches_1930_2022.csv	All FIFA World Cup matches from 1930–2022
world_cup.csv	Tournament-level metadata (champions, runners-up, hosts)
fifa_ranking_2022-10-06.csv	Pre-Qatar FIFA rankings and points
Fifa_world_cup_matches.csv	Detailed match statistics (possession, shots, fouls, passes)
Elo_ranking.csv	Elo ratings representing long-term team strength
Group_matches.csv	FIFA 2026 group stage allocation
3. Feature Engineering

For each national team, the following historical performance features were constructed:

Historical goals per match

Goal difference per match

Win rate

Points per match

Goals conceded per match

Matches played per year

Goals for / against per year

FIFA rank and FIFA points

Elo rating

Custom strength index

Group difficulty score (normalized)

These features capture both attacking and defensive consistency, as well as tournament experience.

4. Label Construction

Teams were labeled based on historical World Cup outcomes:

Label	Meaning
0	Did not reach semifinals
1	Third place
2	Runner-up
3	Champion

This enabled multi-class classification instead of simple win/loss prediction.

5. Models Implemented
Machine Learning Models

Random Forest Classifier

Logistic Regression (Multinomial)

XGBoost Classifier

Why multiple models?

Each model captures different relationships

Ensemble predictions reduce variance and bias

Improves robustness and interpretability

6. Model Performance (Training Summary)
Model	Accuracy	Notes
Random Forest	~93%	Strong performance with interpretable features
Logistic Regression	~57%	Baseline linear model
XGBoost	~100%	High capacity model (carefully regularized)

Note: Final predictions are not based on training accuracy alone but on ensemble probability outputs combined with simulation.

7. Feature Importance Analysis

A Random Forest feature importance analysis revealed:

Most influential features:

Historical goals per match

Goal difference

Win rate

Points per match

FIFA points and ranking

This confirms that long-term consistency and scoring efficiency are the strongest predictors of World Cup success.

8. Ensemble Prediction Strategy

Final ML probability was computed as a weighted ensemble of:

Random Forest probabilities

Logistic Regression probabilities

XGBoost probabilities

This ensemble was further adjusted using:

Team strength index

Group difficulty normalization

9. Monte Carlo Tournament Simulation

To model real tournament uncertainty:

2,000+ tournament simulations were run

Group stage → Knockout rounds → Champion

Match outcomes sampled probabilistically using team strength

Accounts for bracket randomness and matchup effects

This produces a realistic tournament-based win probability.

10. Key Visualizations

The project includes multiple explanatory visualizations:

Random Forest Feature Importance

Top 15 Teams by ML Champion Probability

Top 10 Teams by Monte Carlo Win Probability

Team Strength vs Group Difficulty Scatter Plot

Correlation Heatmap of Features and Probabilities

These visuals ensure model transparency and interpretability, critical for real-world ML applications.

11. Final Predictions (FIFA World Cup 2026)
Machine Learning Ensemble (Top 3)

Argentina — 0.216

Spain — 0.164

Brazil — 0.081

Monte Carlo Simulation (Top 3)

Brazil — 19.0%

Argentina — 17.6%

Spain — 17.2%

Both approaches consistently identify Argentina, Spain, and Brazil as the strongest contenders.

12. Key Takeaways

Historical consistency matters more than short-term form

Strong teams in easier groups gain significant advantage

Monte Carlo simulation complements ML by modeling uncertainty

Ensemble modeling improves reliability and robustness

13. Technologies Used

Python

Pandas, NumPy

Scikit-learn

XGBoost

Matplotlib & Seaborn

Git & GitHub

14. Future Improvements

Time-aware train/test split (pre-2018 → post-2018)

Player-level features (injuries, squad depth)

Match-level xG modeling

Bayesian uncertainty estimation


