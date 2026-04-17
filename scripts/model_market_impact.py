"""
model_market_impact.py
======================
Model 1: Do Trump's tweets move markets?

Uses market_dataset.csv (tweets with same-day Dow/S&P/NASDAQ data).

Tasks:
  A. Classification — does a tweet cause market to go UP or DOWN?
  B. Regression     — by how much?
  C. Event study    — do economy/trade tweets have larger impact?

Features:
  Tweet: sentiment, caps ratio, exclamations, length, econ/trade/media keywords
  News:  prior-day sentiment, economy intensity, Trump coverage

Models: Logistic Regression, Random Forest, XGBoost

Run:
  py -3 model_market_impact.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, mean_squared_error, r2_score)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # no display needed
import seaborn as sns

MARKET_CSV = "market_dataset.csv"
FIGURES    = "figures"

import os
os.makedirs(FIGURES, exist_ok=True)

# ── Features ─────────────────────────────────────────────────────────────────
TWEET_FEATURES = [
    "tweet_sent", "tweet_sent_pos", "tweet_sent_neg",
    "tweet_caps_ratio", "tweet_exclamations", "tweet_questions",
    "tweet_length", "tweet_word_count",
    "tweet_econ", "tweet_politics", "tweet_media",
    "tweet_china", "tweet_russia",
    "tweet_platform",
]

NEWS_FEATURES = [
    "news_count",
    "news_sent_mean", "news_sent_std", "news_sent_neg_pct",
    "news_econ_intensity", "news_politics_intensity",
    "news_china_pct", "news_russia_pct", "news_trump_pct",
]

ALL_FEATURES = TWEET_FEATURES + NEWS_FEATURES

TARGETS = {
    "dow":    "dow_pct",
    "sp500":  "sp500_pct",
    "nasdaq": "nasdaq_pct",
}


def load_data():
    df = pd.read_csv(MARKET_CSV)
    df = df.dropna(subset=["dow_pct", "sp500_pct", "nasdaq_pct"])

    # Direction labels
    df["dow_up"]    = (df["dow_pct"]    > 0).astype(int)
    df["sp500_up"]  = (df["sp500_pct"]  > 0).astype(int)
    df["nasdaq_up"] = (df["nasdaq_pct"] > 0).astype(int)

    # Drop rows with missing features
    df = df.dropna(subset=ALL_FEATURES)
    print(f"Market dataset: {len(df):,} tweets with full market + feature data")
    print(f"  Dow up: {df['dow_up'].mean():.1%}  |  down: {1-df['dow_up'].mean():.1%}")
    return df


# ── A. Classification: UP vs DOWN ────────────────────────────────────────────
def run_classification(df):
    print("\n" + "="*65)
    print("A. CLASSIFICATION — Does tweet predict market direction?")
    print("="*65)

    X = df[ALL_FEATURES].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    for market_name, col in [("Dow", "dow_up"), ("S&P500", "sp500_up"), ("NASDAQ", "nasdaq_up")]:
        y = df[col].values
        print(f"\n  Target: {market_name} direction")

        models = {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42))
            ]),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                random_state=42, eval_metric="logloss",
                verbosity=0
            ),
        }

        for name, model in models.items():
            aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
            accs = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            print(f"    {name:<25}  AUC={aucs.mean():.3f} ±{aucs.std():.3f}  "
                  f"Acc={accs.mean():.3f} ±{accs.std():.3f}")
            results[f"{market_name}_{name}_auc"] = aucs.mean()

    return results


# ── B. Regression: predict % change ─────────────────────────────────────────
def run_regression(df):
    print("\n" + "="*65)
    print("B. REGRESSION — Predict magnitude of market move")
    print("="*65)

    X = df[ALL_FEATURES].values
    from sklearn.model_selection import cross_val_score, KFold
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for market_name, col in [("Dow", "dow_pct"), ("S&P500", "sp500_pct"), ("NASDAQ", "nasdaq_pct")]:
        y = df[col].values
        print(f"\n  Target: {market_name} % change")

        models = {
            "Ridge Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0))
            ]),
            "Random Forest": RandomForestRegressor(
                n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
            ),
            "XGBoost": xgb.XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                random_state=42, verbosity=0
            ),
        }

        for name, model in models.items():
            r2s  = cross_val_score(model, X, y, cv=cv, scoring="r2")
            mses = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
            rmse = np.sqrt(-mses.mean())
            print(f"    {name:<25}  R²={r2s.mean():.4f} ±{r2s.std():.4f}  RMSE={rmse:.4f}")


# ── C. Feature importance ─────────────────────────────────────────────────────
def run_feature_importance(df):
    print("\n" + "="*65)
    print("C. FEATURE IMPORTANCE — What tweet/news features matter most?")
    print("="*65)

    X = df[ALL_FEATURES].values
    y = df["dow_up"].values

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        random_state=42, eval_metric="logloss", verbosity=0
    )
    model.fit(X, y)

    imp = pd.Series(model.feature_importances_, index=ALL_FEATURES)
    imp = imp.sort_values(ascending=False)

    print("\n  Top 15 features (XGBoost, predicting Dow direction):")
    for feat, score in imp.head(15).items():
        bar = "█" * int(score * 200)
        print(f"    {feat:<35} {score:.4f}  {bar}")

    # Save figure
    fig, ax = plt.subplots(figsize=(10, 7))
    imp.head(15).sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importance: Predicting Dow Direction from Trump Tweets + News",
                 fontsize=12)
    ax.set_xlabel("XGBoost Importance Score")
    plt.tight_layout()
    fig.savefig(f"{FIGURES}/feature_importance_market.png", dpi=150)
    plt.close()
    print(f"\n  Saved: {FIGURES}/feature_importance_market.png")

    return imp


# ── D. Event study: economy/trade tweets ─────────────────────────────────────
def run_event_study(df):
    print("\n" + "="*65)
    print("D. EVENT STUDY — Economy vs non-economy tweets")
    print("="*65)

    econ   = df[df["tweet_econ"] > 0.01]
    non    = df[df["tweet_econ"] <= 0.01]
    trade  = df[df["tweet_china"] == 1]

    groups = [
        ("Economy tweets",     econ),
        ("Non-economy tweets", non),
        ("China/trade tweets", trade),
        ("All tweets",         df),
    ]

    print(f"\n  {'Group':<25} {'N':>6}  {'Dow mean%':>10}  {'Dow std%':>9}  {'|Dow|>0.5%':>10}")
    for name, g in groups:
        if len(g) == 0:
            continue
        d = g["dow_pct"].dropna()
        big = (d.abs() > 0.5).mean()
        print(f"  {name:<25} {len(d):>6}  {d.mean():>10.4f}  {d.std():>9.4f}  {big:>10.1%}")

    # Plot distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (mkt, col) in zip(axes, [("Dow", "dow_pct"), ("S&P 500", "sp500_pct"), ("NASDAQ", "nasdaq_pct")]):
        econ[col].dropna().hist(ax=ax, bins=60, alpha=0.6, label="Economy tweets", color="crimson")
        non[col].dropna().hist( ax=ax, bins=60, alpha=0.6, label="Other tweets",   color="steelblue")
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"{mkt} % change on tweet days")
        ax.set_xlabel("% change")
        ax.legend(fontsize=8)
    plt.suptitle("Market Impact: Economy vs Non-Economy Trump Tweets", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{FIGURES}/event_study_market.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {FIGURES}/event_study_market.png")

    # T-test: do economy tweets cause bigger moves?
    from scipy import stats
    for mkt, col in [("Dow", "dow_pct"), ("S&P500", "sp500_pct"), ("NASDAQ", "nasdaq_pct")]:
        e = econ[col].dropna()
        n = non[col].dropna()
        t, p = stats.ttest_ind(e.abs(), n.abs())
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  |{mkt}| economy vs other: t={t:.3f}, p={p:.4f} {sig}")


# ── E. Timeline: tweet sentiment vs market ────────────────────────────────────
def run_timeline(df):
    print("\n" + "="*65)
    print("E. TIMELINE — Tweet sentiment vs Dow over time")
    print("="*65)

    df2 = df.copy()
    df2["tweet_date"] = pd.to_datetime(df2["tweet_date"], errors="coerce")
    df2 = df2.dropna(subset=["tweet_date", "dow_pct"])
    df2 = df2.set_index("tweet_date").sort_index()

    monthly = df2[["tweet_sent", "dow_pct"]].resample("ME").mean()

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()
    ax1.plot(monthly.index, monthly["tweet_sent"],  color="navy",   lw=1.5, label="Avg tweet sentiment")
    ax2.plot(monthly.index, monthly["dow_pct"],     color="crimson", lw=1.5, alpha=0.7, label="Avg Dow % change")
    ax1.set_ylabel("Tweet Sentiment (VADER)", color="navy")
    ax2.set_ylabel("Dow % Change", color="crimson")
    ax1.set_xlabel("Date")
    plt.title("Trump Tweet Sentiment vs Dow Jones % Change (monthly averages)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{FIGURES}/timeline_sentiment_vs_dow.png", dpi=150)
    plt.close()
    print(f"  Saved: {FIGURES}/timeline_sentiment_vs_dow.png")

    # Correlation
    corr = monthly[["tweet_sent","dow_pct"]].corr().iloc[0,1]
    print(f"  Pearson correlation (monthly): r = {corr:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("MODEL 1: TWEET → MARKET IMPACT")
    print("="*65)

    df = load_data()

    run_classification(df)
    run_regression(df)
    run_feature_importance(df)
    run_event_study(df)
    run_timeline(df)

    print("\n" + "="*65)
    print("Done. Figures saved to figures/")
    print("="*65)


if __name__ == "__main__":
    main()
