"""
model_news_to_tweet.py
======================
Model 2: Can news predict Trump's tweets?

Hypothesis: Trump's tweets are commentaries on the news cycle.
If true, the tone and topic of news from the prior 24h should
predict the tone and topic of his next tweet.

Tasks:
  A. Topic modeling (LDA) — discover latent topics in tweets and news
  B. Sentiment prediction — news features → tweet sentiment
  C. Topic prediction     — news topics → tweet topic
  D. Granger causality    — does news Granger-cause tweet sentiment?
  E. Correlation heatmap  — news signals vs tweet signals

Run:
  py -3 model_news_to_tweet.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
import xgboost as xgb

MASTER_CSV = "master_dataset.csv"
FIGURES    = "figures"
os.makedirs(FIGURES, exist_ok=True)

NEWS_FEATURES = [
    "news_sent_mean", "news_sent_std", "news_sent_neg_pct",
    "news_econ_intensity", "news_politics_intensity",
    "news_media_intensity",
    "news_china_pct", "news_russia_pct", "news_trump_pct",
    "news_count",
]


def load_data():
    df = pd.read_csv(MASTER_CSV)
    df = df.dropna(subset=NEWS_FEATURES)
    # Filter to years where we have good news data (NYT starts 2009)
    df["tweet_date"] = pd.to_datetime(df["tweet_date"], errors="coerce")
    df = df[df["tweet_date"] >= "2009-01-01"]
    print(f"Master dataset: {len(df):,} tweets with news context")
    print(f"  Date range: {df['tweet_date'].min().date()} → {df['tweet_date'].max().date()}")
    return df


# ── A. Topic modeling on tweets (LDA) ────────────────────────────────────────
def run_lda_tweets(df, n_topics=10):
    print("\n" + "="*65)
    print("A. TOPIC MODELING — Latent topics in Trump tweets (LDA)")
    print("="*65)

    texts = df["tweet_content"].fillna("").tolist()

    vec = CountVectorizer(
        max_features=3000,
        stop_words="english",
        min_df=5,
        ngram_range=(1,2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vec.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20,
        learning_method="batch",
    )
    lda.fit(X)

    terms = vec.get_feature_names_out()
    print(f"\n  {n_topics} latent topics discovered in Trump tweets:\n")
    topic_names = []
    for i, comp in enumerate(lda.components_):
        top_words = [terms[j] for j in comp.argsort()[-10:][::-1]]
        label = f"T{i+1}"
        topic_names.append(label)
        print(f"  Topic {i+1:>2}: {', '.join(top_words)}")

    # Assign dominant topic to each tweet
    doc_topics = lda.transform(X)
    df = df.copy()
    df["tweet_topic"] = doc_topics.argmax(axis=1)

    # Topic distribution
    topic_counts = df["tweet_topic"].value_counts().sort_index()
    print(f"\n  Topic distribution:")
    for t, n in topic_counts.items():
        bar = "█" * (n // 100)
        print(f"    Topic {t+1:>2}: {n:>5,}  {bar}")

    # Save topic distribution figure
    fig, ax = plt.subplots(figsize=(10, 5))
    topic_counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_xticklabels([f"T{i+1}" for i in topic_counts.index], rotation=0)
    ax.set_title("Distribution of Trump Tweet Topics (LDA)")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Number of Tweets")
    plt.tight_layout()
    fig.savefig(f"{FIGURES}/tweet_topic_distribution.png", dpi=150)
    plt.close()

    return df, lda, vec, doc_topics


# ── B. Sentiment prediction: news → tweet ────────────────────────────────────
def run_sentiment_prediction(df):
    print("\n" + "="*65)
    print("B. SENTIMENT PREDICTION — News features → Tweet sentiment")
    print("="*65)

    df2 = df.dropna(subset=NEWS_FEATURES + ["tweet_sent"])
    X = df2[NEWS_FEATURES].values
    cv5 = KFold(n_splits=5, shuffle=True, random_state=42)
    cv5s = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Predict tweet VADER compound score (regression)
    y_reg = df2["tweet_sent"].values
    print("\n  1. Predict tweet VADER compound score (regression):")
    for name, model in [
        ("Ridge",         Pipeline([("sc", StandardScaler()), ("m", Ridge())])),
        ("Random Forest", RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)),
        ("XGBoost",       xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, verbosity=0)),
    ]:
        r2s = cross_val_score(model, X, y_reg, cv=cv5, scoring="r2")
        print(f"    {name:<20}  R²={r2s.mean():.4f} ±{r2s.std():.4f}")

    # 2. Predict tweet sentiment direction (positive/negative)
    y_cls = df2["tweet_sent_pos"].values
    print(f"\n  2. Predict tweet positive sentiment (classification):")
    print(f"     Class balance: {y_cls.mean():.1%} positive")
    for name, model in [
        ("Logistic Reg",  Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=500, random_state=42))])),
        ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)),
        ("XGBoost",       xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss", verbosity=0)),
    ]:
        aucs = cross_val_score(model, X, y_cls, cv=cv5s, scoring="roc_auc")
        accs = cross_val_score(model, X, y_cls, cv=cv5s, scoring="accuracy")
        print(f"    {name:<20}  AUC={aucs.mean():.3f} ±{aucs.std():.3f}  Acc={accs.mean():.3f}")

    # 3. Predict tweet aggression (caps ratio > 0.15)
    y_agg = (df2["tweet_caps_ratio"] > 0.15).astype(int).values
    print(f"\n  3. Predict tweet aggression/caps (classification):")
    print(f"     Class balance: {y_agg.mean():.1%} aggressive")
    for name, model in [
        ("Logistic Reg",  Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=500, random_state=42))])),
        ("XGBoost",       xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss", verbosity=0)),
    ]:
        aucs = cross_val_score(model, X, y_agg, cv=cv5s, scoring="roc_auc")
        accs = cross_val_score(model, X, y_agg, cv=cv5s, scoring="accuracy")
        print(f"    {name:<20}  AUC={aucs.mean():.3f} ±{aucs.std():.3f}  Acc={accs.mean():.3f}")


# ── C. Topic prediction: news signals → tweet topic ──────────────────────────
def run_topic_prediction(df):
    print("\n" + "="*65)
    print("C. TOPIC PREDICTION — News signals → Tweet topic (LDA)")
    print("="*65)

    if "tweet_topic" not in df.columns:
        print("  [skip — run A first]")
        return

    df2 = df.dropna(subset=NEWS_FEATURES + ["tweet_topic"])
    X = df2[NEWS_FEATURES].values
    y = df2["tweet_topic"].values
    n_classes = len(np.unique(y))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"  {n_classes} topics, predicting from news features:")

    for name, model in [
        ("Logistic Reg",  Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=500, random_state=42))])),
        ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)),
        ("XGBoost",       xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="mlogloss", verbosity=0, num_class=n_classes)),
    ]:
        accs = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        base = 1.0 / n_classes
        lift = accs.mean() / base
        print(f"    {name:<20}  Acc={accs.mean():.3f} ±{accs.std():.3f}  (baseline={base:.3f}, lift={lift:.2f}x)")


# ── D. Granger causality: news sentiment → tweet sentiment ───────────────────
def run_granger(df):
    print("\n" + "="*65)
    print("D. GRANGER CAUSALITY — Does news sentiment predict tweet sentiment?")
    print("="*65)

    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        print("  [statsmodels not installed — skipping Granger test]")
        print("  Install: py -3 -m pip install statsmodels")
        return

    df2 = df.dropna(subset=["news_sent_mean", "tweet_sent", "tweet_date"])
    df2 = df2.sort_values("tweet_date")

    # Resample to daily
    df2 = df2.set_index("tweet_date")
    daily = df2[["news_sent_mean","tweet_sent"]].resample("D").mean().dropna()

    print(f"  Daily time series: {len(daily)} days")
    print("  H0: news_sent does NOT Granger-cause tweet_sent\n")

    for lag in [1, 3, 7]:
        res = grangercausalitytests(daily[["tweet_sent","news_sent_mean"]].values,
                                    maxlag=lag, verbose=False)
        p = res[lag][0]["ssr_ftest"][1]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  Lag {lag} day(s):  p={p:.4f}  {sig}")

    print("\n  *** p<0.001  ** p<0.01  * p<0.05  ns=not significant")


# ── E. Correlation heatmap ────────────────────────────────────────────────────
def run_correlation_heatmap(df):
    print("\n" + "="*65)
    print("E. CORRELATION HEATMAP — News signals vs Tweet signals")
    print("="*65)

    cols_news  = ["news_sent_mean","news_econ_intensity","news_politics_intensity",
                  "news_media_intensity","news_china_pct","news_russia_pct","news_trump_pct"]
    cols_tweet = ["tweet_sent","tweet_caps_ratio","tweet_exclamations",
                  "tweet_econ","tweet_politics","tweet_media","tweet_china","tweet_russia"]

    sub = df[cols_news + cols_tweet].dropna()
    corr = sub.corr()

    # Extract only news → tweet block
    block = corr.loc[cols_tweet, cols_news]

    labels_n = [c.replace("news_","").replace("_"," ") for c in cols_news]
    labels_t = [c.replace("tweet_","").replace("_"," ") for c in cols_tweet]

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(block, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                xticklabels=labels_n, yticklabels=labels_t,
                linewidths=0.5, ax=ax, vmin=-0.4, vmax=0.4)
    ax.set_title("Pearson Correlation: Prior-Day News Signals → Tweet Signals", fontsize=12)
    ax.set_xlabel("News features (prior 24h)")
    ax.set_ylabel("Tweet features")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(f"{FIGURES}/correlation_news_to_tweet.png", dpi=150)
    plt.close()
    print(f"  Saved: {FIGURES}/correlation_news_to_tweet.png")

    # Print top correlations
    flat = block.stack().reset_index()
    flat.columns = ["tweet_feat","news_feat","corr"]
    flat["abs"] = flat["corr"].abs()
    top = flat.sort_values("abs", ascending=False).head(10)
    print("\n  Top 10 news → tweet correlations:")
    for _, row in top.iterrows():
        bar = "█" * int(abs(row["corr"]) * 30)
        sign = "+" if row["corr"] > 0 else "-"
        print(f"    {row['news_feat']:<30} → {row['tweet_feat']:<25}  r={sign}{abs(row['corr']):.3f}  {bar}")


# ── F. Tweet volume and sentiment over time ───────────────────────────────────
def run_time_analysis(df):
    print("\n" + "="*65)
    print("F. TIME ANALYSIS — Tweet patterns over 17 years")
    print("="*65)

    df2 = df.copy()
    df2["tweet_date"] = pd.to_datetime(df2["tweet_date"])
    df2 = df2.sort_values("tweet_date").set_index("tweet_date")

    monthly = df2[["tweet_sent","tweet_caps_ratio","tweet_econ","tweet_politics",
                   "news_sent_mean","news_econ_intensity"]].resample("ME").agg({
        "tweet_sent": "mean",
        "tweet_caps_ratio": "mean",
        "tweet_econ": "mean",
        "tweet_politics": "mean",
        "news_sent_mean": "mean",
        "news_econ_intensity": "mean",
    })
    monthly["tweet_count"] = df2.resample("ME")["tweet_sent"].count()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(monthly.index, monthly["tweet_count"], color="steelblue", lw=1.5)
    axes[0].set_ylabel("Tweets per month")
    axes[0].set_title("Tweet Volume")
    axes[0].fill_between(monthly.index, monthly["tweet_count"], alpha=0.3, color="steelblue")

    axes[1].plot(monthly.index, monthly["tweet_sent"],    color="navy",    lw=1.5, label="Tweet sentiment")
    axes[1].plot(monthly.index, monthly["news_sent_mean"], color="crimson", lw=1.2, alpha=0.7, label="News sentiment")
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].set_ylabel("VADER sentiment")
    axes[1].set_title("Tweet vs News Sentiment")
    axes[1].legend(fontsize=9)

    axes[2].plot(monthly.index, monthly["tweet_econ"],        color="green",   lw=1.5, label="Tweet economy")
    axes[2].plot(monthly.index, monthly["news_econ_intensity"], color="orange", lw=1.2, alpha=0.7, label="News economy")
    axes[2].set_ylabel("Economy keyword intensity")
    axes[2].set_title("Economy Topic Intensity: News vs Tweets")
    axes[2].legend(fontsize=9)

    plt.xlabel("Date")
    plt.tight_layout()
    fig.savefig(f"{FIGURES}/time_analysis.png", dpi=150)
    plt.close()
    print(f"  Saved: {FIGURES}/time_analysis.png")

    # Key stats
    print(f"\n  Overall tweet sentiment: {df2['tweet_sent'].mean():.4f}")
    print(f"  Twitter vs Truth Social sentiment:")
    tw = df2[df2["tweet_platform"]==1]["tweet_sent"].mean()
    ts = df2[df2["tweet_platform"]==0]["tweet_sent"].mean()
    print(f"    Twitter:       {tw:.4f}")
    print(f"    Truth Social:  {ts:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("MODEL 2: NEWS → TWEET PREDICTION")
    print("="*65)

    df = load_data()

    df, lda, vec, doc_topics = run_lda_tweets(df)
    run_sentiment_prediction(df)
    run_topic_prediction(df)
    run_granger(df)
    run_correlation_heatmap(df)
    run_time_analysis(df)

    print("\n" + "="*65)
    print("Done. Figures saved to figures/")
    print("="*65)


if __name__ == "__main__":
    main()
