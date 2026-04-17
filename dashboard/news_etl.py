import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

import feedparser

FEEDS = [
    ("Reuters",  "https://feeds.reuters.com/reuters/topNews"),
    ("AP News",  "https://feeds.ap.org/rss/apf-topnews"),
    ("NYT",      "https://rss.nytimes.com/services/xml/rss/nyt/US.xml"),
    ("BBC",      "https://feeds.bbci.co.uk/news/world/us_canada/rss.xml"),
    ("Fox News", "https://moxie.foxnews.com/google-publisher/politics.xml"),
    ("Politico", "https://www.politico.com/rss/politics08.xml"),
]

TRUMP_KEYWORDS = {
    "trump", "president", "white house", "tariff", "trade", "congress",
    "senate", "federal", "border", "immigration", "economy", "fed",
    "interest rate", "stock", "market", "china", "russia", "iran",
    "nato", "ukraine", "israel", "tax", "budget", "inflation", "dollar",
    "crypto", "bitcoin", "musk", "doge", "elon",
}


@dataclass
class NewsItem:
    id: str
    title: str
    source: str
    url: str
    published: str
    relevance: float  # 0-1, how Trump-tweet-worthy


def _relevance(title: str) -> float:
    words = set(title.lower().split())
    hits = len(words & TRUMP_KEYWORDS)
    return min(hits / 3.0, 1.0)


def fetch_all_news(max_per_feed: int = 10) -> list[NewsItem]:
    items: list[NewsItem] = []
    seen: set[str] = set()

    for source, url in FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                title = (entry.get("title") or "").strip()
                if not title or title in seen:
                    continue
                seen.add(title)
                item_id = hashlib.md5(title.encode()).hexdigest()[:10]
                published = entry.get("published", datetime.now(timezone.utc).isoformat())
                items.append(
                    NewsItem(
                        id=item_id,
                        title=title,
                        source=source,
                        url=entry.get("link", ""),
                        published=published,
                        relevance=_relevance(title),
                    )
                )
        except Exception as exc:
            print(f"[ETL] Feed error {source}: {exc}")

    # Sort highest relevance first
    items.sort(key=lambda x: x.relevance, reverse=True)
    return items
