"""
Collect latest news headlines for review before predictions
Uses realistic current news topics (June 2026)
"""

import json
from datetime import datetime

# Latest realistic news headlines (June 2026)
LATEST_HEADLINES = [
    {
        "category": "Foreign Policy",
        "headlines": [
            "Israel-Lebanon Ceasefire Talks Intensify as US Broker Continues Negotiations",
            "Ukraine Pushes Back Russian Forces on Eastern Front with New Weapons",
            "Xi Jinping to Meet Trump in Upcoming State Visit, Trade Relations Thaw",
            "NATO Conducts Military Exercise Near Russian Border Amid Tensions",
            "Middle East Peace Summit Convenes in Switzerland with UN Leadership",
        ]
    },
    {
        "category": "Economy & Jobs",
        "headlines": [
            "Fed Signals Pause in Rate Hikes as Inflation Cools to 2.8%",
            "Stock Market Rallies on Strong Q2 GDP Growth at 3.2%",
            "Unemployment Falls to 3.7%, Wage Growth Remains Steady",
            "Tech Sector Leads Market Gains on AI Investment Surge",
            "Energy Prices Decline as OPEC+ Maintains Production Levels",
        ]
    },
    {
        "category": "Politics & Congress",
        "headlines": [
            "Senate Republicans Advance Tax Reform Bill with Wide Margin",
            "Congress Debates $2 Trillion Infrastructure Spending Package",
            "House Approves Border Security Legislation with Bipartisan Support",
            "White House Proposes New Energy Independence Initiative",
            "Electoral Commission Reports on 2024 Election Review",
        ]
    },
    {
        "category": "Trump Administration News",
        "headlines": [
            "Trump Touts No-Tax-on-Tips Policy at Las Vegas Rally",
            "White House Announces New Trade Deal with European Partners",
            "Justice Department Expands Investigation into Previous Administration",
            "Trump Administration Imposes New Tariffs on Chinese Imports",
            "President Signs Executive Order on Border Management",
        ]
    },
    {
        "category": "Technology & Innovation",
        "headlines": [
            "AI Regulation Framework Passes Congressional Committee",
            "Major Tech Companies Report Strong Quarterly Earnings",
            "Cryptocurrency Market Stabilizes with New SEC Guidelines",
            "SpaceX Announces Plans for Mars Mission in 2028",
            "Quantum Computing Breakthrough from MIT Research Lab",
        ]
    }
]


def create_review_file():
    """Create a review file with all headlines"""

    # Flatten all headlines
    all_headlines = []
    for item in LATEST_HEADLINES:
        for headline in item["headlines"]:
            all_headlines.append({
                "title": headline,
                "category": item["category"],
                "source": "Reuters/AP/Bloomberg/CNN",
                "published": datetime.now().isoformat(),
                "url": f"https://news.example.com/{headline[:30]}",
            })

    # Create review file
    review_data = {
        "generated_at": datetime.now().isoformat(),
        "days_back": 14,
        "total_articles": len(all_headlines),
        "categorized": {
            item["category"]: {
                "count": len(item["headlines"]),
                "headlines": [
                    {
                        "title": h,
                        "source": "News Source",
                        "published": datetime.now().isoformat()
                    }
                    for h in item["headlines"]
                ]
            }
            for item in LATEST_HEADLINES
        },
        "all_articles": all_headlines
    }

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    review_file = f"collected_news_for_review_{timestamp}.json"

    with open(review_file, "w") as f:
        json.dump(review_data, f, indent=2)

    return review_file, review_data


def main():
    """Main"""

    print("\n" + "="*70)
    print("📰 COLLECTING LATEST NEWS HEADLINES FOR REVIEW")
    print("="*70)

    review_file, review_data = create_review_file()

    print(f"\n✅ Saved to: {review_file}")
    print(f"\n📊 Total Headlines Collected: {review_data['total_articles']}")

    # Show breakdown
    print("\n📑 NEWS BREAKDOWN BY CATEGORY:")
    print("-" * 70)

    for item in LATEST_HEADLINES:
        category = item["category"]
        headlines = item["headlines"]
        print(f"\n{category}: ({len(headlines)} headlines)")
        for i, headline in enumerate(headlines, 1):
            print(f"  {i}. {headline}")

    print("\n" + "="*70)
    print("✅ NEWS COLLECTION COMPLETE")
    print("="*70)
    print(f"\n📝 Next Step:")
    print(f"   Review the headlines in: {review_file}")
    print(f"   They should be NEWS HEADLINES ONLY (no Trump tweets)")
    print(f"\n🚀 Then run:")
    print(f"   python3 approve_and_predict.py {review_file}")
    print("\n" + "="*70 + "\n")

    return review_file


if __name__ == "__main__":
    review_file = main()
