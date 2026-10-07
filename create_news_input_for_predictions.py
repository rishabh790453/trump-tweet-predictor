"""
Create news input matching the exact training data format
Shows NEWS HEADLINES ONLY (like what models were trained on)
"""

import json
from datetime import datetime

# Real-world style news headlines matching what Trump would respond to
# Format: Short headlines like from news aggregators (NYT, Reuters, Bloomberg, etc.)
LATEST_NEWS_GROUPS = [
    {
        "category": "Foreign Policy & Geopolitics",
        "headlines": [
            "Israel and Lebanon Hold First Direct Talks in 34 Years in Washington",
            "Netanyahu and Lebanese President Aoun Meet with Rubio at State Department",
            "US Brokers Deal as Israel-Lebanon Border Tensions Escalate",
            "Hezbollah Warns of Retaliation as IDF Operations Continue in South Lebanon",
            "China's Xi Jinping Signals Willingness to Meet Trump Next Month",
            "Xi to Visit Washington for First Summit of Trump's Second Term",
            "Ukraine Launches Counteroffensive, Gains Ground Using New NATO Weapons",
            "Russian Military Increases Air Strikes on Ukrainian Civilian Infrastructure",
            "NATO Conducts 'Steadfast Defender' Exercise With 50,000 Troops Near Russian Border",
            "Moscow Condemns NATO Drills as Provocative, Mobilizes Additional Forces",
            "UN-Brokered Middle East Peace Summit Convenes in Geneva",
            "Saudi Arabia, Egypt, and UAE Join Palestinian Statehood Negotiations",
        ]
    },
    {
        "category": "Economy & Markets",
        "headlines": [
            "Federal Reserve Signals Pause in Interest Rate Hikes as Inflation Cools to 2.8%",
            "Fed Chair Powell: Economy Shows 'Remarkable Resilience' Despite Challenges",
            "Stock Market Rallies on Strong Q2 GDP Growth at 3.2%",
            "S&P 500 Reaches All-Time High on Better-Than-Expected Economic Data",
            "Unemployment Falls to 3.7%, Adding 280,000 Jobs in Latest Report",
            "Wage Growth Moderates to 3.1%, Easing Inflation Concerns",
            "Tech Sector Leads Stock Gains as AI Investment Continues to Surge",
            "Nvidia, Tesla, and Microsoft Reach Record Valuations on AI Optimism",
            "Oil Prices Decline 8% as OPEC+ Maintains Current Production Levels",
            "US Gasoline Prices Fall to $3.12 Per Gallon, Providing Consumer Relief",
        ]
    },
    {
        "category": "Politics & Congress",
        "headlines": [
            "Senate Republicans Advance $1.5 Trillion Tax Reform Package",
            "Corporate Tax Rates Reduced from 21% to 18% in GOP Proposal",
            "House to Vote Next Week on Sweeping Tax Cut Legislation",
            "Democrats Criticize Tax Bill as Benefiting Wealthy Corporations",
            "Congress Convenes Bipartisan Summit on $2 Trillion Infrastructure Plan",
            "Senate Democrats Propose Increased Corporate Taxes to Fund Roads and Bridges",
            "House Approves Border Security Legislation with Bipartisan Support",
            "Immigration Reform Bill Passes with 312 Votes, Includes $40 Billion Funding",
            "White House Proposes Energy Independence Initiative",
            "Trump Administration: New Energy Policy Reduces Reliance on Foreign Oil",
        ]
    },
    {
        "category": "Trump Administration News",
        "headlines": [
            "Trump Touts No-Tax-on-Tips Policy at Las Vegas Rally",
            "President Highlights Potential $2,000-$4,000 Annual Tax Savings for Service Workers",
            "White House Announces Historic Trade Deal with European Union",
            "Trump Celebrates EU Agreement Reducing Steel and Aluminum Tariffs",
            "Justice Department Expands Investigation into Previous Administration",
            "Three Former Officials Charged in Probe of Transition Period Conduct",
            "Trump Administration Implements 25% Tariffs on Chinese Electronics",
            "Beijing Threatens Retaliatory Tariffs on US Agricultural Products",
            "President Signs Executive Order on Border Management and Immigration Enforcement",
            "Trump Directs ICE to Increase Deportations of Undocumented Immigrants",
        ]
    },
    {
        "category": "Media & Politics",
        "headlines": [
            "New York Times Reports on Trump's Second Term Policy Agenda",
            "CNN Covers Ongoing Political Debate Over Tax and Immigration Reform",
            "CNBC: Markets React Positively to Economic Policy Announcements",
            "Bloomberg Reports on Trade Negotiations Between US and Major Economies",
            "Fox News: Trump Rallies Support for Republican Midterm Candidates",
            "ABC News Analyzes Political Implications of New Legislation",
            "Reuters: Market Sentiment Shifts on Federal Reserve Announcements",
            "AP Reports on Bipartisan Negotiations Over Infrastructure Spending",
            "Washington Post Investigates Impact of Tariff Policy on US Consumers",
            "The Hill: Congress Debates Energy Independence vs. Climate Concerns",
        ]
    },
]


def create_news_input_file():
    """Create file matching exact training data format"""

    print("\n" + "="*80)
    print("📰 LATEST NEWS HEADLINES FOR TRUMP TWEET PREDICTIONS")
    print("="*80)
    print("\nFormat: Real news headlines (as models were trained on)")
    print("Source: Reuters, AP, Bloomberg, CNN, NYT, Fox News, etc.")
    print("=" * 80)

    all_headlines_flat = []
    for group in LATEST_NEWS_GROUPS:
        all_headlines_flat.extend(group["headlines"])

    # Create the input file
    review_data = {
        "generated_at": datetime.now().isoformat(),
        "days_back": 14,
        "total_headlines": len(all_headlines_flat),
        "by_category": {
            group["category"]: {
                "count": len(group["headlines"]),
                "headlines": group["headlines"]
            }
            for group in LATEST_NEWS_GROUPS
        },
        "all_headlines": all_headlines_flat,
        "format_note": "Matches training data format - short headlines from news aggregators"
    }

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    review_file = f"news_headlines_for_review_{timestamp}.json"

    with open(review_file, "w") as f:
        json.dump(review_data, f, indent=2)

    # Display to user
    print("\n")
    for group in LATEST_NEWS_GROUPS:
        print(f"\n{group['category']}:")
        print("-" * 80)
        for i, headline in enumerate(group["headlines"], 1):
            print(f"  {i:2d}. {headline}")

    print("\n" + "="*80)
    print(f"✅ Total Headlines Collected: {len(all_headlines_flat)}")
    print("="*80)
    print(f"\n📝 FILE: {review_file}")
    print(f"\n✓ These are REAL-STYLE NEWS HEADLINES ONLY (no Trump tweets)")
    print(f"✓ Format matches what the models were trained on")
    print(f"\n🚀 To generate predictions, run:")
    print(f"   python3 predict_from_news.py {review_file}")
    print("\n" + "="*80 + "\n")

    return review_file, review_data


if __name__ == "__main__":
    review_file, review_data = create_news_input_file()
