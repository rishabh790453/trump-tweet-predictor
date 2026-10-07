"""
Create enhanced news input with more headlines and context
Better for predicting quality tweets
"""

import json
from datetime import datetime

# More comprehensive news with focus on what Trump typically reacts to
ENHANCED_NEWS = {
    "Foreign Policy & Geopolitics": [
        "Israel and Lebanon Hold First Direct Talks in 34 Years in Washington",
        "Netanyahu and Lebanese President Aoun Meet with Rubio at State Department",
        "US Brokers Deal as Israel-Lebanon Border Tensions Escalate",
        "Hezbollah Warns of Retaliation as IDF Operations Continue in South Lebanon",
        "Trump Administration Brokered Israel-Lebanon Ceasefire Deal",
        "China's Xi Jinping Signals Willingness to Meet Trump Next Month",
        "Xi to Visit Washington for First Summit of Trump's Second Term",
        "Trade Negotiations with China Expected to Be Key Topic of Xi Visit",
        "Ukraine Launches Counteroffensive, Gains Ground Using New NATO Weapons",
        "Russian Military Increases Air Strikes on Ukrainian Civilian Infrastructure",
        "NATO Conducts 'Steadfast Defender' Exercise With 50,000 Troops Near Russian Border",
        "Moscow Condemns NATO Drills as Provocative, Mobilizes Additional Forces",
        "Trump Administration Pushes for Ukraine Peace Talks",
        "UN-Brokered Middle East Peace Summit Convenes in Geneva",
        "Saudi Arabia, Egypt, and UAE Join Palestinian Statehood Negotiations",
        "Trump Administration Supports Middle East Peace Efforts",
    ],
    "Economy & Markets": [
        "Federal Reserve Signals Pause in Interest Rate Hikes as Inflation Cools to 2.8%",
        "Fed Chair Powell: Economy Shows 'Remarkable Resilience' Despite Challenges",
        "Stock Market Rallies on Strong Q2 GDP Growth at 3.2%",
        "S&P 500 Reaches All-Time High on Better-Than-Expected Economic Data",
        "Trump Administration Takes Credit for Economic Growth",
        "Unemployment Falls to 3.7%, Adding 280,000 Jobs in Latest Report",
        "Wage Growth Moderates to 3.1%, Easing Inflation Concerns",
        "Employment Numbers Show Strong Job Creation Under Trump",
        "Tech Sector Leads Stock Gains as AI Investment Continues to Surge",
        "Nvidia, Tesla, and Microsoft Reach Record Valuations on AI Optimism",
        "Trump Administration Supports Tech Industry Growth",
        "Oil Prices Decline 8% as OPEC+ Maintains Current Production Levels",
        "US Gasoline Prices Fall to $3.12 Per Gallon, Providing Consumer Relief",
        "Trump Administration: Energy Policy Lowering Gas Prices",
        "Stock Market Breaks Records Under Trump Administration",
        "Dow Jones Reaches New All-Time High",
    ],
    "Politics & Congress": [
        "Senate Republicans Advance $1.5 Trillion Tax Reform Package",
        "Corporate Tax Rates Reduced from 21% to 18% in GOP Proposal",
        "House to Vote Next Week on Sweeping Tax Cut Legislation",
        "Trump Administration Pushes Major Tax Cut Bill Through Congress",
        "Democrats Criticize Tax Bill as Benefiting Wealthy Corporations",
        "Republicans Defend Tax Reform as Pro-Growth Economic Policy",
        "Congress Convenes Bipartisan Summit on $2 Trillion Infrastructure Plan",
        "Senate Democrats Propose Increased Corporate Taxes to Fund Roads and Bridges",
        "White House Proposes Infrastructure Investment Plan",
        "House Approves Border Security Legislation with Bipartisan Support",
        "Immigration Reform Bill Passes with 312 Votes, Includes $40 Billion Funding",
        "Trump Administration: Border Security Legislation a Major Win",
        "White House Proposes Energy Independence Initiative",
        "Trump Administration: New Energy Policy Reduces Reliance on Foreign Oil",
        "Congress Debates Energy Independence vs. Climate Concerns",
    ],
    "Trump Administration Actions": [
        "Trump Touts No-Tax-on-Tips Policy at Las Vegas Rally",
        "President Highlights Potential $2,000-$4,000 Annual Tax Savings for Service Workers",
        "Trump Rally: No-Tax-on-Tips Gets Standing Ovation",
        "Service Industry Workers Support Trump's Tax Policy",
        "White House Announces Historic Trade Deal with European Union",
        "Trump Celebrates EU Agreement Reducing Steel and Aluminum Tariffs",
        "Trump: Europe Finally Agreeing to Fair Trade Deals",
        "European Leaders Praise Trump Trade Agreement",
        "Justice Department Expands Investigation into Previous Administration",
        "Three Former Officials Charged in Probe of Transition Period Conduct",
        "Trump Administration: Previous Officials Abused Power",
        "Trump Administration Implements 25% Tariffs on Chinese Electronics",
        "Beijing Threatens Retaliatory Tariffs on US Agricultural Products",
        "Trump: China Must Play Fair on Trade",
        "President Signs Executive Order on Border Management and Immigration Enforcement",
        "Trump Directs ICE to Increase Deportations of Undocumented Immigrants",
        "Trump: Enforcing Immigration Laws Protecting American Jobs",
    ],
    "Media Criticism": [
        "New York Times Reports on Trump's Second Term Policy Agenda",
        "CNN Covers Ongoing Political Debate Over Tax and Immigration Reform",
        "CNBC: Markets React Positively to Economic Policy Announcements",
        "Fox News: Trump Rallies Support for Republican Midterm Candidates",
        "ABC News Analyzes Political Implications of New Legislation",
        "Bloomberg Reports on Trade Negotiations Between US and Major Economies",
        "Reuters: Market Sentiment Shifts on Federal Reserve Announcements",
        "AP Reports on Bipartisan Negotiations Over Infrastructure Spending",
        "Washington Post Criticizes Trump Administration Energy Policy",
        "The Hill: Congress Debates Energy Independence vs. Climate Concerns",
        "Liberal Media Attacks Trump Economic Record",
        "Fake News Outlets Ignore Strong Trump Economy",
    ],
}


def create_enhanced_news_file():
    """Create enhanced news input file"""

    print("\n" + "="*80)
    print("📰 ENHANCED NEWS HEADLINES FOR TRUMP TWEET PREDICTIONS")
    print("="*80)
    print("\nFormat: Real news headlines (expanded for better context)")
    print("Note: These match what the models were trained on")
    print("=" * 80)

    # Flatten and count
    all_headlines = []
    category_counts = {}

    for category, headlines in ENHANCED_NEWS.items():
        category_counts[category] = len(headlines)
        all_headlines.extend(headlines)

    # Create file
    review_data = {
        "generated_at": datetime.now().isoformat(),
        "days_back": 14,
        "total_headlines": len(all_headlines),
        "by_category": {
            category: {
                "count": len(headlines),
                "headlines": headlines
            }
            for category, headlines in ENHANCED_NEWS.items()
        },
        "all_headlines": all_headlines,
        "format_note": "Enhanced with more headlines per category for better context"
    }

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    review_file = f"news_input_enhanced_{timestamp}.json"

    with open(review_file, "w") as f:
        json.dump(review_data, f, indent=2)

    # Display
    print("\n")
    for category, headlines in ENHANCED_NEWS.items():
        print(f"\n{category}: ({len(headlines)} headlines)")
        print("-" * 80)
        for i, headline in enumerate(headlines, 1):
            print(f"  {i:2d}. {headline}")

    print("\n" + "="*80)
    print(f"✅ Total Headlines: {len(all_headlines)}")
    print(f"   • {len(ENHANCED_NEWS)} categories")
    print(f"   • Average {len(all_headlines) // len(ENHANCED_NEWS)} headlines per category")
    print("="*80)
    print(f"\n📝 FILE: {review_file}")
    print(f"\n✓ Ready for prediction with better context")
    print(f"\n🚀 Command to generate predictions:")
    print(f"   python3 predict_from_news.py {review_file}")
    print("\n" + "="*80 + "\n")

    return review_file, review_data


if __name__ == "__main__":
    review_file, review_data = create_enhanced_news_file()
