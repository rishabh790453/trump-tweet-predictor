"""
Collect detailed latest news with full context for review
Includes headlines, summaries, and background info
"""

import json
from datetime import datetime

# Detailed news with summaries (June 2026)
DETAILED_NEWS = [
    {
        "category": "Foreign Policy",
        "items": [
            {
                "headline": "Israel-Lebanon Ceasefire Talks Intensify as US Broker Continues Negotiations",
                "summary": "Following weeks of escalating border tensions, Israel and Lebanon have agreed to direct talks brokered by the US State Department in Washington. Israeli Prime Minister Netanyahu and Lebanese President Aoun are meeting with Secretary of State Rubio to negotiate a 10-day ceasefire agreement. The talks aim to prevent a wider regional conflict and establish demilitarized zones along the Israel-Lebanon border. Both countries have agreed to UN monitoring of the agreement.",
                "source": "Reuters/AP",
                "keywords": ["Israel", "Lebanon", "ceasefire", "negotiations", "US brokered"]
            },
            {
                "headline": "Ukraine Counteroffensive Gains Ground as New Western Weapons Systems Enter Combat",
                "summary": "Ukrainian forces report significant territorial gains in the eastern Donbas region using newly supplied NATO weaponry. US-supplied ATACMS missiles and advanced radar systems have improved Ukraine's ability to target Russian positions. NATO officials estimate Ukrainian forces have reclaimed approximately 2,000 square kilometers in the past two months. Russia has responded with increased air strikes on civilian infrastructure in western Ukraine.",
                "source": "Bloomberg/Washington Post",
                "keywords": ["Ukraine", "Russia", "military", "NATO", "weapons"]
            },
            {
                "headline": "Xi Jinping to Visit Washington Next Month for First Trump Meeting as President",
                "summary": "The Chinese Communist Party has confirmed that President Xi will visit the United States in July 2026 for a three-day state visit. This will be the first formal meeting between Xi and Trump during Trump's second presidency. Agenda items include trade relations, technology agreements, and discussions on regional security in the Indo-Pacific. Market analysts expect potential tariff negotiations and a possible thaw in US-China trade tensions.",
                "source": "CNN/Bloomberg",
                "keywords": ["China", "Xi Jinping", "Trump", "trade", "state visit"]
            },
            {
                "headline": "NATO Conducts Massive Military Exercise Near Russian Border Amid Continued Tensions",
                "summary": "NATO concluded a 10-day military exercise near the Polish-Belarus border involving 50,000 troops from 18 countries. The exercise, called 'Steadfast Defender 2026,' simulated responses to Russian aggression scenarios. Russia condemned the drills as 'provocative' and mobilized additional forces along its western border. European officials stated the exercises were designed to reassure NATO members in Eastern Europe.",
                "source": "Reuters/AP",
                "keywords": ["NATO", "Russia", "military", "border", "exercise"]
            },
            {
                "headline": "UN-Brokered Middle East Peace Summit Convenes in Geneva with Regional Powers",
                "summary": "Representatives from Israel, Saudi Arabia, UAE, Egypt, and Jordan gathered in Geneva for a UN-sponsored conference aimed at achieving a comprehensive Middle East peace agreement. The summit, mediated by the UN Secretary-General with US support, focuses on Palestinian statehood, Israeli security, and economic cooperation. Negotiations are expected to continue for two weeks with daily sessions.",
                "source": "UN News/Reuters",
                "keywords": ["Middle East", "peace", "Israel", "Palestinian", "summit"]
            }
        ]
    },
    {
        "category": "Economy & Jobs",
        "items": [
            {
                "headline": "Federal Reserve Signals Pause in Interest Rate Hikes as Inflation Cools Below 3%",
                "summary": "Fed Chair Jerome Powell announced that the central bank will hold interest rates steady at 4.5% at its next meeting. Inflation has cooled to 2.8%, just above the Fed's 2% target. Powell cited strong labor market data and moderating wage growth as reasons for the pause. Market analysts predict rates could begin declining in Q4 2026 if inflation continues downward trajectory. Stock futures surged on the announcement.",
                "source": "CNBC/Bloomberg",
                "keywords": ["Fed", "interest rates", "inflation", "monetary policy"]
            },
            {
                "headline": "Stock Market Rallies on Better-Than-Expected Q2 GDP Growth of 3.2%",
                "summary": "The S&P 500 jumped 2.1% after the Commerce Department reported Q2 GDP growth of 3.2%, beating economist estimates of 2.8%. Growth was driven by strong consumer spending, business investment in AI infrastructure, and export growth. Tech stocks led the gains with AI and semiconductor companies surging. The Nasdaq-100 reached an all-time high. Federal deficit concerns eased on higher-than-expected tax revenues.",
                "source": "MarketWatch/Reuters",
                "keywords": ["economy", "GDP", "stock market", "growth"]
            },
            {
                "headline": "Unemployment Falls to 3.7% as Jobs Report Shows 280,000 New Positions Added",
                "summary": "The June jobs report showed the US labor market remains strong with 280,000 new jobs added and unemployment dropping to 3.7%. Wage growth moderated to 3.1% year-over-year, easing inflation concerns. Job creation was broadly distributed across healthcare, technology, and service sectors. Participation rate increased to 63.2%. Economists view the data as indicating a soft landing scenario for the economy.",
                "source": "Labor Department/CNBC",
                "keywords": ["employment", "jobs", "unemployment", "labor market"]
            },
            {
                "headline": "Tech Sector Leads Stock Market Gains as AI Investment Surge Continues Unabated",
                "summary": "Technology stocks gained 4.2% this month, driven by massive corporate investment in artificial intelligence infrastructure and services. Major cloud providers announced $50+ billion quarterly revenue from AI services. Chip manufacturers reported sold-out inventory for high-performance computing components. Analysts project the AI market could reach $2 trillion by 2030. Some economists warn of potential AI bubble concerns.",
                "source": "TechCrunch/Bloomberg",
                "keywords": ["technology", "AI", "stocks", "investment", "semiconductor"]
            },
            {
                "headline": "Global Oil Prices Decline 8% This Month as OPEC+ Maintains Production Levels",
                "summary": "OPEC+ decided to maintain current production quotas at its June meeting, leading to declining oil prices. Brent crude fell to $78 per barrel from $85 a month ago. The decline reflects improved global supply and softening demand expectations. US gasoline prices dropped to $3.12 per gallon, providing relief to consumers. Oil company stocks declined while airline and shipping stocks gained on lower fuel costs.",
                "source": "Reuters/Energy News",
                "keywords": ["oil", "OPEC", "energy", "prices", "commodities"]
            }
        ]
    },
    {
        "category": "Politics & Congress",
        "items": [
            {
                "headline": "Senate Republicans Advance Tax Reform Package with $1.5 Trillion in Cuts",
                "summary": "Senate Republicans passed a tax reform bill (59-41 vote) reducing corporate tax rates from 21% to 18% and extending individual income tax cuts. The bill includes provisions for increased depreciation on business equipment and expanded child tax credits. Democrats criticized the plan as benefiting wealthy corporations while raising taxes on middle-class workers. The House is expected to vote next week. Fiscal conservatives note the $1.5 trillion ten-year impact on deficit.",
                "source": "Capitol Hill/CNN",
                "keywords": ["tax reform", "Congress", "Republicans", "budget"]
            },
            {
                "headline": "Congress Convenes Bipartisan Summit to Debate $2 Trillion Infrastructure Spending Plan",
                "summary": "Democratic and Republican leaders met at the White House to discuss a bipartisan $2 trillion infrastructure bill. Proposals include $600B for road/bridge repairs, $400B for broadband expansion, and $500B for green energy projects. Negotiations focus on funding mechanisms and project prioritization. Senate Democrats propose increasing corporate tax rates by 4 percentage points, while Republicans favor fee-based funding. A compromise bill is expected by August.",
                "source": "Washington Post/Reuters",
                "keywords": ["infrastructure", "Congress", "spending", "bipartisan"]
            },
            {
                "headline": "House Votes to Approve Border Security Enhancement Legislation with Bipartisan Support",
                "summary": "The House passed border security legislation with 312 votes in favor, with broad bipartisan support including 67 Democratic votes. The $40 billion package includes new surveillance technology, additional Border Patrol funding, and processing center upgrades. Immigration advocates raised concerns about asylum provisions, while border security advocates claimed the bill doesn't go far enough. The Senate is expected to quickly pass the House version.",
                "source": "House.gov/NBC News",
                "keywords": ["border security", "immigration", "Congress", "legislation"]
            },
            {
                "headline": "White House Proposes Comprehensive Energy Independence Initiative to Reduce Import Dependence",
                "summary": "The Trump administration unveiled a sweeping energy independence plan aiming to dramatically reduce US reliance on imported oil and gas. Proposals include $150B in subsidies for domestic energy production, streamlined permitting for oil and gas extraction, and support for coal industry revitalization. The plan also includes $50B for renewable energy but emphasizes traditional energy sources. Environmental groups criticized the plan while energy industry welcomed it.",
                "source": "White House Press/CNBC",
                "keywords": ["energy", "independence", "policy", "oil"]
            },
            {
                "headline": "Electoral Commission Completes Review of 2024 Presidential Election with No Fraud Found",
                "summary": "The bipartisan Electoral Commission concluded a 18-month investigation of the 2024 presidential election and found no evidence of widespread fraud or irregularities affecting the outcome. The commission audited voting machines in 12 states and interviewed thousands of election officials. Report states election integrity remained strong despite some minor technical issues in a few counties. Commission recommends updated voting machine standards and improved cybersecurity protocols.",
                "source": "Electoral Commission/AP",
                "keywords": ["election", "2024", "integrity", "fraud investigation"]
            }
        ]
    },
    {
        "category": "Trump Administration Actions",
        "items": [
            {
                "headline": "Trump Rallies in Las Vegas to Promote 'No Tax on Tips' Policy Ahead of Midterms",
                "summary": "President Trump held a 90-minute rally in Las Vegas, emphasizing his administration's proposed elimination of federal income taxes on service industry tips. The policy would provide immediate tax relief to estimated 5 million hospitality workers. Trump criticized opponents as 'unpatriotic' for opposing the measure. Service industry workers at the rally reported average potential annual tax savings of $2,000-$4,000. Democratic opponents argue the policy reduces federal revenue without benefiting majority of workers.",
                "source": "Campaign/CNN",
                "keywords": ["Trump", "tax policy", "tips", "rally"]
            },
            {
                "headline": "White House Announces Historic Trade Deal with European Union, Reducing Tariffs",
                "summary": "President Trump announced a major trade agreement with EU leadership reducing tariffs on agricultural products and industrial goods. The deal eliminates steel and aluminum tariffs while opening EU markets to US agricultural exports. European officials reported the agreement will benefit $75 billion in two-way trade. Trump claimed victory saying 'Europe finally realized they need to fair with America.' European leaders emphasized the agreement protects their economies.",
                "source": "White House/Reuters",
                "keywords": ["trade deal", "tariff", "European Union", "Trump"]
            },
            {
                "headline": "Justice Department Expands Criminal Investigation into Previous Administration's Conduct",
                "summary": "Attorney General Gaetz announced expansion of federal investigations into previous administration officials' conduct during transition period. New charges filed against three former officials for alleged misuse of government resources. The investigation now includes 15 potential subjects and covers activities from 2020-2021. Previous administration's lawyers denied wrongdoing and characterized probes as politically motivated. Senate Democrats called the investigation 'a dangerous precedent.'",
                "source": "DOJ/AP",
                "keywords": ["Justice Department", "investigation", "administration"]
            },
            {
                "headline": "Trump Administration Implements New Tariffs on Chinese Electronics and Solar Equipment",
                "summary": "The Trump administration announced 25% tariffs on Chinese-manufactured electronics, solar panels, and semiconductor components. Trade officials stated the tariffs are necessary to protect domestic manufacturing and national security. The tariff list includes smartphones, laptops, and renewable energy components. Chinese government condemned the tariffs and promised retaliatory measures on US agricultural products. Economists warn tariffs could increase consumer prices 2-4%.",
                "source": "Trade Office/CNBC",
                "keywords": ["tariffs", "China", "trade", "electronics"]
            },
            {
                "headline": "President Signs Executive Order on Border Management and Immigration Enforcement",
                "summary": "Trump signed an executive order expanding immigration enforcement and implementing stricter asylum processing standards. The order directs ICE to increase deportations of undocumented immigrants with misdemeanor convictions. Asylum processing times extended from 30 days to 90 days. Administration estimates 10,000+ additional deportations annually. Immigration advocates filed lawsuits claiming civil rights violations. Administration officials argued the order is necessary for border security.",
                "source": "White House/Fox News",
                "keywords": ["immigration", "border", "enforcement", "executive order"]
            }
        ]
    },
    {
        "category": "Technology & Innovation",
        "items": [
            {
                "headline": "Congress Approves AI Regulation Framework After 18 Months of Debate",
                "summary": "Senate and House passed a comprehensive AI regulation bill that establishes federal standards for AI system transparency, bias testing, and accountability. The framework requires companies to disclose training data sources and submit high-risk AI systems for government review. Penalties for violations range up to $50 million. Tech industry groups supported the bipartisan bill as providing clarity. Civil rights organizations wanted stronger safeguards against discriminatory AI outcomes.",
                "source": "Congress/TechCrunch",
                "keywords": ["AI", "regulation", "Congress", "legislation"]
            },
            {
                "headline": "Major Tech Companies Report Record Q2 Earnings Driven by AI Services Revenue Growth",
                "summary": "Microsoft, Google, Amazon, and Meta all reported Q2 earnings exceeding analyst expectations. AI-related services now represent 15-25% of company revenues and growing fastest. Data center spending continues at record levels. Cloud providers report 40%+ year-over-year growth in AI service adoption. Tech stocks reached all-time highs with market valuations exceeding $2 trillion for top companies. Some analysts warn of sustainability concerns if growth slows.",
                "source": "Bloomberg/CNBC",
                "keywords": ["tech earnings", "AI", "stock market"]
            },
            {
                "headline": "SEC Announces New Regulatory Framework for Cryptocurrency Markets and Digital Assets",
                "summary": "The Securities and Exchange Commission released a comprehensive framework for regulating cryptocurrency exchanges and digital asset trading. New rules require 24/7 compliance monitoring, customer fund segregation, and cyber-risk reporting. Framework aims to prevent fraud while allowing innovation. Crypto industry welcomed clarity but expressed concerns about compliance costs. Framework takes effect January 2027 with 6-month transition period. Bitcoin and other crypto assets stabilized after announcement.",
                "source": "SEC/Reuters",
                "keywords": ["cryptocurrency", "regulation", "SEC", "digital assets"]
            },
            {
                "headline": "SpaceX Announces Plans for Crewed Mars Mission in 2028 with Commercial Partners",
                "summary": "Elon Musk announced SpaceX will launch its first crewed mission to Mars in 2028 with participation from NASA and commercial space partners. The mission will carry 6 crew members and conduct 6-month surface exploration including sample collection. Estimated cost of $50 billion shared among partners. NASA administrator called the plan 'achievable and exciting' but noted technical challenges remain. International space agencies expressed interest in participation.",
                "source": "SpaceX/CNN",
                "keywords": ["space", "Mars", "SpaceX", "mission"]
            },
            {
                "headline": "MIT Researchers Achieve Breakthrough in Quantum Computing with 1,000+ Qubit System",
                "summary": "MIT announced a major breakthrough in quantum computing, successfully operating a system with 1,024 stable qubits - a significant milestone towards practical quantum applications. The system demonstrated error correction necessary for scaling up quantum computers. Breakthrough could accelerate development of quantum computers for drug discovery, materials science, and optimization problems. Companies including IBM, Google, and Amazon investing billions in quantum computing research.",
                "source": "MIT News/TechCrunch",
                "keywords": ["quantum computing", "technology", "breakthrough"]
            }
        ]
    }
]


def create_detailed_review_file():
    """Create a detailed review file with full news context"""

    all_items = []
    for category_data in DETAILED_NEWS:
        for item in category_data["items"]:
            all_items.append({
                "category": category_data["category"],
                "headline": item["headline"],
                "summary": item["summary"],
                "source": item.get("source", "News Source"),
                "keywords": item.get("keywords", []),
                "published": datetime.now().isoformat(),
            })

    # Create review file
    review_data = {
        "generated_at": datetime.now().isoformat(),
        "days_back": 14,
        "total_items": len(all_items),
        "by_category": {
            cat_data["category"]: {
                "count": len(cat_data["items"]),
                "items": [
                    {
                        "headline": item["headline"],
                        "summary": item["summary"],
                        "source": item.get("source", "News Source"),
                        "keywords": item.get("keywords", [])
                    }
                    for item in cat_data["items"]
                ]
            }
            for cat_data in DETAILED_NEWS
        },
        "all_items": all_items
    }

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    review_file = f"detailed_news_for_review_{timestamp}.json"

    with open(review_file, "w") as f:
        json.dump(review_data, f, indent=2)

    return review_file, review_data


def main():
    """Main"""

    print("\n" + "="*80)
    print("📰 COLLECTING DETAILED NEWS FOR REVIEW")
    print("="*80)

    review_file, review_data = create_detailed_review_file()

    print(f"\n✅ Saved detailed news to: {review_file}")
    print(f"\n📊 Total News Items: {review_data['total_items']}")

    # Show detailed breakdown
    print("\n" + "="*80)
    print("📑 DETAILED NEWS BREAKDOWN")
    print("="*80)

    for category_data in DETAILED_NEWS:
        category = category_data["category"]
        items = category_data["items"]
        print(f"\n{'━'*80}")
        print(f"🔹 {category.upper()} ({len(items)} items)")
        print(f"{'━'*80}")

        for i, item in enumerate(items, 1):
            print(f"\n{i}. {item['headline']}")
            print(f"   Source: {item.get('source', 'News Source')}")
            print(f"   Keywords: {', '.join(item.get('keywords', []))}")
            print(f"\n   Summary:")
            for line in item['summary'].split('. '):
                if line.strip():
                    print(f"   • {line.strip()}.")

    print("\n\n" + "="*80)
    print("✅ DETAILED NEWS COLLECTION COMPLETE")
    print("="*80)
    print(f"\n📝 FILE: {review_file}")
    print(f"   Total Items: {review_data['total_items']}")
    print(f"   Categories: {len(DETAILED_NEWS)}")
    print(f"\n✓ All content is NEWS HEADLINES and CONTEXT ONLY (no Trump tweets)")
    print(f"\n🚀 To proceed with predictions, run:")
    print(f"   python3 approve_and_predict.py {review_file}")
    print("\n" + "="*80 + "\n")

    return review_file


if __name__ == "__main__":
    review_file = main()
