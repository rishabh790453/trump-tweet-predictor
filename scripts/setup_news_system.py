"""
Quick Setup Script for News Data Collection System
Helps you get started quickly
"""
import os
import json
import webbrowser

def check_config_file():
    """Check if config file exists"""
    if os.path.exists('news_api_config.json'):
        print("✓ news_api_config.json found")
        with open('news_api_config.json', 'r') as f:
            config = json.load(f)
        
        keys_configured = sum(1 for k, v in config.items() if not k.startswith('_') and v)
        print(f"  → {keys_configured} API keys configured")
        return True, config
    else:
        print("✗ news_api_config.json not found")
        return False, {}


def create_config_from_template():
    """Create config file from template"""
    if os.path.exists('news_api_config_template.json'):
        with open('news_api_config_template.json', 'r') as f:
            template = json.load(f)
        
        # Remove instructions
        template.pop('_instructions', None)
        
        with open('news_api_config.json', 'w') as f:
            json.dump(template, f, indent=2)
        
        print("✓ Created news_api_config.json from template")
        return True
    else:
        print("✗ Template file not found")
        return False


def show_api_key_links():
    """Show links to get API keys"""
    print("\n" + "="*70)
    print("FREE API KEY REGISTRATION LINKS")
    print("="*70)
    
    apis = {
        "NewsAPI": "https://newsapi.org/",
        "The Guardian": "https://open-platform.theguardian.com/access/",
        "NY Times": "https://developer.nytimes.com/",
        "Alpha Vantage": "https://www.alphavantage.co/support/#api-key"
    }
    
    for name, url in apis.items():
        print(f"\n{name}:")
        print(f"  {url}")
    
    print("\n" + "="*70)
    print("\nAll are FREE! Sign up takes 1-2 minutes each.")
    print("After getting keys, edit news_api_config.json")
    print("="*70 + "\n")
    
    # Ask if user wants to open links
    try:
        response = input("Open these links in browser? (y/n): ").lower()
        if response == 'y':
            for name, url in apis.items():
                print(f"Opening {name}...")
                webbrowser.open(url)
                import time
                time.sleep(1)
    except:
        pass


def run_demo():
    """Run the demo"""
    print("\n" + "="*70)
    print("RUNNING DEMO")
    print("="*70 + "\n")
    
    try:
        import demo_news_system
        demo_news_system.main()
    except Exception as e:
        print(f"Error running demo: {e}")
        print("\nTry running directly: python demo_news_system.py")


def run_quick_fetch():
    """Run quick fetch"""
    print("\n" + "="*70)
    print("RUNNING QUICK FETCH")
    print("="*70 + "\n")
    
    import subprocess
    try:
        subprocess.run(['python', 'live_news_monitor.py', '--mode', 'once'])
    except Exception as e:
        print(f"Error: {e}")
        print("\nTry running directly: python live_news_monitor.py --mode once")


def main():
    """Main setup wizard"""
    print("\n" + "="*70)
    print("NEWS DATA COLLECTION SYSTEM - QUICK SETUP")
    print("="*70 + "\n")
    
    # Step 1: Check dependencies
    print("[Step 1/4] Checking dependencies...")
    try:
        import pandas
        import requests
        import feedparser
        print("✓ All dependencies installed")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease run: pip install -r requirements_predictor.txt")
        return
    
    # Step 2: Check config file
    print("\n[Step 2/4] Checking configuration...")
    exists, config = check_config_file()
    
    if not exists:
        print("\nCreating config file...")
        if create_config_from_template():
            exists, config = check_config_file()
    
    # Step 3: API Keys
    print("\n[Step 3/4] API Keys Setup")
    
    has_any_key = any(v for k, v in config.items() if not k.startswith('_'))
    
    if not has_any_key:
        print("\n⚠ No API keys configured yet")
        print("\nYou can still use:")
        print("  • GDELT (free, no key needed, 1979-present)")
        print("  • RSS Feeds (free, no key needed)")
        print("\nFor maximum coverage, get free API keys:")
        
        try:
            response = input("\nShow API key registration links? (y/n): ").lower()
            if response == 'y':
                show_api_key_links()
        except:
            show_api_key_links()
    else:
        print("✓ API keys configured")
    
    # Step 4: What to do
    print("\n[Step 4/4] What would you like to do?")
    print("\nOptions:")
    print("  1. Run demo (recommended for first time)")
    print("  2. Quick fetch (test with current settings)")
    print("  3. Show setup guide")
    print("  4. Exit and configure manually")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            run_demo()
        elif choice == '2':
            run_quick_fetch()
        elif choice == '3':
            print("\n" + "="*70)
            print("Please read: NEWS_SETUP_GUIDE.md")
            print("="*70)
            print("\nIt contains:")
            print("  • Complete setup instructions")
            print("  • Usage examples")
            print("  • API key registration details")
            print("  • Python API reference")
            print("  • Troubleshooting tips")
        elif choice == '4':
            print("\n" + "="*70)
            print("MANUAL SETUP")
            print("="*70)
            print("\n1. Edit news_api_config.json and add your API keys")
            print("2. Run: python demo_news_system.py")
            print("3. Read: NEWS_SETUP_GUIDE.md")
            print("\nQuick commands:")
            print("  python live_news_monitor.py --mode once          # Quick test")
            print("  python live_news_monitor.py --mode backfill --years 10  # Get 10 years")
            print("  python live_news_monitor.py --mode live          # Start monitoring")
        else:
            print("Invalid choice")
    except:
        print("\n\nTo run manually:")
        print("  python demo_news_system.py")
        print("  python live_news_monitor.py --mode once")
    
    print("\n" + "="*70)
    print("SETUP COMPLETE")
    print("="*70)
    print("\nFor full documentation, see: NEWS_SETUP_GUIDE.md")
    print("For quick reference, see: README_NEWS.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
