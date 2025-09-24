# main.py

import argparse
import json
import sys
import os
from typing import Dict, Any

# Add the current directory to Python path for module resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Core Application Imports ---
from search_bar import search_bar
from recommendation import generate_recommendation
from trending import generate_trending
from update_me import generate_update_element

# Try to import the API key from config.py
try:
    from config import OPENAI_API_KEY
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
except ImportError:
    print("Warning: config.py with OPENAI_API_KEY not found. API calls may fail.")


class SmallTalkApp:
    """Core SmallTalk application class without any evaluation logic."""
    def __init__(self):
        self.supported_modes = ['search', 'trending', 'recommendation', 'update']
        print("✅ SmallTalkApp initialized in standard mode.")

    def run_search(self, keywords: str, **kwargs) -> Dict[str, Any]:
        print(f"\n🔍 Searching for: {keywords}")
        try:
            data = search_bar(keywords, allow_options=kwargs.get('allow_options', True))
            return {'status': 'success', 'mode': 'search', 'keywords': keywords, 'data': data}
        except Exception as e:
            return {'status': 'error', 'mode': 'search', 'keywords': keywords, 'error': str(e)}

    def run_trending(self, keywords: str, **kwargs) -> Dict[str, Any]:
        print(f"\n📈 Analyzing trending topic: {keywords}")
        try:
            data = generate_trending(keywords, **kwargs)
            return {'status': 'success', 'mode': 'trending', 'keywords': keywords, 'data': data}
        except Exception as e:
            return {'status': 'error', 'mode': 'trending', 'keywords': keywords, 'error': str(e)}

    def run_recommendation(self, keywords: str, **kwargs) -> Dict[str, Any]:
        print(f"\n💡 Generating recommendations for: {keywords}")
        try:
            data = generate_recommendation(keywords, **kwargs)
            return {'status': 'success', 'mode': 'recommendation', 'keywords': keywords, 'data': data}
        except Exception as e:
            return {'status': 'error', 'mode': 'recommendation', 'keywords': keywords, 'error': str(e)}

    def run_update(self, keywords: str, **kwargs) -> Dict[str, Any]:
        print(f"\n📋 Generating update for: {keywords}")
        try:
            data = generate_update_element(keywords, **kwargs)
            return {'status': 'success', 'mode': 'update', 'keywords': keywords, 'data': data}
        except Exception as e:
            return {'status': 'error', 'mode': 'update', 'keywords': keywords, 'error': str(e)}

    def run_all_modes(self, keywords: str, **kwargs) -> Dict[str, Any]:
        """Runs all modes for a given keyword without evaluation."""
        print(f"🚀 Running comprehensive analysis for: {keywords}")
        results = {
            'search': self.run_search(keywords, **kwargs),
            'trending': self.run_trending(keywords, **kwargs),
            'recommendation': self.run_recommendation(keywords, **kwargs),
            'update': self.run_update(keywords, **kwargs)
        }
        return {'status': 'success', 'mode': 'comprehensive', 'keywords': keywords, 'data': results}

    def print_result(self, result: Dict[str, Any]):
        """Prints the result in a readable JSON format."""
        print(json.dumps(result, indent=2, ensure_ascii=False))

def interactive_mode():
    """Starts an interactive session to run the SmallTalk app."""
    app = SmallTalkApp()
    print("\nWelcome to SmallTalk Interactive Mode!")
    print("Available modes: search, trending, recommendation, update, all")
    print("Type 'quit' or press Ctrl+C to exit.\n")
    
    while True:
        try:
            keywords = input("Enter keywords: ").strip()
            if keywords.lower() == 'quit': break
            if not keywords: continue
            
            mode = input(f"Enter mode (default: all): ").strip().lower() or 'all'
            
            if mode == 'all':
                result = app.run_all_modes(keywords=keywords)
            elif mode in app.supported_modes:
                mode_functions = {
                    'search': app.run_search, 'trending': app.run_trending,
                    'recommendation': app.run_recommendation, 'update': app.run_update
                }
                result = mode_functions[mode](keywords=keywords)
            else:
                print(f"❌ Unknown mode. Please choose from {app.supported_modes} or 'all'.")
                continue
            
            app.print_result(result)
            print("\n" + "="*60 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!"); break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

def main():
    """Main CLI for running a single instance of the application."""
    parser = argparse.ArgumentParser(description='SmallTalk - AI-powered conversation assistant.')
    parser.add_argument('--mode', choices=['search', 'trending', 'recommendation', 'update', 'all'], help='Operation mode for a single run.')
    parser.add_argument('--keywords', type=str, help='Keywords for analysis.')
    args = parser.parse_args()
    
    if args.keywords and args.mode:
        app = SmallTalkApp()
        if args.mode == 'all':
            result = app.run_all_modes(keywords=args.keywords)
        else:
            mode_functions = { 'search': app.run_search, 'trending': app.run_trending, 'recommendation': app.run_recommendation, 'update': app.run_update }
            result = mode_functions[args.mode](keywords=args.keywords)
        app.print_result(result)
    else:
        interactive_mode()

if __name__ == "__main__":
    main()
