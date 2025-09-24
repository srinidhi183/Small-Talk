# evaluation/run_evaluation.py
"""
Automated evaluation runner for the SmallTalk application.
This script provides a command-line interface to run batch tests
and generate evaluation reports in Excel format.
"""
import argparse
import sys
import os
from typing import Dict, Any, List

# --- Path Correction ---
# This is crucial: it adds the project's root directory (one level up from `evaluation/`)
# to the Python path. This allows us to import modules like `main`, `config`, and `api_utils`.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# --- Imports ---
# Import the base application class from the root directory
from evaluation.main import SmallTalkApp
# Import API utilities needed for fetching evaluation context
from api_utils import get_urls_from_google, get_articles, get_news

# Import evaluation pipelines and batch tester utilities from the `evaluation` package
try:
    from evaluation.pipeline import SmallTalkEvaluationPipeline as EvaluationPipeline1
    from evaluation.pipeline2 import EvaluationPipeline2
    from evaluation.batch_tester import run_test_from_cli_keywords, run_batch_test_from_file, run_all_modes_and_save_to_excel
    EVAL_MODULES_AVAILABLE = True
    print("✅ Evaluation modules loaded successfully.")
except ImportError as e:
    print(f"❌ Critical Error: Could not import evaluation modules. Details: {e}")
    EVAL_MODULES_AVAILABLE = False


class EvaluatedSmallTalkApp(SmallTalkApp):
    """
    An extended version of the SmallTalkApp that integrates evaluation pipelines.
    This class is used exclusively within the evaluation runner.
    """
    def __init__(self):
        super().__init__()
        self.pipeline1 = EvaluationPipeline1() if EVAL_MODULES_AVAILABLE else None
        self.pipeline2 = EvaluationPipeline2() if EVAL_MODULES_AVAILABLE else None
        print("⚒️  EvaluatedSmallTalkApp initialized with evaluation pipelines.")

    def _get_shared_context(self, keywords: str, num_articles: int = 3) -> List[str]:
        """Fetches shared context for fair evaluation across modes."""
        print(f"\nFetching shared context for '{keywords}'...")
        try:
            urls = get_urls_from_google(keywords, num_results=num_articles)
            articles = get_articles(urls, max_articles=num_articles)
            if not articles:
                news_items = get_news(keywords, num_articles=num_articles)
                return [item.get('description', '') for item in news_items] if news_items else []
            return articles
        except Exception as e:
            print(f"❌ Error fetching shared context: {e}")
            return []
    
    def _apply_evaluations(self, keywords: str, mode: str, result: Dict[str, Any], retrieval_context: List[str]):
        """Applies evaluation pipelines to a result."""
        if not EVAL_MODULES_AVAILABLE or result.get('status') != 'success':
            return result
        
        if self.pipeline1:
            result['evaluation_p1'] = self.pipeline1.evaluate_output(keywords, mode, result, retrieval_context)
        if self.pipeline2:
            result['evaluation_p2'] = self.pipeline2.evaluate_output(keywords, mode, result, retrieval_context)
        return result

    # Override run methods to include context fetching and evaluation
    def run_search(self, keywords: str, retrieval_context: List[str], **kwargs):
        result = super().run_search(keywords, **kwargs)
        return self._apply_evaluations(keywords, 'search', result, retrieval_context)

    def run_trending(self, keywords: str, retrieval_context: List[str], **kwargs):
        result = super().run_trending(keywords, **kwargs)
        return self._apply_evaluations(keywords, 'trending', result, retrieval_context)

    def run_recommendation(self, keywords: str, retrieval_context: List[str], **kwargs):
        result = super().run_recommendation(keywords, **kwargs)
        return self._apply_evaluations(keywords, 'recommendation', result, retrieval_context)

    def run_update(self, keywords: str, retrieval_context: List[str], **kwargs):
        result = super().run_update(keywords, **kwargs)
        return self._apply_evaluations(keywords, 'update', result, retrieval_context)


def run_evaluation():
    """
    Execute the automatic evaluation process based on CLI arguments.
    """
    parser = argparse.ArgumentParser(description='SmallTalk Automated Evaluation Runner')
    
    batch_test_group = parser.add_argument_group('Batch Testing (Single Mode Excel Report)')
    comprehensive_group = parser.add_argument_group('Comprehensive Report (All Modes to Excel)')

    batch_test_group.add_argument('--batch-file', type=str, help='Path to an Excel file with test keywords.')
    batch_test_group.add_argument('--cli-keywords', nargs='+', help='One or more keywords for a single-mode batch test.')
    comprehensive_group.add_argument('--run-all-to-excel', nargs='+', help='Run ALL modes for the given keywords and save to a single Excel report.')
    parser.add_argument('--mode', choices=['search', 'trending', 'recommendation', 'update'], help='The specific mode to test for --batch-file or --cli-keywords.')
    
    args = parser.parse_args()

    if not EVAL_MODULES_AVAILABLE:
        sys.exit(1)

    app = EvaluatedSmallTalkApp()
    
    if args.run_all_to_excel:
        print("\n--- Running Comprehensive (All Modes) Excel Report ---")
        run_all_modes_and_save_to_excel(app, args.run_all_to_excel)
    elif args.batch_file:
        if not args.mode: parser.error("--mode is required when using --batch-file.")
        print(f"\n--- Running Batch Test from File for Mode: {args.mode.upper()} ---")
        run_batch_test_from_file(app, args.batch_file, args.mode)
    elif args.cli_keywords:
        if not args.mode: parser.error("--mode is required when using --cli-keywords.")
        print(f"\n--- Running Batch Test from CLI Keywords for Mode: {args.mode.upper()} ---")
        run_test_from_cli_keywords(app, args.cli_keywords, args.mode)
    else:
        parser.print_help()


if __name__ == "__main__":
    print("--- SmallTalk Evaluation Runner ---")
    run_evaluation()
