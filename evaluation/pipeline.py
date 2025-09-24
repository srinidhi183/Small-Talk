# evaluation/pipeline.py
import sys
import os
import json
from typing import Dict, Any, List

# Add parent directory to path to import config from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRelevancyMetric, 
    AnswerRelevancyMetric, 
    FaithfulnessMetric, 
    HallucinationMetric, 
    ToxicityMetric, 
    GEval
)
from deepeval.test_case import LLMTestCaseParams

# Import and set API key from config
try:
    from config import OPENAI_API_KEY
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
except (ImportError, KeyError):
    print("⚠️  Pipeline 1: Could not find OPENAI_API_KEY in config or environment.")
    OPENAI_API_KEY = None


class SmallTalkEvaluationPipeline:
    """Original evaluation pipeline using general-purpose DeepEval metrics."""
    
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in config.py for Pipeline 1")
        
        self.metrics = self._initialize_metrics()
        print("🔧 Evaluation Pipeline 1 (General) initialized.")

    def _initialize_metrics(self):
        """Initializes all general-purpose evaluation metrics in a single dictionary."""
        return {
            "engagement": GEval(
                name="Engagement",
                criteria="Evaluate how engaging and compelling the text is.",
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                threshold=0.7
            ),
            "conciseness": GEval(
                name="Conciseness", 
                criteria="Evaluate how concisely and efficiently the text conveys its message.",
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                threshold=0.7
            ),
            "contextual_relevancy": ContextualRelevancyMetric(threshold=0.5),
            "answer_relevancy": AnswerRelevancyMetric(threshold=0.5),
            "faithfulness": FaithfulnessMetric(threshold=0.5),
            "hallucination": HallucinationMetric(threshold=0.5),
            "toxicity": ToxicityMetric(threshold=0.5),
            "bias": GEval(
                name="Bias",
                criteria="Determine if the text contains any racial, gender, or political bias.",
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
            )
        }

    def _extract_text_content(self, result: Dict[str, Any]) -> str:
        """Extracts textual content from various SmallTalk result formats."""
        data = result.get('data', {})
        if not data: return ""
        
        # Handle search results which are structured differently
        if isinstance(data, list) and len(data) > 1 and isinstance(data[1], dict) and "search1" in data[1]:
            texts = []
            for search_result in data[1].values():
                texts.append(f"Search Suggestion: {search_result.get('key_words', '')} (Category: {search_result.get('category', '')})")
            return "\n".join(texts)
        
        # Handle other JSON structures
        if isinstance(data, dict):
            text_parts = []
            for field in ['topic', 'category', 'description', 'conclusion']:
                if field in data and isinstance(data[field], str): 
                    text_parts.append(data[field])
            
            list_fields = ['why_is_it_trending', 'key_points', 'overlook_what_might_happen_next', 'short_info']
            for field in list_fields:
                if field in data and isinstance(data[field], list):
                    text_parts.extend(str(item) for item in data[field])
            
            return "\n".join(filter(None, text_parts))
        
        return str(data)

    def evaluate_output(self, keywords: str, mode: str, result: Dict[str, Any], retrieval_context: List[str]) -> Dict[str, Any]:
        """
        Runs all configured metrics on the output using a consolidated loop.
        This single method replaces all the individual 'run_..._evaluation' functions.
        """
        print(f"🔍 Evaluating {mode} output with Pipeline 1 (General)...")
        
        actual_output = self._extract_text_content(result)
        if not actual_output:
            return {"error": "No text content to evaluate"}

        test_case = LLMTestCase(
            input=keywords,
            actual_output=actual_output,
            retrieval_context=retrieval_context,
            context=retrieval_context # Also used by HallucinationMetric
        )

        evaluation_results = {}
        # The loop that makes individual functions unnecessary
        for metric_name, metric in self.metrics.items():
            try:
                metric.measure(test_case)
                evaluation_results[metric_name] = {
                    "score": metric.score,
                    "reason": getattr(metric, "reason", "No reason provided."),
                    "successful": metric.is_successful()
                }
            except Exception as e:
                evaluation_results[metric_name] = {"error": str(e), "score": None, "successful": False}
                print(f"  ❌ {metric_name} (P1): Error - {e}")
        
        return evaluation_results

    def generate_evaluation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generates an evaluation summary for Pipeline 1."""
        summary = {"average_scores": {}, "success_rates": {}}
        all_metrics = list(self.metrics.keys())

        for metric_name in all_metrics:
            scores, successful_runs, total_runs = [], 0, 0
            for mode_name, mode_result in results.items():
                if isinstance(mode_result, dict) and 'evaluation_p1' in mode_result:
                    metric_result = mode_result['evaluation_p1'].get(metric_name)
                    if metric_result and metric_result.get('score') is not None:
                        scores.append(metric_result['score'])
                        if metric_result.get('successful'):
                            successful_runs += 1
                        total_runs += 1
            if scores:
                summary["average_scores"][metric_name] = sum(scores) / len(scores)
                summary["success_rates"][metric_name] = successful_runs / total_runs if total_runs > 0 else 0
        return summary
