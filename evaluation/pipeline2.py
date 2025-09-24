# evaluation/pipeline2.py
import sys
import os
import json
from typing import List, Dict, Any

# Add parent directory to path to import config from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric

# Import and set API key from config.py
try:
    from config import OPENAI_API_KEY
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
except (ImportError, KeyError):
    print("⚠️  Pipeline 2: Could not find OPENAI_API_KEY in config or environment.")
    OPENAI_API_KEY = None

# --- Use relative imports for sibling modules ---
from .answer_relevency_calculate import KeywordJSONAnswerRelevancyMetric
from .bias_calculate import KeywordJSONBiasMetric
from .contextual_relevency_calculate import KeywordJSONContextualRelevancyMetric
from .conciseness_calculate import KeywordJSONConcisenessMetric
from .engagement_calculate import KeywordJSONEngagementMetric
from .hallucination_calculate import KeywordJSONHallucinationMetric
from .toxicity_calculate import KeywordJSONToxicityMetric


class EvaluationPipeline2:
    """
    Second evaluation pipeline using custom Keyword-JSON specific metrics.
    These metrics are designed to evaluate the structure and content of a JSON string.
    """
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is required for Pipeline 2")
        
        self.metrics = self._initialize_metrics()
        print("🔧 Evaluation Pipeline 2 (Keyword-JSON) initialized.")

    def _initialize_metrics(self):
        """Initializes all custom Keyword-JSON metrics."""
        return {
            "json_answer_relevancy": KeywordJSONAnswerRelevancyMetric(threshold=0.8),
            "json_bias": KeywordJSONBiasMetric(threshold=0.8),
            "json_contextual_relevancy": KeywordJSONContextualRelevancyMetric(threshold=0.7),
            "json_engagement": KeywordJSONEngagementMetric(threshold=0.7),
            "json_conciseness": KeywordJSONConcisenessMetric(threshold=0.7),
            "json_hallucination": KeywordJSONHallucinationMetric(threshold=0.7),
            "json_toxicity": KeywordJSONToxicityMetric(threshold=0.8)
        }

    # --- FIX: The method signature is updated to accept the 'retrieval_context' argument ---
    def evaluate_output(self, keywords: str, mode: str, result: Dict[str, Any], retrieval_context: List[str]) -> Dict[str, Any]:
        """
        Runs all Keyword-JSON metrics using an INDEPENDENT retrieval_context passed from main.py.
        """
        print(f"🔍 Evaluating {mode} output with Pipeline 2 (Keyword-JSON)...")
        
        json_data = result.get('data', {})
        actual_output_json_string = json.dumps(json_data, indent=2) if json_data else "{}"
        
        # --- FIX: The test case now uses the externally provided retrieval_context ---
        # This removes the previous circular logic.
        test_case = LLMTestCase(
            input=keywords,
            actual_output=actual_output_json_string,
            retrieval_context=retrieval_context
        )

        evaluation_results = {}
        for metric_name, metric in self.metrics.items():
            try:
                metric.measure(test_case)
                evaluation_results[metric_name] = {
                    "score": metric.score,
                    "reason": getattr(metric, "reason", "No reason provided."),
                    "successful": metric.is_successful()
                }
                print(f"  ✅ {metric_name}: Score={metric.score:.3f}")
            except Exception as e:
                evaluation_results[metric_name] = {"error": str(e), "score": None, "successful": False}
                print(f"  ❌ {metric_name}: Error - {e}")
        
        return evaluation_results

    def _extract_text_from_json(self, data: Dict) -> List[str]:
        """A simple text extractor from JSON. Note: This is no longer used for retrieval_context in the main evaluation flow."""
        texts = []
        if isinstance(data, dict):
            if "search1" in data and isinstance(data["search1"], dict):
                 for search_key in data:
                     if isinstance(data[search_key], dict):
                         for key, value in data[search_key].items():
                             if isinstance(value, str):
                                 texts.append(value)
            else:
                for key, value in data.items():
                    if isinstance(value, str):
                        texts.append(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                texts.append(item)
                    elif isinstance(value, dict):
                        for inner_key, inner_value in value.items():
                            if isinstance(inner_value, str):
                                texts.append(inner_value)
        return texts if texts else ["No textual context available."]
        
    def generate_evaluation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generates an evaluation summary for Pipeline 2."""
        summary = {"average_scores": {}, "success_rates": {}}
        all_metrics = list(self.metrics.keys())

        for metric_name in all_metrics:
            scores, successful_runs, total_runs = [], 0, 0
            for mode_result in results.values():
                if isinstance(mode_result, dict) and 'evaluation_p2' in mode_result:
                    metric_result = mode_result['evaluation_p2'].get(metric_name)
                    if metric_result and metric_result.get('score') is not None:
                        scores.append(metric_result['score'])
                        if metric_result.get('successful'):
                            successful_runs += 1
                        total_runs += 1

            if scores:
                summary["average_scores"][metric_name] = sum(scores) / len(scores)
                summary["success_rates"][metric_name] = successful_runs / total_runs if total_runs > 0 else 0
        return summary
