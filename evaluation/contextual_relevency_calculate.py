# evaluation/contextual_relevency_calculate.py

# --- FIX: Added necessary imports for type hinting and JSON parsing ---
from typing import List, Dict, Any, Optional
import json
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

# This template is now simplified as it processes one context statement at a time.
class KeywordJSONContextualRelevancyTemplate:
    @staticmethod
    def generate_verdict(keywords: str, json_output: str, context_statement: str):
        return f"""Evaluate whether the given 'Context Statement' is relevant for generating the 'JSON Output' based on the 'Keywords'.

**
IMPORTANT: Please make sure to only return in JSON format with a single 'verdict' and a 'reason'.
The 'verdict' key should STRICTLY be either 'yes' or 'no'.
- 'yes': The context statement is relevant and useful for generating the JSON.
- 'no': The context statement is irrelevant.

Example:
{{
    "verdict": "yes",
    "reason": "This statement directly relates to the 'AI ethics' keyword and supports the creation of the JSON fields."
}}
**

Keywords: "{keywords}"
JSON Output:
{json_output}

Context Statement: "{context_statement}"

JSON:
"""

class KeywordJSONContextualRelevancyMetric(BaseMetric):
    def __init__(self, threshold: float = 0.7, model: str = "gpt-4o", 
                 include_reason: bool = True, strict_mode: bool = False):
        self.threshold = threshold
        self.model = model
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.evaluation_template = KeywordJSONContextualRelevancyTemplate()

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Measure contextual relevancy of a JSON output by evaluating each context statement individually.
        """
        if not test_case.retrieval_context:
            self.score = 1.0  # Or 0.0, depending on how you want to treat no context
            self.reason = "No retrieval context was provided."
            self.success = True
            return self.score

        # --- FIX: Iterate over each statement in the retrieval context for robustness ---
        relevant_count = 0
        verdicts = []
        for i, context_statement in enumerate(test_case.retrieval_context):
            try:
                # Generate prompt for a single context statement
                prompt = self.evaluation_template.generate_verdict(
                    keywords=test_case.input,
                    json_output=test_case.actual_output,
                    context_statement=context_statement
                )

                # Use deepeval's built-in LLM call for consistency
                # Assuming you have a `self.llm` or similar attribute if you inherit from a class that has one
                # For now, let's assume we need to import and use it.
                from deepeval.llm import anyscale_llm
                llm = anyscale_llm.AnyscaleLLM(model=self.model) # Or the LLM you use
                
                # --- FIX: Improved error handling to catch JSON and LLM issues ---
                raw_res = llm.generate(prompt)
                try:
                    verdict_data = json.loads(raw_res)
                    verdict = verdict_data.get('verdict', 'no').lower()
                    if verdict == 'yes':
                        relevant_count += 1
                    verdicts.append(verdict_data)
                except json.JSONDecodeError:
                    # This gives a much more useful error message
                    self.reason = f"Failed to parse LLM response for context statement #{i+1}. Raw response: '{raw_res}'"
                    self.score = 0.0
                    self.success = False
                    return self.score

            except Exception as e:
                # Catch any other exceptions during the process
                self.reason = f"An unexpected error occurred on context statement #{i+1}: {str(e)}"
                self.score = 0.0
                self.success = False
                return self.score

        # Calculate the final score
        self.score = relevant_count / len(test_case.retrieval_context)
        
        if self.include_reason:
            self.reason = self._generate_detailed_reason(self.score, relevant_count, len(test_case.retrieval_context), verdicts)

        self.success = self.score >= self.threshold
        return self.score

    def _generate_detailed_reason(self, score: float, relevant_count: int, total_count: int, verdicts: List[Dict]) -> str:
        """Generates a concise reason for the final score."""
        if total_count == 0:
            return "No context provided to evaluate."
        
        reason = f"Score is {score:.2f} because {relevant_count} out of {total_count} retrieved context statements were relevant. "
        
        irrelevant_verdicts = [v for v in verdicts if v.get('verdict', 'no').lower() == 'no']
        if irrelevant_verdicts:
            # Show the reason for the first irrelevant statement as an example
            first_irrelevant_reason = irrelevant_verdicts[0].get('reason', 'the statement was not relevant.')
            reason += f"Example of irrelevance: '{first_irrelevant_reason}'"
            
        return reason

    def is_successful(self) -> bool:
        return self.score >= self.threshold
