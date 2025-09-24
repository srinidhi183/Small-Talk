#conciseness_calculate.py
from typing import List, Dict, Any
import json
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
class KeywordJSONConcisenessTemplate:
    @staticmethod
    def generate_verdicts(keywords: str, json_output: str):
        return f"""Evaluate how concisely and efficiently each element in the JSON output conveys information for the given keywords.

**
IMPORTANT: Please make sure to only return in JSON format.
For each JSON field, evaluate:
1. Is the content expressed using minimum necessary words?
2. Does every word contribute meaningfully to the message?
3. Is there any redundancy or unnecessary elaboration?
4. Is the information complete without being verbose?
5. Could the same message be conveyed more efficiently?

Conciseness factors to consider:
- Brevity without losing essential information
- Elimination of redundant words and phrases
- Direct and clear expression
- Efficient communication
- Avoiding filler words and unnecessary elaboration

Example:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "json_field": "title",
            "content": "Machine Learning Guide",
            "conciseness_factors": ["direct", "no_redundancy"],
            "reason": "Title is concise and direct, conveying the core message in minimal words without losing clarity."
        }},
        {{
            "verdict": "no",
            "json_field": "description",
            "content": "This is a very comprehensive and detailed guide that will help you learn about machine learning in a very thorough manner",
            "conciseness_factors": ["redundancy", "filler_words"],
            "reason": "Contains redundant phrases 'very comprehensive and detailed', 'very thorough manner' and could be shortened to 'Comprehensive guide to machine learning'."
        }}
    ]
}}
**

Keywords: {keywords}
JSON Output: {json_output}

JSON:
"""

    @staticmethod
    def generate_reason(keywords: str, score: float, concise_fields: List[str], 
                       verbose_fields: List[str]):
        return f"""Based on the keywords and JSON conciseness analysis, provide a concise reason for the conciseness score.

**
IMPORTANT: Please make sure to only return in JSON format.
Example JSON:
{{
    "reason": "The score is <conciseness_score> because <your_reason>. The JSON output contains <concise_count> concise fields and <verbose_count> verbose fields."
}}
**

Keywords: {keywords}
Conciseness Score: {score}
Concise Fields: {concise_fields}
Verbose Fields: {verbose_fields}

JSON:
"""

class KeywordJSONConcisenessMetric(BaseMetric):
    def __init__(self, threshold: float = 0.7, model: str = "gpt-4o", 
                 include_reason: bool = True, strict_mode: bool = False):
        self.threshold = threshold
        self.model = model
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.evaluation_template = KeywordJSONConcisenessTemplate

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Measure conciseness of JSON output relative to input keywords
        """
        # Validate JSON output
        try:
            json_output = json.loads(test_case.actual_output)
        except json.JSONDecodeError as e:
            self.score = 0.0
            self.reason = f"Output is not valid JSON. Error: {str(e)}"
            self.success = False
            return self.score

        # Extract fields for conciseness evaluation
        field_analysis = self._extract_json_fields(json_output, test_case.input)

        # Evaluate conciseness in each field
        verdicts = self._evaluate_conciseness(
            test_case.input,  # keywords
            test_case.actual_output,
            field_analysis
        )

        # Calculate conciseness score
        self.score = self._calculate_conciseness_score(verdicts)

        # Generate detailed reason
        if self.include_reason:
            concise_fields = [v["json_field"] for v in verdicts if v["verdict"] == "yes"]
            verbose_fields = [v["json_field"] for v in verdicts if v["verdict"] == "no"]
            
            self.reason = self._generate_detailed_reason(
                test_case.input, self.score, concise_fields, verbose_fields, verdicts
            )
        else:
            self.reason = f"Conciseness score: {self.score:.2f}"

        self.success = self.score >= self.threshold
        return self.score

    def _extract_json_fields(self, json_output: Dict[str, Any], keywords: str) -> List[Dict[str, Any]]:
        """Extract text fields from JSON for conciseness evaluation"""
        fields = []

        def extract_fields(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    field_path = f"{prefix}.{key}" if prefix else key
                    
                    if isinstance(value, (str, int, float, bool)):
                        if isinstance(value, str) and len(str(value).strip()) > 0:
                            fields.append({
                                "field_path": field_path,
                                "field_name": key,
                                "content": str(value),
                                "expected_length": self._get_expected_length_range(key),
                                "content_type": self._classify_content_type(key)
                            })
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            if isinstance(item, str) and len(item.strip()) > 0:
                                fields.append({
                                    "field_path": f"{field_path}[{i}]",
                                    "field_name": key,
                                    "content": str(item),
                                    "expected_length": self._get_expected_length_range(key),
                                    "content_type": self._classify_content_type(key)
                                })
                            elif isinstance(item, dict):
                                extract_fields(item, f"{field_path}[{i}]")
                    else:
                        extract_fields(value, field_path)

        extract_fields(json_output)
        return fields

    def _get_expected_length_range(self, field_name: str) -> tuple:
        """Get expected word count range for different field types"""
        field_lower = field_name.lower()
        
        length_expectations = {
            "title": (2, 8), "headline": (3, 10), "name": (1, 5),
            "description": (5, 25), "summary": (10, 30), "overview": (8, 20),
            "tag": (1, 3), "keyword": (1, 3), "category": (1, 3),
            "content": (10, 50), "text": (5, 30), "body": (15, 100),
            "note": (3, 15), "comment": (5, 20)
        }
        
        for key, range_tuple in length_expectations.items():
            if key in field_lower:
                return range_tuple
        
        return (3, 20)  # Default range

    def _classify_content_type(self, field_name: str) -> str:
        """Classify content type for targeted conciseness evaluation"""
        field_lower = field_name.lower()
        
        if any(term in field_lower for term in ["title", "headline", "name"]):
            return "title"
        elif any(term in field_lower for term in ["description", "summary", "overview"]):
            return "descriptive"
        elif any(term in field_lower for term in ["tag", "keyword", "category"]):
            return "label"
        elif any(term in field_lower for term in ["content", "text", "body"]):
            return "content"
        
        return "general"

    def _evaluate_conciseness(self, keywords: str, json_output: str,
                            field_analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate conciseness of each JSON field"""
        verdicts = []

        for field_info in field_analysis:
            conciseness_analysis = self._analyze_conciseness_factors(
                field_info["content"],
                field_info["expected_length"],
                field_info["content_type"]
            )
            
            is_concise = conciseness_analysis["conciseness_score"] >= 0.7
            
            verdicts.append({
                "verdict": "yes" if is_concise else "no",
                "json_field": field_info["field_path"],
                "content": field_info["content"],
                "conciseness_factors": conciseness_analysis["factors"],
                "reason": self._generate_field_conciseness_reason(
                    field_info, conciseness_analysis, is_concise
                )
            })

        return verdicts

    def _analyze_conciseness_factors(self, content: str, expected_length: tuple, content_type: str) -> Dict[str, Any]:
        """Analyze conciseness factors in content"""
        factors = []
        conciseness_score = 1.0
        
        words = content.split()
        word_count = len(words)
        min_length, max_length = expected_length
        
        # Length appropriateness
        if min_length <= word_count <= max_length:
            factors.append("appropriate_length")
        elif word_count > max_length:
            factors.append("too_verbose")
            excess_ratio = (word_count - max_length) / max_length
            conciseness_score -= min(0.5, excess_ratio * 0.3)
        elif word_count < min_length and content_type != "label":
            factors.append("too_brief")
            conciseness_score -= 0.2
        
        # Redundancy detection
        redundant_phrases = [
            "very very", "really really", "quite quite", "extremely extremely",
            "in order to", "due to the fact that", "it is important to note that",
            "as a matter of fact", "the fact of the matter is", "at this point in time"
        ]
        
        content_lower = content.lower()
        redundancy_count = sum(1 for phrase in redundant_phrases if phrase in content_lower)
        if redundancy_count > 0:
            factors.append("redundancy")
            conciseness_score -= redundancy_count * 0.15
        
        # Filler words
        filler_words = ["very", "really", "quite", "rather", "somewhat", "actually", "basically", "literally"]
        filler_count = sum(1 for word in words if word.lower() in filler_words)
        if filler_count > 0:
            factors.append("filler_words")
            conciseness_score -= min(0.3, filler_count * 0.05)
        
        # Repetitive words
        word_freq = {}
        for word in words:
            if len(word) > 3:
                word_lower = word.lower()
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        repeated_words = [word for word, count in word_freq.items() if count > 1]
        if repeated_words:
            factors.append("word_repetition")
            conciseness_score -= min(0.2, len(repeated_words) * 0.05)
        
        # Direct expression (positive factor)
        if word_count <= max_length and not any(f in factors for f in ["redundancy", "filler_words"]):
            factors.append("direct_expression")
        
        # Information density
        meaningful_words = [word for word in words if len(word) > 3 and word.lower() not in filler_words]
        if len(meaningful_words) / word_count > 0.7:
            factors.append("high_information_density")
        else:
            factors.append("low_information_density")
            conciseness_score -= 0.1
        
        return {
            "conciseness_score": max(0.0, min(1.0, conciseness_score)),
            "factors": factors,
            "word_count": word_count,
            "expected_range": expected_length
        }

    def _generate_field_conciseness_reason(self, field_info: Dict[str, Any], 
                                         conciseness_analysis: Dict[str, Any], 
                                         is_concise: bool) -> str:
        """Generate reason for conciseness verdict"""
        word_count = conciseness_analysis["word_count"]
        expected_range = conciseness_analysis["expected_range"]
        factors = conciseness_analysis["factors"]
        
        if is_concise:
            positive_factors = [f for f in factors if f in ["appropriate_length", "direct_expression", "high_information_density"]]
            return f"Field '{field_info['field_name']}' is concise with {word_count} words (expected: {expected_range[0]}-{expected_range[1]}). Demonstrates: {', '.join(positive_factors) if positive_factors else 'efficient communication'}."
        else:
            negative_factors = [f for f in factors if f in ["too_verbose", "redundancy", "filler_words", "word_repetition", "low_information_density"]]
            return f"Field '{field_info['field_name']}' could be more concise ({word_count} words, expected: {expected_range[0]}-{expected_range[1]}). Issues: {', '.join(negative_factors) if negative_factors else 'exceeds optimal length'}."

    def _calculate_conciseness_score(self, verdicts: List[Dict[str, Any]]) -> float:
        """Calculate overall conciseness score"""
        if not verdicts:
            return 1.0

        if self.strict_mode:
            return 1.0 if all(v["verdict"] == "yes" for v in verdicts) else 0.0
        else:
            concise_count = sum(1 for v in verdicts if v["verdict"] == "yes")
            return concise_count / len(verdicts)

    def _generate_detailed_reason(self, keywords: str, score: float, 
                                concise_fields: List[str], verbose_fields: List[str],
                                verdicts: List[Dict[str, Any]]) -> str:
        """Generate detailed explanation for conciseness score"""
        total_fields = len(concise_fields) + len(verbose_fields)
        
        reason = f"The conciseness score is {score:.2f} because {len(concise_fields)} out of {total_fields} JSON fields are concise for the keywords '{keywords}'. "
        
        if concise_fields:
            reason += f"Well-optimized fields: {', '.join(concise_fields[:3])}. "
        
        if verbose_fields:
            reason += f"Fields needing optimization: {', '.join(verbose_fields[:3])}. "
        
        # Identify common issues
        all_factors = []
        for verdict in verdicts:
            if verdict["verdict"] == "no":
                all_factors.extend(verdict.get("conciseness_factors", []))
        
        if all_factors:
            common_issues = list(set([f for f in all_factors if f in ["redundancy", "filler_words", "too_verbose"]]))[:3]
            if common_issues:
                reason += f"Common optimization opportunities: {', '.join(common_issues)}."
        
        return reason

    def is_successful(self) -> bool:
        """Check if the metric passed the threshold"""
        return hasattr(self, 'success') and self.success
