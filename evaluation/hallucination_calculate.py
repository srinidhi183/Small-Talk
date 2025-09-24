#hallucination_calculate.py

import json
from typing import List, Dict, Any, Optional
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class KeywordJSONHallucinationTemplate:
    @staticmethod
    def generate_verdicts(keywords: str, json_output: str, retrieval_context: List[str]):
        return f"""Evaluate whether each claim/value in the JSON output is supported by the retrieval context, given the input keywords.

**
IMPORTANT: Please make sure to only return in JSON format.
For each JSON field and its content, evaluate:
1. Is the information/claim supported by the retrieval context?
2. Is the field value factually accurate based on the provided context?
3. Are there any fabricated details not present in the context?

A "hallucination" occurs when:
- JSON contains information contradicting the retrieval context
- JSON includes specific details not mentioned in the context
- JSON fabricates data that cannot be verified from the context

Example Keywords: "machine learning algorithms neural networks"
Example Context: ["Neural networks are ML algorithms", "They use backpropagation for training"]
Example JSON: {{"title": "Neural Networks", "inventor": "John Smith", "year": "1943"}}

Example:
{{
    "verdicts": [
        {{
            "verdict": "no",
            "json_field": "title", 
            "claim": "Neural Networks",
            "reason": "Title is supported by context mentioning 'Neural networks are ML algorithms'."
        }},
        {{
            "verdict": "yes",
            "json_field": "inventor",
            "claim": "John Smith", 
            "reason": "No information about inventor 'John Smith' found in retrieval context - this appears to be hallucinated."
        }},
        {{
            "verdict": "yes", 
            "json_field": "year",
            "claim": "1943",
            "reason": "Specific year '1943' is not mentioned in the retrieval context - this appears to be fabricated."
        }}
    ]
}}

Note: 'verdict' should be 'yes' for hallucinations (unsupported claims) and 'no' for supported claims.
**

Keywords: {keywords}
JSON Output: {json_output}
Retrieval Context: {retrieval_context}

JSON:
"""

    @staticmethod
    def generate_reason(keywords: str, score: float, hallucinated_claims: List[str], 
                       supported_claims: List[str]):
        return f"""Based on the keywords, JSON output, and hallucination analysis, provide a concise reason for the score.

**
IMPORTANT: Please make sure to only return in JSON format.
Example JSON:
{{
    "reason": "The score is <hallucination_score> because <your_reason>. The JSON output contains <hallucinated_count> hallucinated claims and <supported_count> supported claims."
}}
**

Keywords: {keywords}
Hallucination Score: {score}
Hallucinated Claims: {hallucinated_claims}
Supported Claims: {supported_claims}

JSON:
"""

class KeywordJSONHallucinationMetric(BaseMetric):
    def __init__(self, threshold: float = 0.7, model: str = "gpt-4o", 
                 include_reason: bool = True, strict_mode: bool = False):
        self.threshold = threshold
        self.model = model
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.evaluation_template = KeywordJSONHallucinationTemplate
        
    def measure(self, test_case: LLMTestCase) -> float:
        """
        Measure hallucination in JSON output - lower scores indicate more hallucinations
        """
        # Validate inputs
        if not test_case.retrieval_context:
            self.score = 0.0
            self.reason = "No retrieval context provided for hallucination detection"
            self.success = False
            return self.score

        # Validate JSON output
        try:
            json_output = json.loads(test_case.actual_output)
        except json.JSONDecodeError as e:
            self.score = 0.0
            self.reason = f"Output is not valid JSON. Error: {str(e)}"
            self.success = False
            return self.score

        # Extract claims from JSON for evaluation
        json_claims = self._extract_json_claims(json_output, test_case.input)
        
        # Evaluate each claim for hallucination
        verdicts = self._evaluate_hallucinations(
            test_case.input,  # keywords
            test_case.actual_output,
            test_case.retrieval_context,
            json_claims
        )
        
        # Calculate hallucination score (1.0 = no hallucinations, 0.0 = all hallucinations)
        self.score = self._calculate_hallucination_score(verdicts)
        
        # Generate detailed reason
        if self.include_reason:
            hallucinated_claims = [v["claim"] for v in verdicts if v["verdict"] == "yes"]
            supported_claims = [v["claim"] for v in verdicts if v["verdict"] == "no"]
            
            self.reason = self._generate_detailed_reason(
                test_case.input, self.score, hallucinated_claims, supported_claims
            )
        else:
            self.reason = f"Hallucination score: {self.score:.2f}"
        
        self.success = self.score >= self.threshold
        return self.score

    def _extract_json_claims(self, json_output: Dict[str, Any], keywords: str) -> List[Dict[str, Any]]:
        """Extract factual claims from JSON output for hallucination evaluation"""
        claims = []
        keywords_list = keywords.split() if isinstance(keywords, str) else []
        
        def extract_claims(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    field_path = f"{prefix}.{key}" if prefix else key
                    
                    if isinstance(value, (str, int, float, bool)):
                        # Skip structural fields that are less likely to contain factual claims
                        if not self._is_structural_field(key):
                            claims.append({
                                "field": field_path,
                                "field_name": key,
                                "claim": str(value),
                                "type": type(value).__name__,
                                "claim_type": self._classify_claim_type(key, value),
                                "keywords_relation": self._assess_keyword_relation(str(value), keywords_list)
                            })
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            if isinstance(item, (str, int, float, bool)):
                                claims.append({
                                    "field": f"{field_path}[{i}]",
                                    "field_name": key,
                                    "claim": str(item),
                                    "type": type(item).__name__,
                                    "claim_type": self._classify_claim_type(key, item),
                                    "keywords_relation": self._assess_keyword_relation(str(item), keywords_list)
                                })
                            else:
                                extract_claims(item, f"{field_path}[{i}]")
                    else:
                        extract_claims(value, field_path)
        
        extract_claims(json_output)
        return claims

    def _is_structural_field(self, field_name: str) -> bool:
        """Check if field is structural (less likely to contain factual claims)"""
        structural_fields = {"id", "type", "category", "status", "level", "priority", "index"}
        return field_name.lower() in structural_fields

    def _classify_claim_type(self, field_name: str, value: Any) -> str:
        """Classify the type of claim for targeted hallucination detection"""
        field_lower = field_name.lower()
        
        claim_types = {
            "factual": ["author", "creator", "inventor", "founder", "date", "year", "location", "place"],
            "descriptive": ["title", "description", "summary", "content", "text"],
            "quantitative": ["count", "number", "amount", "size", "duration", "score", "rating"],
            "categorical": ["category", "type", "genre", "classification", "topic", "subject"]
        }
        
        for claim_type, keywords in claim_types.items():
            if any(keyword in field_lower for keyword in keywords):
                return claim_type
        
        return "general"

    def _assess_keyword_relation(self, claim: str, keywords: List[str]) -> float:
        """Assess how the claim relates to input keywords"""
        if not keywords:
            return 0.0
        
        claim_lower = claim.lower()
        matches = sum(1 for kw in keywords if kw.lower() in claim_lower and len(kw) > 2)
        return matches / len(keywords)

    def _evaluate_hallucinations(self, keywords: str, json_output: str,
                                retrieval_context: List[str], 
                                json_claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate each JSON claim for potential hallucination"""
        verdicts = []
        context_text = " ".join(retrieval_context).lower()
        
        for claim_info in json_claims:
            is_hallucination = self._detect_hallucination(
                claim_info, context_text, keywords
            )
            
            verdicts.append({
                "verdict": "yes" if is_hallucination else "no",
                "json_field": claim_info["field"],
                "claim": claim_info["claim"],
                "reason": self._generate_claim_reason(claim_info, is_hallucination, context_text)
            })
        
        return verdicts

    def _detect_hallucination(self, claim_info: Dict[str, Any], 
                            context_text: str, keywords: str) -> bool:
        """Detect if a claim is likely hallucinated"""
        claim_lower = claim_info["claim"].lower()
        
        # Check for direct contradiction
        if self._contradicts_context(claim_lower, context_text):
            return True
        
        # Check for specific factual claims not in context
        if claim_info["claim_type"] == "factual":
            if not self._is_supported_by_context(claim_lower, context_text):
                # Allow general terms related to keywords
                if claim_info["keywords_relation"] < 0.3:
                    return True
        
        # Check for specific numbers/dates not in context
        if claim_info["claim_type"] == "quantitative":
            if claim_info["claim"].isdigit() and claim_info["claim"] not in context_text:
                return True
        
        # Check for specific names not in context
        if self._is_specific_name(claim_info["claim"]) and claim_lower not in context_text:
            return True
        
        return False

    def _contradicts_context(self, claim: str, context_text: str) -> bool:
        """Check if claim directly contradicts context"""
        # Simple contradiction detection - can be enhanced
        contradiction_patterns = [
            ("not", "is"), ("false", "true"), ("incorrect", "correct"),
            ("never", "always"), ("impossible", "possible")
        ]
        
        for neg, pos in contradiction_patterns:
            if neg in claim and pos in context_text:
                return True
            if pos in claim and neg in context_text:
                return True
        
        return False

    def _is_supported_by_context(self, claim: str, context_text: str) -> bool:
        """Check if claim is supported by context"""
        claim_words = claim.split()
        
        # Check for partial matches with context
        matches = sum(1 for word in claim_words if len(word) > 3 and word in context_text)
        support_ratio = matches / len(claim_words) if claim_words else 0
        
        return support_ratio > 0.4

    def _is_specific_name(self, claim: str) -> bool:
        """Check if claim appears to be a specific name"""
        # Simple heuristic for detecting proper names
        words = claim.split()
        if len(words) <= 3 and all(word.istitle() for word in words):
            return True
        return False

    def _generate_claim_reason(self, claim_info: Dict[str, Any], 
                             is_hallucination: bool, context_text: str) -> str:
        """Generate reason for hallucination verdict"""
        if is_hallucination:
            if claim_info["claim_type"] == "factual":
                return f"Claim '{claim_info['claim']}' appears to be a specific factual detail not supported by the retrieval context."
            elif self._is_specific_name(claim_info["claim"]):
                return f"Specific name '{claim_info['claim']}' is not mentioned in the retrieval context."
            elif claim_info["claim"].isdigit():
                return f"Specific number '{claim_info['claim']}' is not found in the retrieval context."
            else:
                return f"Claim '{claim_info['claim']}' lacks sufficient support from the retrieval context."
        else:
            if claim_info["claim"].lower() in context_text:
                return f"Claim '{claim_info['claim']}' is directly supported by the retrieval context."
            else:
                return f"Claim '{claim_info['claim']}' is consistent with the general information in the retrieval context."

    def _calculate_hallucination_score(self, verdicts: List[Dict[str, Any]]) -> float:
        """Calculate hallucination score (1.0 = no hallucinations, 0.0 = all hallucinations)"""
        if not verdicts:
            return 1.0
        
        if self.strict_mode:
            # Strict mode: any hallucination results in score 0
            return 0.0 if any(v["verdict"] == "yes" for v in verdicts) else 1.0
        else:
            # Proportional scoring
            hallucination_count = sum(1 for v in verdicts if v["verdict"] == "yes")
            return 1.0 - (hallucination_count / len(verdicts))

    def _generate_detailed_reason(self, keywords: str, score: float,
                                hallucinated_claims: List[str], 
                                supported_claims: List[str]) -> str:
        """Generate detailed explanation for the hallucination score"""
        total_claims = len(hallucinated_claims) + len(supported_claims)
        
        reason = f"The hallucination score is {score:.2f} because {len(supported_claims)} out of {total_claims} claims in the JSON output are supported by the retrieval context. "
        
        if hallucinated_claims:
            reason += f"Detected {len(hallucinated_claims)} potential hallucination(s): "
            if len(hallucinated_claims) <= 3:
                reason += f"{', '.join(hallucinated_claims)}. "
            else:
                reason += f"{', '.join(hallucinated_claims[:3])}, and {len(hallucinated_claims)-3} others. "
        
        if supported_claims:
            reason += f"Well-supported claims include content that aligns with the retrieval context for keywords '{keywords}'."
        
        return reason

    def is_successful(self) -> bool:
        """Check if the metric passed the threshold"""
        return hasattr(self, 'success') and self.success
