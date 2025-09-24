#engagement_calculate.py
import json
from typing import List, Dict, Any
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class KeywordJSONEngagementTemplate:
    @staticmethod
    def generate_verdicts(keywords: str, json_output: str):
        return f"""Evaluate how engaging and compelling each element in the JSON output is for the given keywords.

**
IMPORTANT: Please make sure to only return in JSON format.
For each JSON field, evaluate:
1. Does the content capture attention and maintain interest?
2. Is the content compelling and emotionally engaging?
3. Does it use engaging writing techniques (vivid language, questions, examples)?
4. Is the content relevant and meaningful to the audience interested in these keywords?
5. Does it motivate further exploration or action?

Engagement factors to consider:
- Interest level and attention-grabbing content
- Emotional connection and appeal
- Use of compelling language and examples
- Clear value proposition
- Motivational or inspirational elements
- Storytelling or narrative elements

Example:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "json_field": "title",
            "content": "Master AI: Transform Your Career with Machine Learning",
            "engagement_factors": ["compelling_language", "value_proposition"],
            "reason": "Title uses action-oriented language 'Master' and 'Transform' with clear benefit, making it highly engaging for ML learners."
        }},
        {{
            "verdict": "no",
            "json_field": "description",
            "content": "Basic information about algorithms",
            "engagement_factors": ["lacks_specificity"],
            "reason": "Description is generic and vague, lacks compelling details or benefits that would engage the audience."
        }}
    ]
}}
**

Keywords: {keywords}
JSON Output: {json_output}

JSON:
"""

    @staticmethod
    def generate_reason(keywords: str, score: float, engaging_fields: List[str], 
                       non_engaging_fields: List[str]):
        return f"""Based on the keywords and JSON engagement analysis, provide a concise reason for the engagement score.

**
IMPORTANT: Please make sure to only return in JSON format.
Example JSON:
{{
    "reason": "The score is <engagement_score> because <your_reason>. The JSON output contains <engaging_count> engaging fields and <non_engaging_count> non-engaging fields."
}}
**

Keywords: {keywords}
Engagement Score: {score}
Engaging Fields: {engaging_fields}
Non-Engaging Fields: {non_engaging_fields}

JSON:
"""

class KeywordJSONEngagementMetric(BaseMetric):
    def __init__(self, threshold: float = 0.7, model: str = "gpt-4o", 
                 include_reason: bool = True, strict_mode: bool = False):
        self.threshold = threshold
        self.model = model
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.evaluation_template = KeywordJSONEngagementTemplate

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Measure engagement level of JSON output relative to input keywords
        """
        # Validate JSON output
        try:
            json_output = json.loads(test_case.actual_output)
        except json.JSONDecodeError as e:
            self.score = 0.0
            self.reason = f"Output is not valid JSON. Error: {str(e)}"
            self.success = False
            return self.score

        # Extract fields for engagement evaluation
        field_analysis = self._extract_json_fields(json_output, test_case.input)

        # Evaluate engagement in each field
        verdicts = self._evaluate_engagement(
            test_case.input,  # keywords
            test_case.actual_output,
            field_analysis
        )

        # Calculate engagement score
        self.score = self._calculate_engagement_score(verdicts)

        # Generate detailed reason
        if self.include_reason:
            engaging_fields = [v["json_field"] for v in verdicts if v["verdict"] == "yes"]
            non_engaging_fields = [v["json_field"] for v in verdicts if v["verdict"] == "no"]
            
            self.reason = self._generate_detailed_reason(
                test_case.input, self.score, engaging_fields, non_engaging_fields, verdicts
            )
        else:
            self.reason = f"Engagement score: {self.score:.2f}"

        self.success = self.score >= self.threshold
        return self.score

    def _extract_json_fields(self, json_output: Dict[str, Any], keywords: str) -> List[Dict[str, Any]]:
        """Extract content fields from JSON for engagement evaluation"""
        fields = []
        keywords_list = keywords.split() if isinstance(keywords, str) else []

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
                                "field_importance": self._assess_field_importance(key),
                                "content_type": self._classify_content_type(key)
                            })
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            if isinstance(item, str) and len(item.strip()) > 0:
                                fields.append({
                                    "field_path": f"{field_path}[{i}]",
                                    "field_name": key,
                                    "content": str(item),
                                    "field_importance": self._assess_field_importance(key),
                                    "content_type": self._classify_content_type(key)
                                })
                            elif isinstance(item, dict):
                                extract_fields(item, f"{field_path}[{i}]")
                    else:
                        extract_fields(value, field_path)

        extract_fields(json_output)
        return fields

    def _assess_field_importance(self, field_name: str) -> float:
        """Assess importance of field for engagement (higher importance = more weight)"""
        field_lower = field_name.lower()
        
        importance_weights = {
            "title": 0.9, "headline": 0.9, "name": 0.8,
            "description": 0.8, "summary": 0.7, "overview": 0.7,
            "content": 0.8, "text": 0.6, "body": 0.7,
            "introduction": 0.7, "conclusion": 0.6,
            "example": 0.6, "benefit": 0.7, "feature": 0.6,
            "id": 0.1, "index": 0.1, "count": 0.2
        }
        
        for key, weight in importance_weights.items():
            if key in field_lower:
                return weight
        
        return 0.5  # Default importance

    def _classify_content_type(self, field_name: str) -> str:
        """Classify content type for targeted engagement evaluation"""
        field_lower = field_name.lower()
        
        if any(term in field_lower for term in ["title", "headline", "name"]):
            return "title"
        elif any(term in field_lower for term in ["description", "summary", "overview", "about"]):
            return "descriptive"
        elif any(term in field_lower for term in ["content", "text", "body", "detail"]):
            return "content"
        elif any(term in field_lower for term in ["example", "sample", "demo"]):
            return "example"
        elif any(term in field_lower for term in ["benefit", "advantage", "feature"]):
            return "value_proposition"
        
        return "general"

    def _evaluate_engagement(self, keywords: str, json_output: str,
                           field_analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate engagement level of each JSON field"""
        verdicts = []

        for field_info in field_analysis:
            engagement_analysis = self._analyze_engagement_factors(
                field_info["content"], 
                field_info["content_type"],
                keywords
            )
            
            is_engaging = engagement_analysis["engagement_score"] >= 0.6
            
            verdicts.append({
                "verdict": "yes" if is_engaging else "no",
                "json_field": field_info["field_path"],
                "content": field_info["content"],
                "engagement_factors": engagement_analysis["factors"],
                "reason": self._generate_field_engagement_reason(
                    field_info, engagement_analysis, is_engaging, keywords
                )
            })

        return verdicts

    def _analyze_engagement_factors(self, content: str, content_type: str, keywords: str) -> Dict[str, Any]:
        """Analyze various engagement factors in content"""
        content_lower = content.lower()
        
        factors = []
        engagement_score = 0.0
        
        # Action words and compelling language
        action_words = ["master", "transform", "discover", "unlock", "achieve", "boost", "enhance", "revolutionize"]
        if any(word in content_lower for word in action_words):
            factors.append("action_oriented")
            engagement_score += 0.2
        
        # Emotional appeal
        emotional_words = ["amazing", "incredible", "fantastic", "exciting", "powerful", "breakthrough", "innovative"]
        if any(word in content_lower for word in emotional_words):
            factors.append("emotional_appeal")
            engagement_score += 0.15
        
        # Value proposition
        value_words = ["benefit", "advantage", "improve", "better", "faster", "easier", "effective", "results"]
        if any(word in content_lower for word in value_words):
            factors.append("value_proposition")
            engagement_score += 0.2
        
        # Specificity and concrete details
        if any(char.isdigit() for char in content) and len(content.split()) > 3:
            factors.append("specific_details")
            engagement_score += 0.15
        
        # Questions or interactive elements
        if "?" in content or any(word in content_lower for word in ["how", "what", "why", "when", "where"]):
            factors.append("interactive_elements")
            engagement_score += 0.1
        
        # Keywords relevance (engagement through relevance)
        keyword_words = keywords.lower().split()
        keyword_matches = sum(1 for kw in keyword_words if kw in content_lower and len(kw) > 2)
        if keyword_matches > 0:
            factors.append("keyword_relevant")
            engagement_score += min(0.2, keyword_matches * 0.05)
        
        # Length and readability
        word_count = len(content.split())
        if content_type == "title" and 3 <= word_count <= 8:
            factors.append("optimal_length")
            engagement_score += 0.1
        elif content_type == "descriptive" and 10 <= word_count <= 50:
            factors.append("optimal_length")
            engagement_score += 0.1
        
        # Check for generic/boring language
        generic_phrases = ["basic", "simple", "standard", "regular", "normal", "ordinary"]
        if any(phrase in content_lower for phrase in generic_phrases):
            factors.append("generic_language")
            engagement_score -= 0.1
        
        return {
            "engagement_score": min(1.0, max(0.0, engagement_score)),
            "factors": factors
        }

    def _generate_field_engagement_reason(self, field_info: Dict[str, Any], 
                                        engagement_analysis: Dict[str, Any], 
                                        is_engaging: bool, keywords: str) -> str:
        """Generate reason for engagement verdict"""
        if is_engaging:
            factors = engagement_analysis["factors"]
            positive_factors = [f for f in factors if f not in ["generic_language"]]
            
            if positive_factors:
                return f"Field '{field_info['field_name']}' is engaging due to: {', '.join(positive_factors)}. Content effectively captures attention and relates to keywords '{keywords}'."
            else:
                return f"Field '{field_info['field_name']}' shows moderate engagement with relevant content for keywords '{keywords}'."
        else:
            factors = engagement_analysis["factors"]
            if "generic_language" in factors:
                return f"Field '{field_info['field_name']}' lacks engagement due to generic language and limited compelling elements."
            else:
                return f"Field '{field_info['field_name']}' needs more engaging elements like action words, emotional appeal, or specific benefits to capture audience attention."

    def _calculate_engagement_score(self, verdicts: List[Dict[str, Any]]) -> float:
        """Calculate overall engagement score with field importance weighting"""
        if not verdicts:
            return 0.0

        if self.strict_mode:
            return 1.0 if all(v["verdict"] == "yes" for v in verdicts) else 0.0
        else:
            engaging_count = sum(1 for v in verdicts if v["verdict"] == "yes")
            return engaging_count / len(verdicts)

    def _generate_detailed_reason(self, keywords: str, score: float, 
                                engaging_fields: List[str], non_engaging_fields: List[str],
                                verdicts: List[Dict[str, Any]]) -> str:
        """Generate detailed explanation for engagement score"""
        total_fields = len(engaging_fields) + len(non_engaging_fields)
        
        reason = f"The engagement score is {score:.2f} because {len(engaging_fields)} out of {total_fields} JSON fields are engaging for the keywords '{keywords}'. "
        
        if engaging_fields:
            reason += f"Engaging fields include: {', '.join(engaging_fields[:3])}. "
        
        if non_engaging_fields:
            reason += f"Fields needing improvement: {', '.join(non_engaging_fields[:3])}. "
        
        # Identify common engagement factors
        all_factors = []
        for verdict in verdicts:
            if verdict["verdict"] == "yes":
                all_factors.extend(verdict.get("engagement_factors", []))
        
        if all_factors:
            common_factors = list(set(all_factors))[:3]
            reason += f"Strong engagement factors include: {', '.join(common_factors)}."
        
        return reason

    def is_successful(self) -> bool:
        """Check if the metric passed the threshold"""
        return hasattr(self, 'success') and self.success
