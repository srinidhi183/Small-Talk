#answer_relevency_calculate.py
import json
from typing import List, Dict, Any, Optional
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class KeywordJSONAnswerRelevancyTemplate:
    @staticmethod
    def generate_verdicts(keywords: str, json_output: str):
        return f"""Given the input keywords and JSON output, evaluate whether each JSON field is relevant and serves its intended purpose.

**
IMPORTANT: Please make sure to only return in JSON format.
For each JSON field, evaluate:
1. Is the field content relevant to the input keywords?
2. Does the field serve its intended purpose?
3. Is the information appropriately structured for the keywords?

Example Keywords: "machine learning algorithms neural networks"
Example JSON Output: {{"title": "ML Guide", "topics": ["neural networks", "algorithms"], "irrelevant_field": "cooking recipes"}}

Example:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "json_field": "title",
            "content": "ML Guide",
            "purpose": "Content title",
            "reason": "Title 'ML Guide' is directly relevant to machine learning keywords."
        }},
        {{
            "verdict": "yes",
            "json_field": "topics",
            "content": ["neural networks", "algorithms"],
            "purpose": "Topic categorization", 
            "reason": "Topics directly match the input keywords 'neural networks' and 'algorithms'."
        }},
        {{
            "verdict": "no",
            "json_field": "irrelevant_field",
            "content": "cooking recipes",
            "purpose": "Unknown purpose",
            "reason": "Content about 'cooking recipes' is completely unrelated to machine learning keywords."
        }}
    ]
}}
**

Keywords: {keywords}
JSON Output: {json_output}

JSON:
"""

    @staticmethod
    def generate_reason(keywords: str, score: float, relevant_fields: List[str], irrelevant_fields: List[str]):
        return f"""Based on the keywords and JSON field relevancy analysis, provide a concise reason for the answer relevancy score.

**
IMPORTANT: Please make sure to only return in JSON format.
Example JSON:
{{
    "reason": "The score is <answer_relevancy_score> because <your_reason>. The JSON output contains <relevant_count> relevant fields and <irrelevant_count> irrelevant fields to the given keywords."
}}
**

Keywords: {keywords}
Answer Relevancy Score: {score}
Relevant Fields: {relevant_fields}
Irrelevant Fields: {irrelevant_fields}

JSON:
"""

class KeywordJSONAnswerRelevancyMetric(BaseMetric):
    def __init__(self, threshold: float = 0.7, model: str = "gpt-4o", 
                 include_reason: bool = True, strict_mode: bool = False):
        self.threshold = threshold
        self.model = model
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.evaluation_template = KeywordJSONAnswerRelevancyTemplate
        
    def measure(self, test_case: LLMTestCase) -> float:
        """
        Measure answer relevancy of JSON output against input keywords
        """
        # Validate JSON output
        try:
            json_output = json.loads(test_case.actual_output)
        except json.JSONDecodeError as e:
            self.score = 0.0
            self.reason = f"Output is not valid JSON. Error: {str(e)}"
            self.success = False
            return self.score

        # Extract and analyze JSON fields
        field_analysis = self._extract_json_fields(json_output, test_case.input)
        
        # Generate evaluation verdicts
        verdicts = self._evaluate_field_relevancy(
            test_case.input, 
            test_case.actual_output,
            field_analysis
        )
        
        # Calculate relevancy score
        self.score = self._calculate_relevancy_score(verdicts)
        
        # Generate detailed reason
        if self.include_reason:
            relevant_fields = [v["json_field"] for v in verdicts if v["verdict"] == "yes"]
            irrelevant_fields = [v["json_field"] for v in verdicts if v["verdict"] == "no"]
            
            self.reason = self._generate_detailed_reason(
                test_case.input, self.score, relevant_fields, irrelevant_fields
            )
        else:
            self.reason = f"Answer relevancy score: {self.score:.2f}"
        
        self.success = self.score >= self.threshold
        return self.score

    def _extract_json_fields(self, json_output: Dict[str, Any], keywords: str) -> List[Dict[str, Any]]:
        """Extract all fields from JSON output for evaluation"""
        fields = []
        keywords_list = keywords.split() if isinstance(keywords, str) else []
        
        def extract_fields(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    field_path = f"{prefix}.{key}" if prefix else key
                    
                    if isinstance(value, (str, int, float, bool)):
                        fields.append({
                            "field_path": field_path,
                            "field_name": key,
                            "content": str(value),
                            "type": type(value).__name__,
                            "purpose": self._infer_field_purpose(key, value),
                            "keyword_relevance": self._assess_keyword_relevance(str(value), keywords_list)
                        })
                    elif isinstance(value, list):
                        if value and isinstance(value[0], (str, int, float, bool)):
                            fields.append({
                                "field_path": field_path,
                                "field_name": key,
                                "content": value,
                                "type": "list",
                                "purpose": self._infer_field_purpose(key, value),
                                "keyword_relevance": self._assess_keyword_relevance(str(value), keywords_list)
                            })
                        else:
                            extract_fields(value, field_path)
                    else:
                        extract_fields(value, field_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_fields(item, f"{prefix}[{i}]")
        
        extract_fields(json_output)
        return fields

    def _infer_field_purpose(self, field_name: str, value: Any) -> str:
        """Infer the purpose of a JSON field based on its name and content"""
        field_lower = field_name.lower()
        
        purpose_mapping = {
            "title": "Content title/heading",
            "description": "Content description",
            "summary": "Content summary", 
            "abstract": "Content abstract",
            "content": "Main content body",
            "text": "Textual content",
            "body": "Main body text",
            "author": "Author identification",
            "creator": "Content creator",
            "date": "Temporal information",
            "time": "Temporal information",
            "category": "Content categorization",
            "type": "Content type classification",
            "tags": "Content tags/keywords",
            "keywords": "Content keywords",
            "topics": "Topic classification",
            "subject": "Subject matter",
            "url": "Resource location",
            "link": "Resource link",
            "id": "Unique identifier",
            "name": "Entity name",
            "status": "Status information",
            "level": "Difficulty/importance level",
            "priority": "Priority classification",
            "score": "Scoring/rating",
            "rating": "Content rating"
        }
        
        for key, purpose in purpose_mapping.items():
            if key in field_lower:
                return purpose
        
        return f"General field: {field_name}"

    def _assess_keyword_relevance(self, content: str, keywords: List[str]) -> float:
        """Assess how relevant content is to the input keywords"""
        if not keywords:
            return 0.0
        
        content_lower = content.lower()
        matches = sum(1 for kw in keywords if kw.lower() in content_lower and len(kw) > 2)
        return matches / len(keywords)

    def _evaluate_field_relevancy(self, keywords: str, json_output: str, 
                                field_analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate relevancy of each JSON field to the keywords"""
        verdicts = []
        
        for field_info in field_analysis:
            # Simple relevancy assessment (replace with LLM call in production)
            is_relevant = field_info["keyword_relevance"] > 0.2 or self._is_structurally_relevant(
                field_info["field_name"], keywords
            )
            
            verdicts.append({
                "verdict": "yes" if is_relevant else "no",
                "json_field": field_info["field_path"],
                "content": field_info["content"],
                "purpose": field_info["purpose"],
                "reason": self._generate_field_reason(field_info, is_relevant, keywords)
            })
        
        return verdicts

    def _is_structurally_relevant(self, field_name: str, keywords: str) -> bool:
        """Check if field is structurally relevant even without direct keyword matches"""
        structural_fields = ["id", "type", "category", "status", "date", "author"]
        return field_name.lower() in structural_fields

    def _generate_field_reason(self, field_info: Dict[str, Any], is_relevant: bool, keywords: str) -> str:
        """Generate reason for field relevancy verdict"""
        if is_relevant:
            if field_info["keyword_relevance"] > 0:
                return f"Field '{field_info['field_name']}' contains content relevant to keywords: {keywords}"
            else:
                return f"Field '{field_info['field_name']}' provides structural relevance for the JSON output"
        else:
            return f"Field '{field_info['field_name']}' content is not related to the input keywords: {keywords}"

    def _calculate_relevancy_score(self, verdicts: List[Dict[str, Any]]) -> float:
        """Calculate overall relevancy score based on field verdicts"""
        if not verdicts:
            return 0.0
        
        if self.strict_mode:
            # Strict mode: all fields must be relevant
            return 1.0 if all(v["verdict"] == "yes" for v in verdicts) else 0.0
        else:
            # Proportional scoring
            relevant_count = sum(1 for v in verdicts if v["verdict"] == "yes")
            return relevant_count / len(verdicts)

    def _generate_detailed_reason(self, keywords: str, score: float, 
                                relevant_fields: List[str], irrelevant_fields: List[str]) -> str:
        """Generate detailed explanation for the relevancy score"""
        total_fields = len(relevant_fields) + len(irrelevant_fields)
        
        reason = f"The answer relevancy score is {score:.2f} because {len(relevant_fields)} out of {total_fields} JSON fields are relevant to the keywords '{keywords}'. "
        
        if relevant_fields:
            if len(relevant_fields) <= 3:
                reason += f"Relevant fields: {', '.join(relevant_fields)}. "
            else:
                reason += f"Relevant fields include: {', '.join(relevant_fields[:3])}, and {len(relevant_fields)-3} others. "
        
        if irrelevant_fields:
            if len(irrelevant_fields) <= 3:
                reason += f"Irrelevant fields: {', '.join(irrelevant_fields)}. "
            else:
                reason += f"Irrelevant fields include: {', '.join(irrelevant_fields[:3])}, and {len(irrelevant_fields)-3} others. "
        
        return reason

    def is_successful(self) -> bool:
        """Check if the metric passed the threshold"""
        return hasattr(self, 'success') and self.success
