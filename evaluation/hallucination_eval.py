# hallucination_eval.py
# This can leverage parts of the faithfulness logic.
# For instance, a hallucination can be defined as 1 - faithfulness_score,
# or more specifically, the proportion of claims that are *not* supported by the context.

from llm_handler import get_llm_response # Or your chosen LLM interaction module [1, 5]
# You might import functions from faithfulness_eval.py if you structure it for reusability
# from faithfulness_eval import extract_claims_from_output, verify_claim_against_context


# For this standalone example, we'll include simplified versions or re-define.
def _extract_claims(text_to_analyze, api_key): # Underscore to denote internal helper
    """Simplified claim extraction for hallucination context."""
    prompt = (f"Extract all distinct factual claims from the following text. "
              f"Return claims as a numbered list.\n\nText: \"{text_to_analyze}\"\n\nClaims:\n")
    response_str = get_llm_response(prompt, api_key)
    claims = [claim.strip() for claim in response_str.splitlines() if claim.strip() and claim[0].isdigit()]
    return [claim.split('.', 1)[1].strip() if '.' in claim else claim for claim in claims]

def _verify_claim(claim, context, api_key): # Underscore to denote internal helper
    """Simplified claim verification for hallucination context."""
    prompt = (f"Context:\n---\n{context}\n---\n\nClaim: \"{claim}\"\n\n"
              f"Is the claim supported by the context? Answer ONLY with 'yes', 'no', or 'idk'.\n\nAnswer:")
    response = get_llm_response(prompt, api_key).strip().lower()
    if response in ['yes', 'no', 'idk']: return response
    return 'idk'

def calculate_hallucination_score(llm_output, retrieved_context, api_key):
    """
    Calculates a hallucination score.
    This score is the proportion of claims in llm_output that are either contradicted by ('no')
    or not found ('idk') in the retrieved_context.
    """
    if not llm_output: # No output means no hallucination.
        return 0.0
    
    claims = _extract_claims(llm_output, api_key)
    
    if not claims: # No claims extracted from existing output.
        return 0.0 # Could also be an error state or 1.0 if output was non-factual.

    if not retrieved_context: # If output exists but no context, all claims are ungrounded/hallucinated.
        return 1.0 if claims else 0.0

    hallucinated_claims_count = 0
    for claim in claims:
        verification_status = _verify_claim(claim, retrieved_context, api_key)
        # A claim is considered a hallucination if it's contradicted ('no')
        # or if the context doesn't contain information to verify it ('idk') [17].
        if verification_status == 'no' or verification_status == 'idk':
            hallucinated_claims_count += 1
            
    hallucination_rate = hallucinated_claims_count / len(claims)
    return hallucination_rate

# --- Example of how you might call this ---
# if __name__ == '__main__':
#     from config import OPENAI_API_KEY
#     example_output = "The sun revolves around the Earth. Water boils at 50 degrees Celsius."
#     example_context = "The Earth revolves around the sun. Water's boiling point at standard pressure is 100 degrees Celsius."
#     score = calculate_hallucination_score(example_output, example_context, OPENAI_API_KEY)
#     print(f"Hallucination Score: {score}") # Expected: 1.0 (both claims are hallucinated against context)
#
#     example_output_mixed = "Paris is in France. The Eiffel Tower is made of wood."
#     example_context_mixed = "Paris is the capital city of France. The Eiffel Tower is a wrought-iron lattice tower."
#     score_mixed = calculate_hallucination_score(example_output_mixed, example_context_mixed, OPENAI_API_KEY)
#     # Expected: 0.5 (one claim hallucinated, one not)
#     print(f"Hallucination Score (Mixed): {score_mixed}")
