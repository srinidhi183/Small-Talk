# faithfulness_eval.py
# Ensure you use the correct import for your LLM call function, 
# e.g., from llm_handler import get_llm_response, or from llm_utils.
# Also, manage OPENAI_API_KEY as per your project's setup (e.g., from config.py [4]).

from llm_handler import get_llm_response # Or your chosen LLM interaction module [1, 5]

def extract_claims_from_output(text_to_analyze, api_key):
    """
    Uses an LLM to extract distinct factual claims from the LLM's output.
    """
    prompt = (f"Analyze the following text and extract all distinct factual claims. "
              f"Return these claims as a numbered list.\n\nText: \"{text_to_analyze}\"\n\nClaims:\n")
    response_str = get_llm_response(prompt, api_key)
    # Parse the response_str into a list of claims.
    # Robust parsing (e.g., expecting JSON from LLM) is recommended for production.
    claims = [claim.strip() for claim in response_str.splitlines() if claim.strip() and claim[0].isdigit()]
    return [claim.split('.', 1)[1].strip() if '.' in claim else claim for claim in claims]


def verify_claim_against_context(claim, context, api_key):
    """
    Uses an LLM to verify if a single claim is supported by the provided context.
    Returns 'yes' (supported), 'no' (contradicted/not supported), or 'idk' (insufficient info in context).
    """
    prompt = (f"Given the following context:\n\n---\n{context}\n---\n\n"
              f"Is the following claim supported by the context? "
              f"Answer ONLY with 'yes', 'no', or 'idk'.\n\nClaim: \"{claim}\"\n\nAnswer:")
    response = get_llm_response(prompt, api_key).strip().lower()
    if response in ['yes', 'no', 'idk']:
        return response
    return 'idk' # Default for ambiguous LLM responses


def calculate_faithfulness_score(llm_output, retrieved_context, api_key):
    """
    Calculates the faithfulness score using a Question-Answer Generation (QAG) like approach [10].
    The score is the proportion of claims in the llm_output that are factually aligned with the retrieved_context.
    """
    if not llm_output: # If there is no output, it cannot be unfaithful.
        return 1.0 
    if not retrieved_context: # If there is output but no context, all claims are ungrounded.
        return 0.0

    claims = extract_claims_from_output(llm_output, api_key)
    if not claims:
        # If output exists but no claims extracted, interpretation might vary.
        # Could be 1.0 (no unsupported claims) or indicate an issue with claim extraction.
        return 1.0 

    supported_claims_count = 0
    for claim in claims:
        verification_status = verify_claim_against_context(claim, retrieved_context, api_key)
        # According to [10], 'idk' (context does not contain relevant information) can be counted as not unfaithful.
        if verification_status == 'yes' or verification_status == 'idk':
            supported_claims_count += 1
    
    faithfulness_score = supported_claims_count / len(claims)
    return faithfulness_score

# --- Example of how you might call this ---
# if __name__ == '__main__':
#     from config import OPENAI_API_KEY # Your API key configuration [4]
#     example_output = "The sky is blue. Penguins live in Antarctica."
#     example_context = "The sky appears blue due to Rayleigh scattering. Penguins are flightless birds found in Antarctica."
#     score = calculate_faithfulness_score(example_output, example_context, OPENAI_API_KEY)
#     print(f"Faithfulness Score: {score}") # Expected: 1.0
#
#     example_output_unfaithful = "The moon is made of cheese."
#     score_unfaithful = calculate_faithfulness_score(example_output_unfaithful, example_context, OPENAI_API_KEY)
#     print(f"Faithfulness Score (Unfaithful): {score_unfaithful}") # Expected: 0.0 (if "moon is cheese" is one claim)
