# contextual_precision_eval.py
from llm_handler import get_llm_response # Or your chosen LLM interaction module [1, 5]

def check_context_chunk_relevance(chunk, user_query, llm_generated_response, api_key):
    """
    Uses an LLM to assess if a retrieved context chunk is relevant for addressing the user_query
    and for generating the llm_generated_response.
    Returns True if relevant, False otherwise.
    """
    # This prompt implements LLMContextPrecisionWithoutReference from Ragas [12]
    # by comparing the chunk to the user query and the generated response.
    prompt = (f"User Query: \"{user_query}\"\n"
              f"LLM's Generated Response: \"{llm_generated_response}\"\n\n"
              f"Retrieved Context Chunk:\n---\n{chunk}\n---\n\n"
              f"Is the Retrieved Context Chunk relevant and helpful for answering the User Query and "
              f"informing the LLM's Generated Response? Answer ONLY with 'yes' or 'no'.\n\nAnswer:")
    
    relevance_response = get_llm_response(prompt, api_key).strip().lower()
    return relevance_response == 'yes'

def calculate_contextual_precision_score(retrieved_context_chunks, user_query, llm_generated_response, api_key):
    """
    Calculates the contextual precision score.
    'retrieved_context_chunks' should be a list of text strings.
    The score is the proportion of retrieved chunks deemed relevant [12].
    """
    if not retrieved_context_chunks:
        return 0.0 # No context provided, so precision is 0.

    relevant_chunks_count = 0
    for chunk in retrieved_context_chunks:
        if check_context_chunk_relevance(chunk, user_query, llm_generated_response, api_key):
            relevant_chunks_count += 1
            
    contextual_precision = relevant_chunks_count / len(retrieved_context_chunks)
    return contextual_precision

# --- Example of how you might call this ---
# if __name__ == '__main__':
#     from config import OPENAI_API_KEY
#     example_query = "What are the benefits of exercise?"
#     example_response = "Exercise helps improve cardiovascular health and mood."
#     example_chunks = [
#         "Regular physical activity can improve your muscle strength and boost your endurance.", # Relevant
#         "The history of the internet began with the development of electronic computers in the 1950s.", # Irrelevant
#         "Exercise controls weight and combats health conditions like heart disease." # Relevant
#     ]
#     score = calculate_contextual_precision_score(example_chunks, example_query, example_response, OPENAI_API_KEY)
#     print(f"Contextual Precision Score: {score}") # Expected: 0.666... (2 out of 3 relevant)
