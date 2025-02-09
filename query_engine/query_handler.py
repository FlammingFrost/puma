# TODO: Implement this module
from retriever import retrieve_relevant_chunks
from generator import generate_response

def process_query(user_query: str):
    """
    Main function to process user queries.
    Steps:
    1. Retrieve relevant code snippets from vector database.
    2. Generate a response using an LLM (Claude/GPT).
    3. Return the formatted response.
    
    input: user_query (str)
    output: response (str)
    """
    print(f"#Log query_engine/query_handler.py# [Query Received]: {user_query}")

    # Retrieve relevant code chunks
    retrieved_chunks = retrieve_relevant_chunks(user_query)
    
    if not retrieved_chunks:
        return "No relevant code snippets found in the project."

    # Generate LLM response based on retrieved code
    response = generate_response(user_query, retrieved_chunks)
    
    return response