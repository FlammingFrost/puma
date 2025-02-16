# TODO: Implement this module
from query_engine.retriever import VBRetriever
from query_engine.generator import generate_response
from tools.logger import logger
from configs.configurator import config_loader

def process_query(user_query: str, retriever: VBRetriever) -> str:
    """
    Main function to process user queries.
    Steps:
    1. Retrieve relevant code snippets from vector database.
    2. Generate a response using an LLM (Claude/GPT).
    3. Return the formatted response.
    
    Args:
        user_query (str): The user query.
        retriever (VBRetriever): The VectorBase retriever.
        
    Returns:
        str: The response from the LLM.
    """
    logger.info(f"[Query Received]: {user_query}")

    config = config_loader.config

    # Retrieve relevant code chunks
    retrieved_chunks = retriever.retrieve_relevant_chunks(user_query)
    
    if not retrieved_chunks:
        return "No relevant code snippets found in the project."

    # Generate LLM response based on retrieved code
    response = generate_response(user_query, retrieved_chunks)
    logger.info(f"[Response Generated]: {response}")
    return response