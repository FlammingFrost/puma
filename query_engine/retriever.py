# TODO: Implement this module
from embedding.vector_store import VectorStore

# Assuming we have a VectorStore class that interacts with the vector DB
vector_store = VectorStore()

def retrieve_relevant_chunks(query: str, top_k: int = 5) -> list:
    """
    Retrieves the most relevant code snippets based on the query.
    
    Returns:
        List of tuples [(code_chunk, metadata)]
    """
    results = vector_store.query(query, top_k=top_k)
    print(f"#Log query_engine/retriever.py# [Retrieved Chunks]: {results}")
    
    retrieved_chunks = [
        {
            "code": result["text"], 
            "filename": result["metadata"]["filename"],
            "line_number": result["metadata"]["line_number"]
        } 
        for result in results
    ]

    return retrieved_chunks