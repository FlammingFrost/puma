# Query Engine
Handles query processing using LangChain:
- Retrieves relevant chunks from the vector database.
- Generates responses based on retrieved content.

## `retriever.py`
The file contains the class `VBRetriever` handles the request to **retrieve** relevant chunks from the vector database.
Attributes:
- `vector_db`: The vector database instance.

## `generator.py`
`generator.generate_response` provides a framework to generate responses based on the retrieved content.
Built with LangChain, it calls api to generate responses.