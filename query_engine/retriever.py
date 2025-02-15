# TODO: Implement this module
from embedding.vector_store import VectorBase, DummyVB
class VBRetriever:
    def __init__(self, config: dict):
        """
        Initializes the VectorBase retriever.
        **Note**: The VectorBase is a dummy implementation for now.
        
        Args:
            config (dict): Configuration settings.
                top_k (int): Number of code snippets to retrieve.
                
        """
        try:
            selVB = VectorBase()
        except NotImplementedError as e:
            print(f"Error initializing VectorBase: {e}, using DummyVB instead.")
            self.VB = DummyVB()
        except TypeError as e:
            print(f"Error initializing VectorBase: {e}, using DummyVB instead.")
            self.VB = DummyVB()
        
        self.top_k = config["vectorbase_top_k"]
        self.name = config.get("retriever_name", "VBRetriever")

    def retrieve_relevant_chunks(self, query: str) -> list:
        """
        Retrieves the most relevant code snippets based on the query.
        
        Args:
            query (str): The user query.
        
        Returns:
            List of tuples [(code_chunk, metadata)]
        """
        print(f"#Log query_engine/retriever.py# [Query Received]: {query}, \n Retrieving top {self.top_k} chunks...")
        results = self.VB.retrieve(query, top_k = self.top_k)
        
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

    def __str__(self):
        return 'Retriever: ' + self.name + '\n' + 'VectorBase: ' + str(self.VB)+ '--> Top K: ' + str(self.top_k)