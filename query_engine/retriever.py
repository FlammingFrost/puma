# TODO: Implement this module
from embedding.vector_store import VectorBase, DummyVB
from configs.configurator import config

class VBRetriever:
    def __init__(self, vector_store_path: str = None, embedding_func = None, conf = None):
        """
        Initializes the VectorBase retriever.
        **Note**: The VectorBase is a dummy implementation for now.
        
        Args:
            config (dict): Configuration settings.
                top_k (int): Number of code snippets to retrieve.
                
        """
        try:
            self.VB = VectorBase(vector_store_path, embedding_func)
        except NotImplementedError as e:
            print(f"Error initializing VectorBase: {e}, using DummyVB instead.")
            self.VB = DummyVB()
        except Exception as e:
            print(f"Error initializing VectorBase: {e}, using DummyVB instead.")
            self.VB = DummyVB()
        
        if conf is None:
            conf = config
        
        self.top_k = conf["vectorbase_top_k"]
        self.name = conf.get("retriever_name", "VBRetriever")

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
                "file_name": result["metadata"]["file_name"],
                "file_path": result["metadata"]["file_path"],
                "line_number_start": result["metadata"]["line_number_start"],
                "line_number_end": result["metadata"]["line_number_end"],
            } 
            for result in results
        ]

        return retrieved_chunks

    def __str__(self):
        return 'Retriever: ' + self.name + '\n' + 'VectorBase: ' + str(self.VB)+ '--> Top K: ' + str(self.top_k)