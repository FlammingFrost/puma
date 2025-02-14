# TODO: Implement this module
import chromadb
import os

from data_processing.chunker import chunk_file

from tools.logger_config import logger

module_implemented = False
META_DATA_FORMAT = {
    'chunked_document': str,
    'file_name': str,
    'file_path': str,
    'line_number_start': int,
    'line_number_end': int,
    'file_type': ['code', 'text', 'config', 'data'],
    'file_language': ['py', 'java', 'c', 'cpp', 'json', 'yaml', 'toml', 'md', 'txt'],
    'function_type': None,
    'function_name': None,
    'dependencies': None
}
    

class VectorBase:
    """
    A class to store and retrieve vectors.
    
    Attributes:
        collection: The collection to store the vectors.
        embedding_func: The embedding function to use to encode the text to a vector.
    """
    
    def __init__(self, vector_store_path: str, embedding_func):
        # TODO: Implement vector store initialization
        client = chromadb.PersistentClient(path=vector_store_path)
        try:
            self.collection = client.get_collection("vector_store")
        except chromadb.CollectionNotFoundError:
            self.collection = client.create_collection("vector_store")
            logger.info("VectorBase collection created.")
        logger.info("VectorBase initialized.")
        
        self.embedding_func = embedding_func
        
    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """
        Retrieve top k vectors from the vector store based on the query.
        
        Args:
            query: The query string.
            top_k: The number of vectors to retrieve.
            embedding_func: The embedding function to use to encode the query to a vector
        
        Returns:
            A list of dictionaries containing the retrieved vectors.
        """
        try:
            query_embedding = self.embedding_func(query)
        except Exception as e:
            logger.error(f"Error encoding query: {e}")
            raise e
        query_result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=None,
            filter=None
        )
        
        return query_result
    
    def insert(self, text: str, metadata: dict):
        """
        Insert a new vector to the vector store.
        
        Args:
            text: The text to store.
            metadata: The metadata to store.
            embedding_func: The embedding function to use to encode the text to a vector.
            
        Returns:
            None
        """
        text_embedding = self.embedding_func(text)
        # Check metadata format
        for key, value in metadata.items():
            if key not in META_DATA_FORMAT:
                raise ValueError(f"Missing metadata key: {key}")
            if isinstance(META_DATA_FORMAT[key], list):
                if value not in META_DATA_FORMAT[key]:
                    raise ValueError(f"Unsupported {key} value: {value}")
            elif not isinstance(value, META_DATA_FORMAT[key]):
                raise ValueError(f"Invalid {key} value type: {type(value)}")
            
        import hashlib
        id_str = f"{metadata['file_path']}:{metadata['line_number_start']}-{metadata['line_number_end']}"
        metadata['id'] = hashlib.md5(id_str.encode()).hexdigest()
        
        self.collection.insert(
            text=text,
            text_embeddings=text_embedding,
            metadata=metadata
        )
        
    def update(self, file_path: str, file_name: str, text: str) -> None:
        """
        Update the vector within the given file path.
        
        Args:
            file_path: The file path to update.
            file_name: The file name to update.
            text: The text to update.
        
        Returns:
            None
        """
        logger.info(f"Updating vector for file: {file_path}")
        logger.warning("Dependency not implemented.")
        assert os.path.exists(file_path), "File path does not exist."
        assert os.path.isfile(file_path), "File path is not a file."
        
        processed_text = chunk_file(file_path)
        
        self._delete_by_path(file_path)
        for chunk in processed_text:
            metadata = {
                "chunked_document": chunk["text"],
                "file_name": file_name,
                "file_path": file_path,
                "line_number_start": chunk["line_number_start"],
                "line_number_end": chunk["line_number_end"],
                "file_type": chunk["file_type"],
                "file_language": chunk["file_language"],
                "function_type": chunk["function_type"],
                "function_name": chunk["function_name"],
                "dependencies": chunk["dependencies"]
            }
            self.insert(chunk["text"], metadata)
        
    
    def _delete_by_path(self, file_path: str):
        """
        Delete all vectors with the given file path.
        
        Args:
            file_path: The file path to delete.
        
        Returns:
            None
        """
        self.collection.delete(where={"file_path": file_path})
        
    def __len__(self):  
        return len(self.collection)
    
    def __str__(self):
        return "VectorBase"
        

class DummyVB(VectorBase):
    def __init__(self):
        # read csv file
        import pandas as pd
        data = pd.read_csv("tests/data/dummyVB/Fake_Python_Math_Functions.csv")
        self.data = [{"text": row["text"], "metadata": eval(row["metadata"])} for _, row in data.iterrows()]
        logger.info("Dummy VectorBase initialized.")
    def retrieve(self, query: str, top_k: int) -> list[dict]:
        return self.data
    
    def __str__(self):
        return "Dummy VectorBase"