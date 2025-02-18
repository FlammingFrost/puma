# TODO: Implement this module
import chromadb
import os

from data_processing.chunker import Chunker

from tools.logger import logger

module_implemented = False

    

class VectorBase:
    """
    A class to store and retrieve vectors.
    
    Attributes:
        META_DATA_FORMAT (dict): The metadata format.
        collection (chromadb.Collection): The collection to store the vectors.
        embedding_func (function): The embedding function to encode text to a vector.
    """
    META_DATA_FORMAT = {
    'chunked_document': str,
    'file_name': str,
    'file_path': str,
    'line_number_start': int,
    'line_number_end': int,
    'file_type': ['code', 'text', 'config', 'data'],
    'file_language': ['py', 'java', 'c', 'cpp', 'json', 'yaml', 'yml', 'toml', 'md', 'txt'],
    'function_type': None,
    'function_name': None,
    'dependencies': None
    }
    collection = None
    embedding_func = None
    
    def __init__(self, vector_store_path: str, embedding_func):
        """
        Example path format: "./chroma_db"
        """
        client = chromadb.PersistentClient(path=vector_store_path)
        try:
            self.collection = client.get_collection("vector_store")
        except Exception as e:
            self.collection = client.create_collection("vector_store")
            logger.info("VectorBase collection created.")
        logger.info("VectorBase initialized.")
        
        self.embedding_func = embedding_func
        self.chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
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
    
    def insert(self, text: str, metadata: dict, match = False):
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
        if match:
            for key, value in metadata.items():
                if key not in self.META_DATA_FORMAT:
                    # raise ValueError(f"Missing metadata key: {key}")
                    continue
                if isinstance(self.META_DATA_FORMAT[key], list):
                    if type(value) not in self.META_DATA_FORMAT[key]:
                        raise ValueError(f"Unsupported {key} value: {type(value)}")
                elif not isinstance(value, self.META_DATA_FORMAT[key]):
                    raise ValueError(f"Invalid {key} value type: {type(value)}, value: {value}")
        for key in self.META_DATA_FORMAT.keys():
            if key not in metadata:
                metadata[key] = ''
            
        import hashlib
        id_str = f"{metadata['file_path']}:{metadata['line_number_start']}-{metadata['line_number_end']}"
        metadata['id'] = hashlib.md5(id_str.encode()).hexdigest()
        
        self.collection.add(
            ids=metadata['id'],
            embeddings=text_embedding,
            metadatas=metadata
        )
        
    def delete(self, where: dict):
        """
        Delete vectors from the vector store based on the where clause.
        
        Args:
            where: The where clause to filter the vectors to delete. Example: {"file_path": "path/to/file.py"}
            
        Returns:
            None
        """
        self.collection.delete(where)
        
    def __len__(self):
        return self.collection.count()
    
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