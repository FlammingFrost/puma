import sys
import os
import hashlib
import chromadb
import pandas as pd
from tools.logger import logger
from configs.configurator import config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class Database:
    """
    Database class to interact with the vector store. This class is responsible for inserting, updating, and retrieving vectors.
    
    Args:
        vector_store_path (str): Path to the vector store.
        embedding_func (function): The embedding function to encode text to a vector.
    """
    META_DATA_FORMAT = {
        'code': str,
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

    def __init__(self, vector_store_path: str, embedding_func = None, conf = None):
        if embedding_func is None:
            self.embedding_func = lambda x: self.get_embedding(text=x, model='text-embedding-ada-002')
        else:
            self.embedding_func = embedding_func
        self.client = chromadb.PersistentClient(path=vector_store_path)
        self.vector_store_path = vector_store_path
        self.client = chromadb.PersistentClient(path=vector_store_path)
        try:
            self.collection = self.client.get_collection("vector_store")
        except Exception as e:
            self.collection = self.client.create_collection("vector_store")
            logger.info("VectorBase collection created.")
        logger.info("VectorBase initialized.")
        
        if conf is None:
            conf = config
        
        self.top_k = conf["vectorbase_top_k"]
        self.name = conf.get("retriever_name", "Database")

    def _format_metadata(self, metadata: dict, match = False):
        """
        Format metadata to match the metadata schema.
        
        Args:
            metadata (dict): The metadata to format.
            
        Returns:
            dict: The formatted metadata.
        """
        formatted_metadata = {}
        for key, _ in self.META_DATA_FORMAT.items():
            formatted_metadata[key] = metadata.get(key, 'None')
            if match and key not in metadata:
                raise ValueError(f"Metadata key {key} not found.")
            
        return formatted_metadata
    
    def _generate_id(self, metadata: dict):
        """
        Generate an id for the metadata.
        
        Args:
            metadata (dict): The metadata to generate an id for.
            
        Returns:
            str: The generated id.
        """
        import uuid
        random_id = uuid.uuid4()
        id_str = f"{metadata['file_path']}:{metadata['line_number_start']}-{metadata['line_number_end']}-{random_id}"
        return id_str

    def setup_vector_store(self, data: list[dict], match = False):
        """
        Initialize the vector store.
        """
        if len(self) != 0:
            logger.info("Vector store already initialized.")
            return
        
        self._insert_many(data, match)

    def insert(self, code: str, metadata: dict, match = False):
        """
        Insert a new vector to the vector store.
        
        Args:
            text (str): The text to store.
            match (bool): Match metadata schema. Default is False. Raises an error if metadata key is missing.
        """
        text_embedding = self.embedding_func(code)
        formatted_metadata = self._format_metadata(metadata, match)

        formatted_metadata['id'] = self._generate_id(formatted_metadata)
        
        self.collection.add(
            ids=formatted_metadata['id'],
            embeddings=text_embedding,
            metadatas=formatted_metadata,
            documents=code
        )
        
    def _insert_many(self, data: list[dict], match = False):
        """
        Insert multiple vectors to the vector store.
        
        Args:
            data (list): List of dictionaries containing the text and metadata.
            match (bool): Match metadata schema. Default is False. Raises an error if metadata key is missing.
        """
        data = [self._format_metadata(row, match) for row in data]
        metadatas = []
        for row in data:
            row['id'] = self._generate_id(row)
            metadata = {k:v for k,v in row.items() if (k != 'code')}
            metadatas.append(metadata)
            
        logger.info(f"Inserting {len(data)} vectors to the vector store.")
        self.collection.add(
            ids=[row['id'] for row in data],
            embeddings=self.embedding_func([row['code'] for row in data]),
            metadatas=metadatas,
            documents=[row['code'] for row in data]
        )
        

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve top k vectors from the vector store based on the query.
        
        Args:
            query: The query string.
        
        Returns:
            A list of dictionaries containing the retrieved vectors.
        """
        query_embedding = self.embedding_func(query)
        query_result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=self.top_k,
            where=None
        )
        return query_result

    def update_file(self, file_path: str, data: list[dict]) -> None:
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
                
        self._delete_by_path(file_path)
        self._insert_many(data)

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
        return self.collection.count()

    def __str__(self):
        return f"Database: {self.name}\nVectorBase: {self.vector_store_path} --> Top K: {self.top_k}"


    def get_embedding(self, text: list[str], model: str = 'text-embedding-ada-002') -> list:
        """
        Get the embedding for the given text.
        
        Args:
            text (list[str]): The text to encode.
            model (str): The model to use for encoding. Default is 'text-embedding-ada-002'.
            
        Returns:
            list: The embedding for the given text.
        """
        import openai
        client = openai.Client(api_key=config["openai"]["api_key"])
        # TODO: handle too long text
        response = client.embeddings.create(model=model, input=text)
        
        return [res.embedding for res in response.data]