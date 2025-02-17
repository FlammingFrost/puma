import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.logger import logger
from embedding.vector_store import VectorBase

def get_embedding(text: str|list[str], model = 'text-embedding-ada-002') -> list:
    """
    Get the embedding for the given text.
    
    Args:
        text (str): The text to encode.
        model (str): The model to use for encoding. Default is 'text-embedding-ada-002'.
        
    Returns:
        list: The embedding for the given text.
    """
    import openai
    from configs.configurator import config
    client = openai.Client(api_key=config["openai"]["api_key"])
    response = client.embeddings.create(model=model, input=text)
    if isinstance(text, str):
        return response.data[0].embedding
    else:
        return [res.embedding for res in response.data]

class Database:
    """
    Database class to interact with the vector store. This class is responsible for inserting and updating vectors.
    * vector_store is not designed to be used directly. It is used by the Database class or through retriever.py.
    
    Args:
        vector_store_path (str): Path to the vector store.
        embedding_func (function): The embedding function to encode text to a vector.
    """
    
    def __init__(self, vector_store_path: str, embedding_func = get_embedding):
        self.vector_store = VectorBase(vector_store_path, embedding_func)
        self.metadataSchema = VectorBase.META_DATA_FORMAT
        
    def _format_metadata(self, metadata: dict, match = False):
        """
        Format metadata to match the metadata schema.
        
        Args:
            metadata (dict): The metadata to format.
            
        Returns:
            dict: The formatted metadata.
        """
        formatted_metadata = {}
        for key, _ in self.metadataSchema.items():
            formatted_metadata[key] = metadata.get(key, '')
            if match and key not in metadata:
                raise ValueError(f"Metadata key {key} not found.")
            
        return formatted_metadata
        
    def initialize(self, data: list[dict], match = False):
        """
        Initialize the vector store.
        """
        if len(self.vector_store) != 0:
            logger.info("Vector store already initialized.")
            return
        
        for idx, row in enumerate(data):
            metadata = self._format_metadata(row)
            self.vector_store.insert(row['code'], metadata, match)
            if idx % 1000 == 0:
                logger.info(f"Inserted {idx} records.")
    
    def insert(self, text: str, metadata: dict, match = False):
        """
        Insert a new vector to the vector store.
        
        Args:
            text (str): The text to store.
            match (bool): Match metadata schema. Default is False. Raises an error if metadata key is missing.
        """
        formatted_metadata = self._format_metadata(metadata, match)
        self.vector_store.insert(text, formatted_metadata)
        
    def insert_file(self, file_path: str, file_name: str, text: str):
        pass
        
    def update_file(self, file_path: str, file_name: str, text: str) -> None:
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
        self.vector_store.delete(where={"file_path": file_path})
