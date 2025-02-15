# TODO: Implement this module
import os
import json
import yaml
import toml
from langchain.document_loaders import TextLoader, JSONLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import re

# from tools.logger_config import logger
    
class Chunker():
    def __init__(self, chunk_size=300, chunk_overlap=50, chunk_type="auto"):
        """
        Code chunking class for RAG.

        Args:
            chunk_size (int): Max chunk size (tokens or characters).
            chunk_overlap (int): Overlapping size between chunks.
            chunk_type (str): "recursive" for generic chunking, "function" for function-based, "auto" for automatic determination.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_type = chunk_type
    
    # def load_file(self, file_path: str):
    #     """
    #     Loads a file using LangChain document loaders or custom logic for code files.
    #     """
    #     if not os.path.exists(file_path):
    #         raise FileNotFoundError(f"File not found: {file_path}")
        
    #     # Determine file language
    #     file_extension = file_path.split(".")[-1].lower()

    #     # Use LangChain Loaders for specific file types
    #     # Supports ['py', 'java', 'c', 'cpp', 'json', 'yaml', 'yml', 'toml', 'md', 'txt']
    #     if file_extension in ["txt"]:
    #         return TextLoader(file_path).load()[0].page_content
    #     elif file_extension in ["md"]:
    #         return UnstructuredMarkdownLoader(file_path).load()[0].page_content
    #     elif file_extension == "json":
    #         return JSONLoader(file_path).load()[0].page_content
    #     elif file_extension in ["yaml", "yml"]:
    #         with open(file_path, "r", encoding="utf-8") as f:
    #             return yaml.safe_dump(yaml.safe_load(f))
    #     elif file_extension == "toml":
    #         with open(file_path, "r", encoding="utf-8") as f:
    #             return toml.dumps(toml.load(f))
    #     elif file_extension in ["py", "java", "c", "cpp"]:
    #         with open(file_path, "r", encoding="utf-8") as f:
    #             return f.read()
    #     else:
    #         raise ValueError(f"Unsupported file type: {file_extension}")
        
    def load_file(self, file_path: str):
        """Loads file content based on its type."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = file_path.split(".")[-1].lower()

        # Load based on file type
        if file_extension in ["txt", "md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif file_extension == "json":
            with open(file_path, "r", encoding="utf-8") as f:
                return json.dumps(json.load(f), indent=2)  # Convert JSON to string
        elif file_extension in ["yaml", "yml"]:
            with open(file_path, "r", encoding="utf-8") as f:
                return yaml.safe_dump(yaml.safe_load(f))  # Convert YAML to string
        elif file_extension == "toml":
            with open(file_path, "r", encoding="utf-8") as f:
                return toml.dumps(toml.load(f))  # Convert TOML to string
        elif file_extension in ["py", "java", "c", "cpp"]:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")


    def chunk_file(self, file_path: str) -> List[Dict]:
        """
        Chunks a file based on the chunk_type.

        Args:
            file_path (str): Path to the file.
            chunk_type (str): "recursive" (default) or "function" (for code).
            chunk_size (int): The maximum chunk size (tokens).
            chunk_overlap (int): Overlap between chunks.

        Returns:
            List[dict]: A list of chunk dictionaries.
        """
        file_content = self.load_file(file_path)
        file_extension = file_path.split(".")[-1].lower()
        
        # If auto mode, determine best chunking strategy
        if self.chunk_type == "auto":
            if file_extension in ["py", "java", "c", "cpp"]:
                print("\n--- (Auto) Implementing Function Based Chunking ---")
                return self.function_based_chunking(file_content, file_path)
            else:
                print("\n--- (Auto) Implementing Recursive Chunking ---")
                return self.recursive_chunking(file_content, file_path)

        # Manual selection of chunking strategy
        if self.chunk_type == "function":
            return self.function_based_chunking(file_content, file_path)
        else:
            return self.recursive_chunking(file_content, file_path)

    def recursive_chunking(self, file_content: str, file_path: str) -> List[Dict]:
        """
        Uses LangChain's RecursiveCharacterTextSplitter to split text.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],  # Prefers splitting at paragraphs, then lines
        )
        chunks = text_splitter.split_text(file_content)

        return [
            {
                "chunked_document": chunk,
                "file_path": file_path,
                "file_type": "text",
                "file_language": file_path.split(".")[-1],
            }
            for chunk in chunks
        ]

    def function_based_chunking(self, file_content, file_path):
        function_pattern = re.compile(r"^(\s*)?(def|class)\s+(\w+)\s*\(", re.MULTILINE)
        
        chunks = []
        last_pos = 0
        last_match = None

        for match in function_pattern.finditer(file_content):
            start = match.start()

            # If there's a previous match, store the previous function/class as a chunk
            if last_match:
                chunks.append({
                    "file_path": file_path,
                    "file_type": "code",
                    "file_language": "py",
                    "chunked_document": file_content[last_pos:start].strip(),
                    "line_number_start": file_content[:last_pos].count("\n"),
                    "line_number_end": file_content[:start].count("\n"),
                    "function_name": last_match.group(3),  # Extract function/class name
                    "function_type": "function" if last_match.group(2) == "def" else "class",
                })

            last_pos = start
            last_match = match

        # Add the last detected function/class
        if last_match:
            chunks.append({
                "file_path": file_path,
                "file_type": "code",
                "file_language": "py",
                "chunked_document": file_content[last_pos:].strip(),
                "line_number_start": file_content[:last_pos].count("\n"),
                "line_number_end": file_content.count("\n"),
                "function_name": last_match.group(3),
                "function_type": "function" if last_match.group(2) == "def" else "class",
            })

        return chunks
    
    
### **TEST CHUNKING FUNCTION**
def test_chunking():
    print("\n--- Testing Chunking ---")
    """Test chunking across different file types."""
    test_files = [
        "tests/data/test_chunking/example.py",   # Python file
        "tests/data/test_chunking/example.java", # Java file
        "tests/data/test_chunking/example.md",   # Markdown file
        "tests/data/test_chunking/example.txt",  # Plain text
        "tests/data/test_chunking/example.yaml", # YAML config
        "tests/data/test_chunking/example.toml"  # TOML config
    ]

    chunker = Chunker(chunk_size=300, chunk_overlap=50, chunk_type="auto")

    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"Skipping {file_path}: File does not exist")
            continue

        print(f"\n--- Chunking: {file_path} ---")
        chunks = chunker.chunk_file(file_path)

        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}:")
            print(chunk["chunked_document"])
            print("-" * 50)

if __name__ == "__main__":
    test_chunking()
