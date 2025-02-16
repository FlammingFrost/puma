from tests.load_data import load_data_python
from embedding.database import Database

file_path_python = "tests/data/python/python_train_0.jsonl.gz"

data_python = load_data_python(file_path_python, num_rows=100)

def get_retrieval_unstructure(data):
    """
    Get the retrieval for unstructured data. Input a collection of code snippets and their comments.
    Construct a database object and initialize it with the data.
    Try to retrieve the top k vectors from the vector store based on the query.
    Return the Top1 and Top5 accuracy.
    
    Args:
        data (list): List of dictionaries with the selected keys.
            keys:
                docstring: The docstring of the code.
                *query*: The query string. Transform from docstring.
                code: The code snippet.
                func_name: The function name.
                file_path: The file path. i.e. repo/path/to/file.py
                
    Returns:
        Database: The database object.
    """
    db = Database("tests/data/python/python_train_0.jsonl.gz", None)
    db.initialize(data)
    return db