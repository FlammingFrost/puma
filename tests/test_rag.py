from tests.load_data import load_data_python
from embedding.database import Database
from query_engine.retriever import VBRetriever
import os

TOP_K = 5

file_path_python = "tests/data/python/python_train_0.jsonl.gz"
file_with_query = "tests/data/python/python_with_query.jsonl.gz"

def generate_query(data):
    # TODO: Use api to generate query from docstring
    pass

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
    db = Database("tests/rag_db/chroma_db", None)
    db.initialize(data)
    rag_sys = VBRetriever(TOP_K)
    for row in data:
        query = row['query']
        retrieved_chunks = rag_sys.retrieve_relevant_chunks(query)
        retrieved_file_paths = [chunk['file_path'] for chunk in retrieved_chunks]
        row['retrieved_file_paths'] = retrieved_file_paths
    
    return data

def cal_accuracy(data):
    """
    Calculate the accuracy of the retrieval.
    
    Args:
        data (list): List of dictionaries with the selected keys.
            keys:
                docstring: The docstring of the code.
                query: The query string. Transform from docstring.
                code: The code snippet.
                func_name: The function name.
                file_path: The file path. i.e. repo/path/to/file.py
                retrieved_file_paths: The retrieved file paths.
                
    Returns:
        float: The accuracy of the retrieval.
    """
    total = len(data)
    correct = 0
    for row in data:
        if row['file_path'] in row['retrieved_file_paths']:
            correct += 1
    
    return correct / total
    
def main():
    if os.path.exists(file_with_query):
        data_python = load_data_python(file_with_query, num_rows=100)
    else:
        data_python_raw = load_data_python(file_path_python, num_rows=100)
        data_python = generate_query(data_python_raw, num_rows=100, seed=224)
        
    data = get_retrieval_unstructure(data_python)
    accuracy = cal_accuracy(data)
    print(f"Accuracy: {accuracy}")
    with open("tests/data/python/python_test_accuracy", "w") as f:
        f.write(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()