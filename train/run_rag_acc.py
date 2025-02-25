import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from retrieval.database import Database
from tools.logger import logger

file_path_python = "tests/data/python/python_train_0.jsonl.gz"
file_with_query = "tests/data/python/python_with_query.jsonl.gz"


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
    import openai
    
    
    db = Database("tests/rag_db/chroma_db")
    clip_data = [row for row in data if len(row['code']) <=4096]
    logger.info(f"Inserting {len(clip_data)} vectors to the vector store. Before clip: {len(data)}")
    data = clip_data
    db.setup_vector_store(data)
    queries = [row['docstring'] for row in data]
    retrievals = db.retrieve(queries)
    for i in range(len(data)):
        data[i]['retrieved_file_paths'] = [row['file_path'] for row in retrievals["metadatas"][i]]   
    
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
        data_python_raw = load_data_python(file_path_python, num_rows=10000)
        data_python = data_python_raw
        # data_python = generate_query(data_python_raw, num_rows=100, seed=224)
        
    data = get_retrieval_unstructure(data_python)
    accuracy = cal_accuracy(data)
    print(f"Accuracy: {accuracy}")
    with open("tests/data/python/python_test_accuracy", "w") as f:
        f.write(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()