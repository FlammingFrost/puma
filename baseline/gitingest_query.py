import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage
from configs import configurator
from gitingest import ingest
from tools.logger import logger

def generate_combined_query(queries: list) -> str:
    """
    Combines multiple queries into a single query with indexing.
    
    Args:
        queries (list): List of queries.
        
    Returns:
        str: The combined query with indexing.
    """
    combined_query = ""
    for idx, query in enumerate(queries, 1):
        combined_query += f"[Question-{idx}]:\n{query}\n"
    return combined_query

def query_codebase(path: str, queries: list, **kwargs) -> str:
    """
    Queries the codebase based on the given path and list of queries.
    
    Args:
        path (str): Path to the codebase.
        queries (list): List of queries.
        **kwargs: Additional parameters for ingestion.
            max_files (int): Maximum number of files to ingest. i.e. 50kb = 50 * 1024.
            include_patterns (list): List of file patterns to include. i.e. [".py", ".java"].
            exclude_patterns (list): List of file patterns to exclude. i.e. [".txt"].
        
    Returns:
        str: The response from the LLM.
    """
    # Load configuration
    config = configurator.config
    model = config["model"]
    api_key = config[model]["api_key"]
    temperature = config[model]["temperature"]
    model_name = config[model]["model_name"]
    
    # Initialize the LLM model
    if model == "openai":
        llm = ChatOpenAI(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature
        )
    elif model == "claude":
        llm = ChatAnthropic(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature
        )
    else:
        raise ValueError("Invalid model name in the config file.")
    
    # Ingest codebase using gitingest
    summary, tree, codebase_prompt = ingest(path, **kwargs)
    
    # Combine queries with indexing
    combined_query = generate_combined_query(queries)
    
    # Create the final prompt
    prompt = f"""
    You are a code assistant. Answer the user's questions based on the ingested codebase.
    
    Here is the information you need:
    {{
        "summary": "{summary}",
        "tree": "{tree}",
        "queries": [
            {combined_query}
        ],
        "codebase": "{codebase_prompt}"
    }}
    
    Provide concise and informative responses in the following JSON format:
    [
        {{
            "question_index": {{index}},
            "related_code_snippet": "{{related_code_snippet}}",
            "explain": "{{explain}}",
            "modification_suggestion": "{{modification_suggestion}}"
        }}
    ]
    """
    
    # Query LLM
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        logger.error(f"Error querying LLM in Baseline (gitingest): {e}")
        return f"Error querying LLM: {e}"
