# TODO: Implement this module
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage
from configs.config_loader import load_config

# Initialize the LLM model
try:
    config = load_config()
    model = config["model"]
    api_key = config[model]["api_key"]
    temperature = config[model]["temperature"]
    model_name = config[model]["model_name"]
    if model == "openai":
        llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=api_key,
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
except KeyError as e:
    raise ValueError(f"Missing configuration key: {e}")
except Exception as e:
    raise ValueError(f"Error initializing model: {e}")


def generate_response(user_query: str, retrieved_chunks: list) -> str:
    """
    Calls an LLM api to generate a response based on the retrieved code snippets.
    
    Args:
        user_query (str): The original user query.
        retrieved_chunks (list): List of retrieved code snippets and metadata.
        
    Returns:
        str: The response from the LLM.
    """
    # Format the retrieved code into a structured prompt
    context = "\n".join(
        [f"File: {chunk['filename']} (Line {chunk['line_number']})\nCode:\n{chunk['code']}\n" 
         for chunk in retrieved_chunks]
    )
    
    prompt = f"""
    You are a code assistant. Answer the user's question based on the retrieved code snippets.

    User Query: {user_query}
    
    Relevant Code Snippets:
    {context}
    
    Provide a concise and informative response.
    """

    # Query LLM
    response = llm.invoke([HumanMessage(content=prompt)])    
    return response.content