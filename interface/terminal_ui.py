# TODO: Implement this module
# interface/terminal_ui.py
from query_engine.retriever import VBRetriever
from query_engine.query_handler import process_query
from configs.configurator import config_loader
from tools.logger import logger

def main():
    config = config_loader.config
    
    # Available models list
    import openai
    client = openai.OpenAI(api_key=config["openai"]["api_key"])
    try:
        available_models = client.models.list()
        model_list = [model.id for model in available_models.data]
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise e
    assert config["openai"]["model_name"] in model_list, f"Invalid model name: {config['openai']['model_name']}. Available models: {model_list}"
        
        
        
        
    print("PUMA - Project Understanding & Modification Accelerator")
    print("Type 'exit()' to quit.")
    
    retriever = VBRetriever(config)

    while True:
        user_input = input("\n[Enter your query]: ")
        if user_input.lower() == "exit()":
            break
        if user_input == "debug":
            logger.debug("Debug message")
            continue
        if user_input == "":
            continue

        response = process_query(user_input, retriever)
        print("\n[Response]:")
        print(response)

if __name__ == "__main__":
    main()