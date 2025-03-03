"""Reads configuration from settings.yaml file"""
import yaml
import os
from tools.logger import logger

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "settings.yaml")
API_KEY_PATH = os.path.join(os.path.dirname(__file__), "api_key.yaml")

class ConfigLoader:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        """Loads configuration from settings.yaml"""
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as file:
                config = yaml.safe_load(file)
        if os.path.exists(API_KEY_PATH):
            with open(API_KEY_PATH, "r") as file:
                api_keys = yaml.safe_load(file)
                config["openai"]["api_key"] = api_keys["openai"]['api_key']
                config["claude"]["api_key"] = api_keys["claude"]['api_key']
            
            # Override API key with environment variable if set
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
            os.environ["OPENAI_API_KEY"] = config["openai"]["api_key"]
            os.environ["CLAUDE_API_KEY"] = config["claude"]["api_key"]
            if OPENAI_API_KEY:
                config["openai"]["api_key"] = OPENAI_API_KEY
            if CLAUDE_API_KEY:
                config["claude"]["api_key"] = CLAUDE_API_KEY
            
            # DEBUG: list all available models from OpenAI
            import openai
            openai.api_key = config["openai"]["api_key"]
            openai_models = openai.models.list()
            openai_models = [model.id for model in openai_models.data]
            logger.debug(f"Available OpenAI models:\n {openai_models}")
        
        return config

config_loader = ConfigLoader()
config = config_loader.config
