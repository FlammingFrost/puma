"""Reads configuration from settings.yaml file"""
import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "settings.yaml")
API_KEY_PATH = os.path.join(os.path.dirname(__file__), "api_key.yaml")

def load_config():
    """Loads configuration from settings.yaml"""
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
    with open(API_KEY_PATH, "r") as file:
        api_keys = yaml.safe_load(file)
        config["openai"]["api_key"] = api_keys["openai"]['api_key']
        config["claude"]["api_key"] = api_keys["claude"]['api_key']
    
        
    # Override API key with environment variable if set
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    if OPENAI_API_KEY:
        config["openai"]["api_key"] = OPENAI_API_KEY
    if CLAUDE_API_KEY:
        config["claude"]["api_key"] = CLAUDE_API_KEY
    
    return config
