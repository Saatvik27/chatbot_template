import json
import os
from typing import Dict, Any
from pathlib import Path

# Configuration file path
CONFIG_FILE = "config/app_config.json"

# Default configuration
DEFAULT_CONFIG = {
    "groq_api_key": "",
    "admin_password": "admin123",
    "vector_store_path": "data/vector_store",
    "chat_history_path": "data/chat_history.json",    "model_settings": {
        "embedding_model": "all-MiniLM-L6-v2",
        "groq_model": "llama-3.3-70b-versatile",
        "max_tokens": 1024,
        "temperature": 0.1,
        "chunk_size": 1000,
        "chunk_overlap": 200
    },
    "ui_settings": {
        "app_title": "RAG Chatbot",
        "theme_color": "#2E86AB",
        "max_chat_history": 100
    },
    "security": {
        "session_timeout": 3600,
        "max_file_size": 10485760,  # 10MB
        "allowed_file_types": [".pdf"]
    }
}

def load_config() -> Dict[str, Any]:
    """
    Load configuration from file or create default if not exists
    
    Returns:
        Configuration dictionary
    """
    # Create config directory if it doesn't exist
    Path(os.path.dirname(CONFIG_FILE)).mkdir(parents=True, exist_ok=True)
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Merge with default config to ensure all keys exist
            merged_config = DEFAULT_CONFIG.copy()
            merged_config.update(config)
            
            return merged_config
            
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration...")
            save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()
    else:
        # Create default config file
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any]) -> bool:
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create config directory if it doesn't exist
        Path(os.path.dirname(CONFIG_FILE)).mkdir(parents=True, exist_ok=True)
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        
        return True
        
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get a specific configuration value
    
    Args:
        key: Configuration key (supports dot notation like 'model_settings.temperature')
        default: Default value if key not found
        
    Returns:
        Configuration value
    """
    config = load_config()
    
    # Handle dot notation for nested keys
    keys = key.split('.')
    value = config
    
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default

def set_config_value(key: str, value: Any) -> bool:
    """
    Set a specific configuration value
    
    Args:
        key: Configuration key (supports dot notation)
        value: Value to set
        
    Returns:
        True if successful, False otherwise
    """
    config = load_config()
    
    # Handle dot notation for nested keys
    keys = key.split('.')
    current = config
    
    try:
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value
        current[keys[-1]] = value
        
        return save_config(config)
        
    except Exception as e:
        print(f"Error setting config value: {e}")
        return False

def reset_config() -> bool:
    """
    Reset configuration to default values
    
    Returns:
        True if successful, False otherwise
    """
    return save_config(DEFAULT_CONFIG.copy())

def validate_config(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate configuration and return any errors
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Dictionary of validation errors (empty if valid)
    """
    errors = {}
    
    # Validate Groq API key format (basic check)
    groq_key = config.get('groq_api_key', '')
    if groq_key and not groq_key.startswith('gsk_'):
        errors['groq_api_key'] = "Groq API key should start with 'gsk_'"
    
    # Validate admin password
    admin_password = config.get('admin_password', '')
    if len(admin_password) < 6:
        errors['admin_password'] = "Admin password should be at least 6 characters long"
    
    # Validate model settings
    model_settings = config.get('model_settings', {})
    
    temperature = model_settings.get('temperature', 0.1)
    if not 0 <= temperature <= 2:
        errors['temperature'] = "Temperature should be between 0 and 2"
    
    max_tokens = model_settings.get('max_tokens', 1024)
    if not 1 <= max_tokens <= 4096:
        errors['max_tokens'] = "Max tokens should be between 1 and 4096"
    
    chunk_size = model_settings.get('chunk_size', 1000)
    if not 100 <= chunk_size <= 5000:
        errors['chunk_size'] = "Chunk size should be between 100 and 5000"
    
    chunk_overlap = model_settings.get('chunk_overlap', 200)
    if not 0 <= chunk_overlap < chunk_size:
        errors['chunk_overlap'] = "Chunk overlap should be between 0 and chunk_size"
    
    # Validate security settings
    security = config.get('security', {})
    
    max_file_size = security.get('max_file_size', 10485760)
    if not 1024 <= max_file_size <= 104857600:  # 1KB to 100MB
        errors['max_file_size'] = "Max file size should be between 1KB and 100MB"
    
    return errors

def get_app_info() -> Dict[str, Any]:
    """
    Get application information
    
    Returns:
        Dictionary with app information
    """
    config = load_config()
    
    return {
        'app_title': config.get('ui_settings', {}).get('app_title', 'RAG Chatbot'),
        'version': '1.0.0',
        'config_file': CONFIG_FILE,
        'has_groq_key': bool(config.get('groq_api_key', '')),
        'embedding_model': config.get('model_settings', {}).get('embedding_model', 'all-MiniLM-L6-v2'),
        'groq_model': config.get('model_settings', {}).get('groq_model', 'llama-3.3-70b-versatile')
    }
