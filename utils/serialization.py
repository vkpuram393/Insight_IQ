"""
Generic Serialization Helpers

Type-safe serialization utilities using TypeVar for any Pydantic model.
These helpers work with ALL Pydantic models in the system (errors, node results, etc.)
"""

from typing import TypeVar, Type, Dict, Any, List
from pydantic import BaseModel

# TypeVar for generic model operations
T = TypeVar('T', bound=BaseModel)


def to_dict(model: T) -> Dict[str, Any]:
    """
    Generic: Convert any Pydantic model to dictionary
    
    Works with ANY Pydantic model in the system including:
    - AgentError, ErrorResponse
    - IntentResult, ToolResult, ResponsePayload
    - ConversationMessage, SessionFact, ContextResult
    - Any custom Pydantic model
    
    Example:
        error_dict = to_dict(error)
        intent_dict = to_dict(intent_result)
        response_dict = to_dict(response_payload)
    
    Args:
        model: Any Pydantic model instance
        
    Returns:
        Dictionary representation of the model
    """
    return model.model_dump()


def from_dict(model_class: Type[T], data: Dict[str, Any]) -> T:
    """
    Generic: Create any Pydantic model from dictionary
    
    Example:
        error = from_dict(AgentError, error_dict)
        intent = from_dict(IntentResult, intent_dict)
        response = from_dict(ResponsePayload, response_dict)
    
    Args:
        model_class: The Pydantic model class to create
        data: Dictionary data to populate the model
        
    Returns:
        Instance of the specified model class
    """
    return model_class.model_validate(data)


def to_json(model: T) -> str:
    """
    Generic: Convert any Pydantic model to JSON string
    
    Example:
        error_json = to_json(error)
        intent_json = to_json(intent_result)
    
    Args:
        model: Any Pydantic model instance
        
    Returns:
        JSON string representation of the model
    """
    return model.model_dump_json()


def from_json(model_class: Type[T], json_str: str) -> T:
    """
    Generic: Create any Pydantic model from JSON string
    
    Example:
        error = from_json(AgentError, json_str)
        intent = from_json(IntentResult, json_str)
    
    Args:
        model_class: The Pydantic model class to create
        json_str: JSON string data
        
    Returns:
        Instance of the specified model class
    """
    return model_class.model_validate_json(json_str)


def copy_model(model: T, **updates) -> T:
    """
    Generic: Create a copy of any Pydantic model with optional field updates
    
    Example:
        new_error = copy_model(original_error, session_id="new-session")
        new_intent = copy_model(original_intent, confidence=0.95)
        new_tool = copy_model(original_tool, retry_count=1)
    
    Args:
        model: Any Pydantic model instance to copy
        **updates: Keyword arguments with field updates
        
    Returns:
        New instance of the same model type with updates applied
    """
    model_dict = model.model_dump()
    model_dict.update(updates)
    return model.__class__.model_validate(model_dict)


def to_dict_list(models: List[T]) -> List[Dict[str, Any]]:
    """
    Generic: Convert list of any Pydantic models to list of dictionaries
    
    Example:
        error_dicts = to_dict_list([error1, error2, error3])
        intent_dicts = to_dict_list([intent1, intent2])
        message_dicts = to_dict_list(conversation_history)
    
    Args:
        models: List of any Pydantic model instances
        
    Returns:
        List of dictionary representations
    """
    return [model.model_dump() for model in models]


def from_dict_list(model_class: Type[T], data_list: List[Dict[str, Any]]) -> List[T]:
    """
    Generic: Create list of any Pydantic models from list of dictionaries
    
    Example:
        errors = from_dict_list(AgentError, error_dicts)
        intents = from_dict_list(IntentResult, intent_dicts)
        messages = from_dict_list(ConversationMessage, message_dicts)
    
    Args:
        model_class: The Pydantic model class to create instances of
        data_list: List of dictionary data
        
    Returns:
        List of model instances
    """
    return [model_class.model_validate(data) for data in data_list]


# Convenience aliases for common operations
serialize = to_dict
deserialize = from_dict
serialize_list = to_dict_list
deserialize_list = from_dict_list

