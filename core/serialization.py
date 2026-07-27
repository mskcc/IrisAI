"""JSON serialization utilities — extracted from app.py.

Pure function for making complex objects JSON-serializable.
"""


def make_json_serializable(obj):
    """Convert complex objects to JSON-serializable format.
    
    Handles LangChain objects, nested dicts/lists, and arbitrary objects.
    
    Args:
        obj: Any Python object
    
    Returns:
        JSON-serializable version of the object
    """
    if hasattr(obj, 'dict'):
        try:
            return obj.dict()
        except Exception:
            return str(obj)
    elif hasattr(obj, '__dict__'):
        try:
            return {k: make_json_serializable(v) for k, v in obj.__dict__.items()}
        except Exception:
            return str(obj)
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)
