"""
Streamlit patch to fix PyTorch compatibility issues.
This file should be imported before streamlit and torch to prevent conflicts.
"""

import sys
import os
import warnings

# Suppress specific warnings that may cause issues
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

def patch_torch_classes():
    """
    Patch torch._classes module to prevent Streamlit watcher issues.
    This prevents the RuntimeError with torch.__path__._path access.
    """
    try:
        import torch
        # Create a mock __path__ attribute if it doesn't exist properly
        if hasattr(torch, '_classes'):
            classes_module = torch._classes
            if not hasattr(classes_module, '__path__') or not hasattr(classes_module.__path__, '_path'):
                # Create a mock path object
                class MockPath:
                    def __init__(self):
                        self._path = []
                    
                    def __iter__(self):
                        return iter(self._path)
                
                if not hasattr(classes_module, '__path__'):
                    classes_module.__path__ = MockPath()
                elif not hasattr(classes_module.__path__, '_path'):
                    classes_module.__path__._path = []
                    
    except ImportError:
        # torch not installed, nothing to patch
        pass
    except Exception as e:
        # Log the error but don't fail
        print(f"Warning: Could not patch torch._classes: {e}")

def patch_streamlit_watcher():
    """
    Patch Streamlit's local sources watcher to handle torch classes better.
    """
    try:
        # Import and patch before streamlit starts watching
        import streamlit.watcher.local_sources_watcher as watcher
        
        original_extract_paths = getattr(watcher, 'extract_paths', None)
        
        def safe_extract_paths(module):
            """Safely extract paths from a module, handling torch._classes specially."""
            try:
                if hasattr(module, '__name__') and 'torch._classes' in str(module.__name__):
                    # Return empty list for torch._classes to avoid the error
                    return []
                
                if original_extract_paths:
                    return original_extract_paths(module)
                else:
                    # Fallback implementation
                    if hasattr(module, '__path__'):
                        if hasattr(module.__path__, '_path'):
                            return list(module.__path__._path)
                        else:
                            return list(module.__path__)
                    return []
                    
            except (RuntimeError, AttributeError) as e:
                # If we can't extract paths, return empty list
                return []
        
        # Replace the function
        watcher.extract_paths = safe_extract_paths
        
    except ImportError:
        # Streamlit not imported yet, that's fine
        pass
    except Exception as e:
        print(f"Warning: Could not patch streamlit watcher: {e}")

def apply_patches():
    """Apply all necessary patches."""
    patch_torch_classes()
    patch_streamlit_watcher()

# Apply patches when this module is imported
apply_patches()
