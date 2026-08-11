"""Pytest configuration for ML_History project."""
import sys
import os


def pytest_configure(config):
    """Add project root and common to Python path."""
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    
    common = os.path.join(root, 'common')
    if common not in sys.path:
        sys.path.insert(0, common)


def pytest_collectstart(collector):
    """Ensure each test file's directory is in path and clear model cache."""
    if hasattr(collector, 'path'):
        test_dir = str(collector.path.parent)
        if test_dir not in sys.path:
            sys.path.insert(0, test_dir)
        
        # Clear cached model module if from different directory
        if 'model' in sys.modules:
            cached_file = getattr(sys.modules['model'], '__file__', '')
            if cached_file and not cached_file.startswith(test_dir):
                del sys.modules['model']
