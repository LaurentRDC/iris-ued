"""
Utility functions for tests
"""
import unittest
import warnings

class FutureProofTestRunner(unittest.TextTestRunner):
    """
    A test runner that raises errors on Deprecation warnings.
    """
    def run(self, *args, **kwargs):
        warnings.filterwarnings("error", category=DeprecationWarning)
        return super().run(*args, **kwargs)