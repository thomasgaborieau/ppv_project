"""
Test suite for HRV study analysis.

Unit tests for utility functions and statistical methods.
"""

import sys
from pathlib import Path

# Add src directory to path for testing
src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Test configuration
TEST_DATA_PATH = Path(__file__).parent / 'test_data'
TEST_HOUSE_NUMS = [1, 5, 15, 20, 30]  # Sample houses for testing

# Import test utilities
import unittest
import numpy as np
import pandas as pd

def create_test_dataframe(n_rows=1000, seed=42):
    """
    Create a test dataframe with similar structure to real data.
    
    Parameters:
    -----------
    n_rows : int
        Number of rows
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Test dataframe
    """
    np.random.seed(seed)
    
    # Create datetime index (2-minute intervals)
    dates = pd.date_range(
        start='2008-07-28 00:00:00',
        periods=n_rows,
        freq='2min'
    )
    
    # Create test data
    df = pd.DataFrame({
        'T-L18': np.random.normal(20, 3, n_rows),      # Living room temp
        'RH-L18': np.random.normal(55, 10, n_rows),    # Living room RH
        'T-B18': np.random.normal(18, 3, n_rows),      # Bedroom temp
        'RH-B18': np.random.normal(60, 10, n_rows),    # Bedroom RH
        'T-R18': np.random.normal(15, 5, n_rows),      # Roof temp
        'RH-R18': np.random.normal(70, 15, n_rows),    # Roof RH
        'ext__T': np.random.normal(12, 4, n_rows),     # External temp
        'ext__SR': np.maximum(0, np.random.normal(200, 150, n_rows)),  # Solar
        'ext__RH': np.random.normal(75, 10, n_rows)    # External RH
    }, index=dates)
    
    # Ensure solar is 0 at night (simple simulation)
    hour = df.index.hour
    df.loc[(hour < 6) | (hour > 20), 'ext__SR'] = np.random.uniform(0, 0.004, 
                                                                     sum((hour < 6) | (hour > 20)))
    
    return df

def assert_dataframe_equal(df1, df2, check_exact=False):
    """
    Assert two dataframes are equal (approximately).
    
    Parameters:
    -----------
    df1, df2 : pd.DataFrame
        Dataframes to compare
    check_exact : bool
        If True, check exact equality; if False, use approximate
    """
    try:
        if check_exact:
            pd.testing.assert_frame_equal(df1, df2)
        else:
            pd.testing.assert_frame_equal(df1, df2, check_exact=False, atol=1e-5)
        return True
    except AssertionError as e:
        print(f"Dataframes not equal: {e}")
        return False

# Base test class with common setup
class HRVTestCase(unittest.TestCase):
    """Base class for HRV analysis tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_df = create_test_dataframe()
        cls.control_houses = [5, 7, 12, 16, 17, 21, 23, 25, 28, 30]
        cls.active_houses = [i for i in range(1, 31) if i not in cls.control_houses]
    
    def assertAlmostEqualRelative(self, first, second, rel_tol=1e-5, abs_tol=1e-8):
        """
        Assert two values are almost equal (relative tolerance).
        
        Parameters:
        -----------
        first, second : float
            Values to compare
        rel_tol : float
            Relative tolerance
        abs_tol : float
            Absolute tolerance
        """
        if abs(first - second) <= max(rel_tol * max(abs(first), abs(second)), abs_tol):
            return
        else:
            raise AssertionError(f"{first} != {second} within tolerance")

# Test discovery
def load_tests():
    """Load all tests from the tests directory."""
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    return suite

__all__ = [
    'create_test_dataframe',
    'assert_dataframe_equal',
    'HRVTestCase',
    'load_tests',
    'TEST_DATA_PATH',
    'TEST_HOUSE_NUMS'
]