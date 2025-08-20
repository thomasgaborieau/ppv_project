#!/usr/bin/env python
"""
Setup verification script for HRV Study Analysis.

This script checks that the environment is properly configured and
that data files are accessible.
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 7:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.7+")
        return False


def check_imports():
    """Check required package imports."""
    required_packages = [
        'pandas',
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn',
        'statsmodels'
    ]
    
    all_ok = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} not installed")
            all_ok = False
    
    return all_ok


def check_data_path():
    """Check data path configuration."""
    try:
        from src.config import DATA_PATH, HOUSE_DATA_DIR, print_config
        
        print("\n" + "="*50)
        print("DATA PATH CONFIGURATION")
        print("="*50)
        
        # Check if using environment variable
        env_path = os.getenv('HRV_DATA_PATH')
        if env_path:
            print(f"✓ Using HRV_DATA_PATH from environment: {env_path}")
        else:
            print("ℹ HRV_DATA_PATH not set, using default location")
        
        print(f"\nConfigured data path: {DATA_PATH}")
        print(f"House data directory: {HOUSE_DATA_DIR}")
        
        if DATA_PATH.exists():
            print(f"✓ Data path exists")
            
            if HOUSE_DATA_DIR.exists():
                print(f"✓ House data directory exists")
                return True
            else:
                print(f"✗ House data directory not found at {HOUSE_DATA_DIR}")
                print(f"  Please create: {HOUSE_DATA_DIR}")
                return False
        else:
            print(f"✗ Data path not found at {DATA_PATH}")
            print(f"  Please create the directory or set HRV_DATA_PATH")
            return False
            
    except Exception as e:
        print(f"✗ Error checking data path: {e}")
        return False


def check_data_files():
    """Check for expected data files."""
    try:
        from src.config import HOUSE_DATA_DIR
        
        print("\n" + "="*50)
        print("DATA FILES CHECK")
        print("="*50)
        
        if not HOUSE_DATA_DIR.exists():
            print("✗ House data directory doesn't exist")
            return False
        
        # Expected file patterns
        expected_files = []
        for week in [1, 2]:
            for year in [8, 9]:
                for house in range(1, 31):
                    filename = f"{week}{year}_{house:02d}.csv"
                    expected_files.append(filename)
        
        # Check existing files
        existing_files = list(HOUSE_DATA_DIR.glob("*.csv"))
        existing_names = [f.name for f in existing_files]
        
        print(f"Found {len(existing_files)} CSV files")
        
        # Check coverage
        missing_files = []
        for expected in expected_files:
            if expected not in existing_names:
                missing_files.append(expected)
        
        if len(missing_files) == 0:
            print(f"✓ All 120 expected files found (30 houses × 2 weeks × 2 years)")
            return True
        elif len(missing_files) == len(expected_files):
            print(f"✗ No data files found in {HOUSE_DATA_DIR}")
            print(f"  Expected format: WY_HH.csv (e.g., 18_01.csv)")
            return False
        else:
            print(f"⚠ Found {len(existing_files)}/{len(expected_files)} expected files")
            print(f"  Missing {len(missing_files)} files")
            
            # Show sample of missing files
            if len(missing_files) <= 10:
                print(f"  Missing: {', '.join(missing_files)}")
            else:
                print(f"  Missing: {', '.join(missing_files[:5])} ... and {len(missing_files)-5} more")
            
            return False
            
    except Exception as e:
        print(f"✗ Error checking data files: {e}")
        return False


def test_data_loading():
    """Test loading a sample data file."""
    try:
        from src.utils import load_house_data
        from src.config import HOUSE_DATA_DIR
        
        print("\n" + "="*50)
        print("DATA LOADING TEST")
        print("="*50)
        
        # Find any available file
        csv_files = list(HOUSE_DATA_DIR.glob("*.csv"))
        if not csv_files:
            print("✗ No CSV files to test")
            return False
        
        # Parse first available file name
        sample_file = csv_files[0].name
        # Expected format: WY_HH.csv
        parts = sample_file.replace('.csv', '').split('_')
        if len(parts) != 2:
            print(f"✗ Unexpected file format: {sample_file}")
            return False
            
        week_year = parts[0]
        house_num = int(parts[1])
        week = int(week_year[0])
        year = int(week_year[1])
        
        print(f"Testing with: Week {week}, Year 200{year}, House {house_num:02d}")
        
        # Try loading
        df = load_house_data(week, year, house_num)
        
        if df.empty:
            print("✗ Loaded dataframe is empty")
            return False
        
        print(f"✓ Successfully loaded {len(df)} records")
        print(f"  Columns: {', '.join(df.columns[:5])}...")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        
        # Check for expected columns
        expected_cols = ['T-L', 'RH-L', 'T-B', 'RH-B', 'T-R', 'RH-R']
        found_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in expected_cols)]
        
        if found_cols:
            print(f"✓ Found {len(found_cols)} expected column patterns")
        else:
            print(f"⚠ Expected column patterns not found")
            print(f"  Looking for: T-L*, RH-L*, T-B*, RH-B*, T-R*, RH-R*")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing data loading: {e}")
        return False


def main():
    """Run all checks."""
    print("\n" + "="*50)
    print("HRV STUDY ANALYSIS - SETUP VERIFICATION")
    print("="*50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Package Imports", check_imports),
        ("Data Path Configuration", check_data_path),
        ("Data Files", check_data_files),
        ("Data Loading", test_data_loading)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\nChecking {name}...")
        print("-" * 30)
        result = check_func()
        results.append((name, result))
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! Your environment is ready.")
        print("\nNext steps:")
        print("1. Run analysis scripts: python scripts/question_01.py")
        print("2. Or use Jupyter: jupyter notebook notebooks/exploratory_analysis.ipynb")
    else:
        print("\n⚠ Some checks failed. Please address the issues above.")
        print("\nCommon fixes:")
        print("1. Set data path: export HRV_DATA_PATH=/path/to/your/data")
        print("2. Install packages: pip install -r requirements.txt")
        print("3. Ensure CSV files are in: your_data_path/house_data/preprocessed/")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())