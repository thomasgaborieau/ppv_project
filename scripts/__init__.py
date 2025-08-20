"""
Analysis scripts for HRV study research questions.

Each script corresponds to a specific research question from the study.
"""

import sys
from pathlib import Path

# Add src directory to Python path so we can import from it
# This allows scripts to find the src module even when run directly
src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import commonly used functions that scripts will need
try:
    from src.utils import (
        load_house_data,
        load_all_houses_for_period,
        get_group_statistics,
        classify_house_status,
        split_by_time_period,
        calculate_weekly_average,
        CONTROL_HOUSES,
        ACTIVE_HOUSES
    )
    
    from src.statistical_tests import (
        independent_t_test,
        paired_t_test,
        anova_one_way
    )
    
    from src.config import (
        DATA_PATH,
        HOUSE_DATA_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        STUDY_PERIODS,
        TIME_PERIODS,
        TEMPERATURE_THRESHOLDS
    )
    
    # Flag to indicate successful import
    IMPORTS_SUCCESSFUL = True
    
except ImportError as e:
    print(f"Warning: Could not import src modules: {e}")
    print("Make sure you're running from the project root directory")
    IMPORTS_SUCCESSFUL = False

# Helper function for scripts
def setup_analysis(question_number: int, description: str = None):
    """
    Common setup for analysis scripts.
    
    Parameters:
    -----------
    question_number : int
        Question number (1-10)
    description : str, optional
        Description of the analysis
    
    Returns:
    --------
    dict
        Configuration for the analysis
    """
    import logging
    from datetime import datetime
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - Q%(question)02d - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'question_{question_number:02d}.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger = logging.LoggerAdapter(logger, {'question': question_number})
    
    # Print header
    print("=" * 60)
    print(f"Question {question_number}: {description or 'Analysis'}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    return {
        'question': question_number,
        'description': description,
        'logger': logger,
        'start_time': datetime.now()
    }

# Define what's available when someone imports from scripts
__all__ = [
    'load_house_data',
    'get_group_statistics',
    'independent_t_test',
    'CONTROL_HOUSES',
    'ACTIVE_HOUSES',
    'setup_analysis',
    'IMPORTS_SUCCESSFUL'
]