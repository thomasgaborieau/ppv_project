import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent

# Data path - can be set via environment variable or defaults to local data folder
DATA_PATH = Path(os.getenv('HRV_DATA_PATH', PROJECT_ROOT / 'data'))

# Validate that DATA_PATH exists
if not DATA_PATH.exists():
    import warnings
    warnings.warn(f"Data path {DATA_PATH} does not exist. Please set HRV_DATA_PATH environment variable or create the directory.")

# Derived data paths
HOUSE_DATA_DIR = DATA_PATH / "house_data" / "preprocessed"
RAW_DATA_DIR = DATA_PATH / "raw"

# Results paths (keep in project directory)
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
REPORTS_DIR = RESULTS_DIR / "reports"

# Create results directories if they don't exist
for dir_path in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# House classifications
CONTROL_HOUSES = [5, 7, 12, 16, 17, 21, 23, 25, 28, 30]
ACTIVE_HOUSES = [i for i in range(1, 31) if i not in CONTROL_HOUSES]

# Study periods
STUDY_PERIODS = {
    '2008_week1': {'year': 8, 'week': 1, 'dates': '28/07/2008 - 03/08/2008'},
    '2008_week2': {'year': 8, 'week': 2, 'dates': '04/08/2008 - 10/08/2008'},
    '2009_week1': {'year': 9, 'week': 1, 'dates': '15/06/2009 - 21/06/2009'},
    '2009_week2': {'year': 9, 'week': 2, 'dates': '22/06/2009 - 28/06/2009'}
}

# Time period definitions
TIME_PERIODS = {
    'full_day': 'all',
    'daytime': 'solar_above_threshold',
    'evening_night': 'solar_below_threshold'
}

# Solar irradiance threshold for day/night definition
SOLAR_DAY_NIGHT_THRESHOLD = 0.005  # W/m² - threshold to distinguish day from night

# Temperature thresholds
TEMPERATURE_THRESHOLDS = {
    'extreme_cold': 12.0,     # °C - acute cardiovascular risk
    'who_min': 18.0,          # °C - WHO minimum recommended
    'who_max': 24.0,          # °C - WHO maximum recommended
    'roof_target': 18.0       # °C - target roof cavity temperature
}

# Relative humidity thresholds
RH_THRESHOLDS = {
    'ashrae_min': 30.0,       # % - ASHRAE minimum recommended
    'ashrae_max': 60.0        # % - ASHRAE maximum recommended
}

# Solar irradiance thresholds
SOLAR_THRESHOLDS = {
    'day_night': 0.005,       # W/m² - threshold for day/night definition
    'min_effective': 400.0    # W/m² - minimum for effective roof heating
}

# Statistical settings
CONFIDENCE_LEVEL = 0.95
SIGNIFICANCE_LEVEL = 0.05

# Plot settings
PLOT_SETTINGS = {
    'figure_size': (10, 6),
    'dpi': 300,
    'font_size': 12,
    'title_size': 14,
    'label_size': 12,
    'legend_size': 10,
    'line_width': 2,
    'marker_size': 8,
    'grid': True,
    'grid_alpha': 0.3
}

# Color scheme for plots
COLORS = {
    'control': '#FF6B6B',     # Red
    'active': '#4ECDC4',      # Teal
    'week1_2008': '#95E77E',  # Light green
    'week2_2008': '#68B684',  # Medium green
    'week1_2009': '#FFE66D',  # Light yellow
    'week2_2009': '#FFC93C',  # Orange
    'daytime': '#FFD93D',     # Yellow
    'evening_night': '#6A0572' # Purple
}

# Output formats
OUTPUT_FORMATS = {
    'figures': ['png', 'pdf'],
    'tables': ['csv', 'xlsx'],
    'reports': ['html', 'pdf']
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': str(PROJECT_ROOT / 'hrv_analysis.log')
        }
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

def get_data_path():
    """
    Get the configured data path.
    
    Returns:
    --------
    Path
        The configured data path
    """
    return DATA_PATH

def set_data_path(new_path):
    """
    Set a new data path programmatically.
    
    Parameters:
    -----------
    new_path : str or Path
        New path to data directory
    """
    global DATA_PATH, HOUSE_DATA_DIR, RAW_DATA_DIR
    DATA_PATH = Path(new_path)
    HOUSE_DATA_DIR = DATA_PATH / "house_data" / "preprocessed"
    RAW_DATA_DIR = DATA_PATH / "raw"
    
    if not DATA_PATH.exists():
        import warnings
        warnings.warn(f"Data path {DATA_PATH} does not exist.")
    
    return DATA_PATH

def print_config():
    """
    Print current configuration settings.
    """
    print("HRV Study Analysis Configuration")
    print("=" * 40)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Path: {DATA_PATH}")
    print(f"  - House Data: {HOUSE_DATA_DIR}")
    print(f"  - Raw Data: {RAW_DATA_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"  - Figures: {FIGURES_DIR}")
    print(f"  - Tables: {TABLES_DIR}")
    print(f"  - Reports: {REPORTS_DIR}")
    print(f"\nData Path Exists: {DATA_PATH.exists()}")
    print(f"House Data Dir Exists: {HOUSE_DATA_DIR.exists()}")
    print("=" * 40)