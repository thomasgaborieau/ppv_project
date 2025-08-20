# src/__init__.py
"""
HRV Study Analysis Package

Core modules for analyzing indoor environmental data from the HRV study.
"""

from .utils import (
    load_house_data,
    load_all_houses_for_period,
    get_column_name,
    classify_house_status,
    split_by_time_period,
    # Remove split_by_time_period_legacy - it doesn't exist
    calculate_weekly_average,
    get_group_statistics,
    calculate_group_mean_ci,
    perform_t_test,
    calculate_comfort_zone_percentage,
    calculate_time_above_threshold,
    get_external_data,
    calculate_humidity_ratio,
    # Remove these if they don't exist in utils.py
    # analyze_solar_patterns,
    # get_solar_based_periods,
    # compare_fixed_vs_solar_periods,
    CONTROL_HOUSES,
    ACTIVE_HOUSES,
    SOLAR_DAY_NIGHT_THRESHOLD
)

from .statistical_tests import (
    independent_t_test,
    paired_t_test,
    anova_one_way,
    mann_whitney_u,
    normality_test,
    levene_test,
    correlation_test
)

from .config import (
    PROJECT_ROOT,
    DATA_PATH,  # Changed from DATA_DIR
    HOUSE_DATA_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    REPORTS_DIR,
    STUDY_PERIODS,
    TIME_PERIODS,
    TEMPERATURE_THRESHOLDS,
    RH_THRESHOLDS,
    SOLAR_THRESHOLDS,
    CONFIDENCE_LEVEL,
    SIGNIFICANCE_LEVEL,
    PLOT_SETTINGS,
    COLORS
)

__version__ = '1.0.0'
__author__ = 'HRV Study Research Team'