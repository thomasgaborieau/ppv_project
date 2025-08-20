"""
Utility functions for HRV study data analysis.
This module contains functions for data loading, preprocessing, and common calculations.
FIXED VERSION: Uses absolute imports that work in all contexts
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
from datetime import datetime, time
from scipy import stats
import sys

# Handle imports whether used as package or direct module
try:
    # Try relative import first (when used as part of src package)
    from .config import HOUSE_DATA_DIR
except ImportError:
    # Fall back to absolute import (when imported directly)
    try:
        from config import HOUSE_DATA_DIR
    except ImportError:
        # Last resort: try to find config in the same directory
        import os
        config_path = Path(__file__).parent
        sys.path.insert(0, str(config_path))
        from config import HOUSE_DATA_DIR

# Constants for house classification
CONTROL_HOUSES = [5, 7, 12, 16, 17, 21, 23, 25, 28, 30]
ACTIVE_HOUSES = [i for i in range(1, 31) if i not in CONTROL_HOUSES]

# Solar irradiance threshold for day/night definition
SOLAR_DAY_NIGHT_THRESHOLD = 0.005  # W/m²

# WHO temperature recommendations
WHO_TEMP_MIN = 18.0  # °C
WHO_TEMP_MAX = 24.0  # °C

# ASHRAE relative humidity recommendations
ASHRAE_RH_MIN = 30.0  # %
ASHRAE_RH_MAX = 60.0  # %


def load_house_data(week: int, year: int, house_num: int, 
                   data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load data for a specific house, week, and year.
    
    Parameters:
    -----------
    week : int
        Week number (1 or 2)
    year : int
        Year (8 for 2008 or 9 for 2009)
    house_num : int
        House number (1-30)
    data_dir : str or Path, optional
        Path to the preprocessed data directory. If None, uses config.HOUSE_DATA_DIR
    
    Returns:
    --------
    pd.DataFrame
        Loaded data with proper datetime index
    """
    # Use configured data directory if not specified
    if data_dir is None:
        # Try to import HOUSE_DATA_DIR using the same strategy
        try:
            from .config import HOUSE_DATA_DIR
        except ImportError:
            try:
                from config import HOUSE_DATA_DIR
            except ImportError:
                # Use the already imported HOUSE_DATA_DIR from top of file
                pass
        data_dir = HOUSE_DATA_DIR
    
    # Format filename
    filename = f"{week}{year}_{house_num:02d}.csv"
    filepath = Path(data_dir) / filename
    
    if not filepath.exists():
        warnings.warn(f"File {filepath} not found")
        return pd.DataFrame()
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Convert timestamps to datetime
    if 'timestamps' in df.columns:
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df.set_index('timestamps', inplace=True)
    
    # Add metadata
    df.attrs['house_num'] = house_num
    df.attrs['week'] = week
    df.attrs['year'] = 2000 + year
    df.attrs['is_control'] = house_num in CONTROL_HOUSES
    
    return df


def get_column_name(variable: str, location: str, week: int, year: int) -> str:
    """
    Generate column name based on variable, location, week, and year.
    
    Parameters:
    -----------
    variable : str
        'T' for temperature or 'RH' for relative humidity
    location : str
        'L' for living room, 'B' for bedroom, 'R' for roof cavity
    week : int
        Week number (1 or 2)
    year : int
        Year (8 or 9)
    
    Returns:
    --------
    str
        Column name in format 'T-L18' or 'RH-B29'
    """
    return f"{variable}-{location}{week}{year}"


def classify_house_status(house_num: int, week: int, year: int) -> str:
    """
    Determine if a house measurement is 'control' or 'active'.
    
    Control houses: Always control
    Active houses: Control in week 1 of 2008, active otherwise
    
    Parameters:
    -----------
    house_num : int
        House number (1-30)
    week : int
        Week number (1 or 2)
    year : int
        Year (8 or 9)
    
    Returns:
    --------
    str
        'control' or 'active'
    """
    if house_num in CONTROL_HOUSES:
        return 'control'
    elif week == 1 and year == 8:
        return 'control'  # Active houses are control in week 1 of 2008
    else:
        return 'active'


def load_all_houses_for_period(week: int, year: int, 
                               data_dir: Optional[Union[str, Path]] = None) -> Dict[int, pd.DataFrame]:
    """
    Load data for all houses for a specific week and year.
    
    Parameters:
    -----------
    week : int
        Week number (1 or 2)
    year : int
        Year (8 or 9)
    data_dir : str or Path, optional
        Path to data directory. If None, uses config.HOUSE_DATA_DIR
    
    Returns:
    --------
    dict
        Dictionary with house numbers as keys and DataFrames as values
    """
    # Use configured data directory if not specified
    if data_dir is None:
        # Use the already imported HOUSE_DATA_DIR
        data_dir = HOUSE_DATA_DIR
    
    data = {}
    for house_num in range(1, 31):
        df = load_house_data(week, year, house_num, data_dir)
        if not df.empty:
            data[house_num] = df
    return data


def split_by_time_period(df: pd.DataFrame, 
                        solar_threshold: float = SOLAR_DAY_NIGHT_THRESHOLD) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into daytime and evening/night periods based on solar irradiance.
    
    Daytime: Solar irradiance >= threshold (default 0.005 W/m²)
    Evening/Night: Solar irradiance < threshold
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with datetime index and solar irradiance column
    solar_threshold : float
        Threshold for solar irradiance to define day/night (default 0.005 W/m²)
    
    Returns:
    --------
    tuple
        (daytime_df, evening_night_df)
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Look for solar irradiance column
    solar_columns = ['ext__SR', 'ext_SR', 'SR', 'solar_irradiance']
    solar_col = None
    
    for col in solar_columns:
        if col in df.columns:
            solar_col = col
            break
    
    if solar_col is None:
        warnings.warn("No solar irradiance column found. Using fallback time-based method (9am-5pm).")
        # Fallback to time-based splitting
        times = df.index.time
        daytime_mask = (times >= time(9, 0)) & (times < time(17, 0))
        evening_night_mask = ~daytime_mask
    else:
        # Use solar irradiance for splitting
        daytime_mask = df[solar_col] >= solar_threshold
        evening_night_mask = df[solar_col] < solar_threshold
        
        # Log some statistics
        daytime_hours = daytime_mask.sum() * 2 / 60  # Assuming 2-minute intervals
        total_hours = len(df) * 2 / 60
        
        if daytime_hours > 0:
            avg_daytime_solar = df.loc[daytime_mask, solar_col].mean()
            avg_nighttime_solar = df.loc[evening_night_mask, solar_col].mean()
            
            # Store metadata
            daytime_df = df[daytime_mask].copy()
            evening_night_df = df[evening_night_mask].copy()
            
            daytime_df.attrs['avg_solar_irradiance'] = avg_daytime_solar
            daytime_df.attrs['hours'] = daytime_hours
            evening_night_df.attrs['avg_solar_irradiance'] = avg_nighttime_solar
            evening_night_df.attrs['hours'] = total_hours - daytime_hours
            
            return daytime_df, evening_night_df
    
    return df[daytime_mask].copy(), df[evening_night_mask].copy()


def calculate_weekly_average(df: pd.DataFrame, column: str, 
                            confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate weekly average with confidence interval.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data frame with the column to analyze
    column : str
        Column name to calculate average for
    confidence_level : float
        Confidence level for CI (default 0.95)
    
    Returns:
    --------
    tuple
        (mean, ci_lower, ci_upper)
    """
    if df.empty or column not in df.columns:
        return np.nan, np.nan, np.nan
    
    data = df[column].dropna()
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    
    mean = data.mean()
    sem = data.sem()  # Standard error of mean
    
    # Calculate confidence interval
    ci = stats.t.interval(confidence_level, len(data)-1, loc=mean, scale=sem)
    
    return mean, ci[0], ci[1]


def get_group_statistics(week: int, year: int, variable: str, location: str,
                         group: str = 'all', time_period: str = 'all',
                         data_dir: Optional[Union[str, Path]] = None,
                         solar_threshold: float = SOLAR_DAY_NIGHT_THRESHOLD) -> pd.DataFrame:
    """
    Calculate statistics for a group of houses.
    
    Parameters:
    -----------
    week : int
        Week number (1 or 2)
    year : int
        Year (8 or 9)
    variable : str
        'T' for temperature or 'RH' for relative humidity
    location : str
        'L', 'B', or 'R'
    group : str
        'all', 'control', or 'active'
    time_period : str
        'all', 'daytime', or 'evening_night'
    data_dir : str or Path, optional
        Path to data directory. If None, uses config.HOUSE_DATA_DIR
    solar_threshold : float
        Solar irradiance threshold for day/night splitting (default 0.005 W/m²)
    
    Returns:
    --------
    pd.DataFrame
        Statistics for each house in the group
    """
    # Use configured data directory if not specified
    if data_dir is None:
        data_dir = HOUSE_DATA_DIR
    
    results = []
    column_name = get_column_name(variable, location, week, year)
    
    for house_num in range(1, 31):
        # Check if house belongs to requested group
        status = classify_house_status(house_num, week, year)
        if group != 'all' and status != group:
            continue
        
        # Load data
        df = load_house_data(week, year, house_num, data_dir)
        if df.empty or column_name not in df.columns:
            continue
        
        # Filter by time period if needed
        if time_period == 'daytime':
            df, _ = split_by_time_period(df, solar_threshold)
        elif time_period == 'evening_night':
            _, df = split_by_time_period(df, solar_threshold)
        
        # Calculate statistics
        mean, ci_lower, ci_upper = calculate_weekly_average(df, column_name)
        
        # Additional solar statistics if available
        solar_stats = {}
        if 'ext__SR' in df.columns:
            solar_stats['avg_solar'] = df['ext__SR'].mean()
            solar_stats['max_solar'] = df['ext__SR'].max()
            if time_period != 'all':
                solar_stats['period_hours'] = df.attrs.get('hours', len(df) * 2 / 60)
        
        results.append({
            'house_num': house_num,
            'status': status,
            'mean': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': df[column_name].std() if column_name in df.columns else np.nan,
            'min': df[column_name].min() if column_name in df.columns else np.nan,
            'max': df[column_name].max() if column_name in df.columns else np.nan,
            'count': len(df[column_name].dropna()) if column_name in df.columns else 0,
            **solar_stats
        })
    
    return pd.DataFrame(results)


def calculate_group_mean_ci(group_stats: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Calculate overall group mean with confidence interval.
    
    Parameters:
    -----------
    group_stats : pd.DataFrame
        Statistics for individual houses
    
    Returns:
    --------
    tuple
        (group_mean, ci_lower, ci_upper)
    """
    if group_stats.empty:
        return np.nan, np.nan, np.nan
    
    means = group_stats['mean'].dropna()
    if len(means) == 0:
        return np.nan, np.nan, np.nan
    
    group_mean = means.mean()
    sem = means.sem()
    ci = stats.t.interval(0.95, len(means)-1, loc=group_mean, scale=sem)
    
    return group_mean, ci[0], ci[1]


def perform_t_test(group1_stats: pd.DataFrame, group2_stats: pd.DataFrame,
                  alternative: str = 'two-sided') -> Tuple[float, float]:
    """
    Perform independent samples t-test between two groups.
    
    Parameters:
    -----------
    group1_stats : pd.DataFrame
        Statistics for group 1
    group2_stats : pd.DataFrame
        Statistics for group 2
    alternative : str
        'two-sided', 'less', or 'greater'
    
    Returns:
    --------
    tuple
        (t_statistic, p_value)
    """
    group1_means = group1_stats['mean'].dropna()
    group2_means = group2_stats['mean'].dropna()
    
    if len(group1_means) == 0 or len(group2_means) == 0:
        return np.nan, np.nan
    
    result = stats.ttest_ind(group1_means, group2_means, alternative=alternative)
    return result.statistic, result.pvalue


def calculate_comfort_zone_percentage(df: pd.DataFrame, week: int, year: int,
                                     location: str) -> float:
    """
    Calculate percentage of time in comfort zone (WHO temp + ASHRAE RH).
    
    Parameters:
    -----------
    df : pd.DataFrame
        House data
    week : int
        Week number
    year : int
        Year (8 or 9)
    location : str
        'L' or 'B'
    
    Returns:
    --------
    float
        Percentage of time in comfort zone
    """
    temp_col = get_column_name('T', location, week, year)
    rh_col = get_column_name('RH', location, week, year)
    
    if temp_col not in df.columns or rh_col not in df.columns:
        return np.nan
    
    # Create comfort zone mask
    temp_comfort = (df[temp_col] >= WHO_TEMP_MIN) & (df[temp_col] <= WHO_TEMP_MAX)
    rh_comfort = (df[rh_col] >= ASHRAE_RH_MIN) & (df[rh_col] <= ASHRAE_RH_MAX)
    comfort_zone = temp_comfort & rh_comfort
    
    return (comfort_zone.sum() / len(df)) * 100


def calculate_time_above_threshold(df: pd.DataFrame, column: str, 
                                  threshold: float) -> float:
    """
    Calculate percentage of time above a threshold.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data frame
    column : str
        Column to analyze
    threshold : float
        Threshold value
    
    Returns:
    --------
    float
        Percentage of time above threshold
    """
    if column not in df.columns:
        return np.nan
    
    above_threshold = df[column] > threshold
    return (above_threshold.sum() / len(df)) * 100


def get_external_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract external (outdoor) data from dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        House data
    
    Returns:
    --------
    pd.DataFrame
        External data columns
    """
    ext_columns = ['ext__T', 'ext__SR', 'ext__RH', 'ext__WD', 'ext__WS']
    available_columns = [col for col in ext_columns if col in df.columns]
    
    if available_columns:
        return df[available_columns].copy()
    else:
        return pd.DataFrame()


def calculate_humidity_ratio(temperature: float, relative_humidity: float, 
                            pressure: float = 101325) -> float:
    """
    Calculate humidity ratio (g water / kg dry air).
    
    Parameters:
    -----------
    temperature : float
        Temperature in Celsius
    relative_humidity : float
        Relative humidity in percentage
    pressure : float
        Atmospheric pressure in Pa (default sea level)
    
    Returns:
    --------
    float
        Humidity ratio in g/kg
    """
    # Saturation vapor pressure (Pa) - Antoine equation
    T_kelvin = temperature + 273.15
    p_sat = 611.21 * np.exp((18.678 - temperature/234.5) * (temperature/(257.14 + temperature)))
    
    # Partial pressure of water vapor
    p_vapor = (relative_humidity / 100) * p_sat
    
    # Humidity ratio
    w = 622 * p_vapor / (pressure - p_vapor)  # g/kg
    
    return w