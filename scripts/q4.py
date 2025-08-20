#!/usr/bin/env python
"""
Question 4 ENHANCED: Comprehensive Temperature Exposure Analysis

Question 4: Is there any difference in household's exposure to temperature below 12°C,
and to WHO recommended temperature (18°C and 24°C) between the active homes and the control homes?

This analysis examines:
- Exposure to extreme cold (<12°C) 
- Time within WHO recommended range (18-24°C)
- Distribution of temperature frequencies
- Temporal patterns of exposure
- Statistical comparisons between groups
- Visualization of exposure patterns
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.stats import gaussian_kde
warnings.filterwarnings('ignore')

# Fix Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent if current_file.parent.name == 'scripts' else current_file.parent
src_path = project_root / 'src'

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import modules
try:
    import utils
    import statistical_tests
    import config
    
    from utils import (
        load_house_data,
        classify_house_status,
        split_by_time_period,
        get_column_name,
        CONTROL_HOUSES,
        ACTIVE_HOUSES,
        SOLAR_DAY_NIGHT_THRESHOLD
    )
    from statistical_tests import (
        independent_t_test,
        normality_test,
        levene_test,
        mann_whitney_u,
        paired_t_test
    )
    from config import (
        RESULTS_DIR,
        FIGURES_DIR, 
        TABLES_DIR,
        COLORS,
        TEMPERATURE_THRESHOLDS
    )
except ImportError:
    sys.path.insert(0, str(current_file.parent.parent))
    from src.utils import *
    from src.statistical_tests import *
    from src.config import *

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Temperature thresholds
EXTREME_COLD = 12.0  # °C - Risk of cardiovascular strain
WHO_MIN = 18.0       # °C - WHO minimum recommended
WHO_MAX = 24.0       # °C - WHO maximum recommended
EXTREME_HOT = 28.0   # °C - Upper comfort limit

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_house_group(house_num: int) -> str:
    """Get the house group (control or active) based on house number."""
    return 'control' if house_num in CONTROL_HOUSES else 'active'


def calculate_exposure_metrics(df: pd.DataFrame, temp_col: str, 
                              time_period: str = 'all') -> dict:
    """
    Calculate comprehensive temperature exposure metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with temperature column
    temp_col : str
        Name of temperature column
    time_period : str
        'all', 'day', 'night', or specific hours
    
    Returns:
    --------
    dict
        Exposure metrics
    """
    if temp_col not in df.columns:
        return {}
    
    temp_data = df[temp_col].dropna()
    
    if len(temp_data) == 0:
        return {}
    
    # Basic statistics
    metrics = {
        'mean': temp_data.mean(),
        'std': temp_data.std(),
        'median': temp_data.median(),
        'min': temp_data.min(),
        'max': temp_data.max(),
        'q25': temp_data.quantile(0.25),
        'q75': temp_data.quantile(0.75),
        'iqr': temp_data.quantile(0.75) - temp_data.quantile(0.25),
        'n_points': len(temp_data),
        'hours': len(temp_data) * 2 / 60  # Assuming 2-minute intervals
    }
    
    # Exposure to different temperature ranges
    metrics['pct_below_12'] = (temp_data < EXTREME_COLD).mean() * 100
    metrics['pct_12_to_18'] = ((temp_data >= EXTREME_COLD) & (temp_data < WHO_MIN)).mean() * 100
    metrics['pct_18_to_24'] = ((temp_data >= WHO_MIN) & (temp_data <= WHO_MAX)).mean() * 100
    metrics['pct_above_24'] = (temp_data > WHO_MAX).mean() * 100
    metrics['pct_above_28'] = (temp_data > EXTREME_HOT).mean() * 100
    
    # Time in minutes for key thresholds
    interval_minutes = 2  # 2-minute data intervals
    metrics['minutes_below_12'] = (temp_data < EXTREME_COLD).sum() * interval_minutes
    metrics['minutes_in_who'] = ((temp_data >= WHO_MIN) & (temp_data <= WHO_MAX)).sum() * interval_minutes
    metrics['minutes_above_24'] = (temp_data > WHO_MAX).sum() * interval_minutes
    
    # Temperature variability
    metrics['range'] = metrics['max'] - metrics['min']
    metrics['cv'] = (metrics['std'] / metrics['mean'] * 100) if metrics['mean'] != 0 else 0
    
    # Extreme events
    metrics['extreme_cold_events'] = count_extreme_events(temp_data, EXTREME_COLD, 'below')
    metrics['extreme_hot_events'] = count_extreme_events(temp_data, EXTREME_HOT, 'above')
    
    return metrics


def count_extreme_events(temp_series: pd.Series, threshold: float, 
                         direction: str = 'below', min_duration: int = 15) -> int:
    """
    Count number of extreme temperature events.
    
    Parameters:
    -----------
    temp_series : pd.Series
        Temperature data
    threshold : float
        Temperature threshold
    direction : str
        'below' or 'above'
    min_duration : int
        Minimum duration in data points for an event (default 15 = 30 minutes)
    
    Returns:
    --------
    int
        Number of events
    """
    if direction == 'below':
        extreme = temp_series < threshold
    else:
        extreme = temp_series > threshold
    
    # Find consecutive extreme periods
    events = 0
    current_event_length = 0
    
    for is_extreme in extreme:
        if is_extreme:
            current_event_length += 1
        else:
            if current_event_length >= min_duration:
                events += 1
            current_event_length = 0
    
    # Check last event
    if current_event_length >= min_duration:
        events += 1
    
    return events


def analyze_temporal_patterns(df: pd.DataFrame, temp_col: str) -> dict:
    """
    Analyze temporal patterns of temperature exposure.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with datetime index
    temp_col : str
        Temperature column name
    
    Returns:
    --------
    dict
        Temporal pattern metrics
    """
    if temp_col not in df.columns or df.empty:
        return {}
    
    patterns = {}
    
    # Hourly patterns
    df['hour'] = df.index.hour
    hourly_means = df.groupby('hour')[temp_col].mean()
    patterns['hourly_range'] = hourly_means.max() - hourly_means.min()
    patterns['coldest_hour'] = hourly_means.idxmin()
    patterns['warmest_hour'] = hourly_means.idxmax()
    
    # Night vs day using solar if available
    if 'ext__SR' in df.columns:
        day_df, night_df = split_by_time_period(df, SOLAR_DAY_NIGHT_THRESHOLD)
        if not day_df.empty and not night_df.empty:
            patterns['day_mean'] = day_df[temp_col].mean()
            patterns['night_mean'] = night_df[temp_col].mean()
            patterns['day_night_diff'] = patterns['day_mean'] - patterns['night_mean']
            
            # Exposure during specific periods
            patterns['night_pct_below_12'] = (night_df[temp_col] < EXTREME_COLD).mean() * 100
            patterns['day_pct_in_who'] = ((day_df[temp_col] >= WHO_MIN) & 
                                         (day_df[temp_col] <= WHO_MAX)).mean() * 100
    
    return patterns


def create_temperature_distribution_plot(all_data: dict):
    """
    Create comprehensive temperature distribution visualization.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    for year_idx, year in enumerate([2008, 2009]):
        for week_idx, week in enumerate([1, 2]):
            for loc_idx, location in enumerate(['living', 'bedroom']):
                ax = axes[loc_idx, year_idx * 2 + week_idx]
                
                # Get data for this combination
                active_temps = []
                control_temps = []
                
                key = f"{year}_week{week}_{location}"
                if key in all_data:
                    active_temps = all_data[key].get('active', [])
                    control_temps = all_data[key].get('control', [])
                
                if active_temps and control_temps:
                    # Create density plots
                    temp_range = np.linspace(
                        min(min(active_temps), min(control_temps)) - 2,
                        max(max(active_temps), max(control_temps)) + 2,
                        200
                    )
                    
                    # Calculate KDE
                    try:
                        active_kde = gaussian_kde(active_temps)
                        control_kde = gaussian_kde(control_temps)
                        
                        active_density = active_kde(temp_range)
                        control_density = control_kde(temp_range)
                        
                        # Plot densities
                        ax.plot(temp_range, active_density, color=COLORS.get('active', '#4ECDC4'),
                               linewidth=2, label='Active houses')
                        ax.plot(temp_range, control_density, color=COLORS.get('control', '#FF6B6B'),
                               linewidth=2, label='Control houses')
                        
                        # Fill areas for different zones
                        ax.fill_between(temp_range, 0, active_density, 
                                      where=(temp_range < EXTREME_COLD),
                                      color='blue', alpha=0.1)
                        ax.fill_between(temp_range, 0, active_density,
                                      where=((temp_range >= WHO_MIN) & (temp_range <= WHO_MAX)),
                                      color='green', alpha=0.1)
                        ax.fill_between(temp_range, 0, active_density,
                                      where=(temp_range > EXTREME_HOT),
                                      color='red', alpha=0.1)
                    except:
                        # Fallback to histogram if KDE fails
                        ax.hist(active_temps, bins=30, alpha=0.5, density=True,
                               color=COLORS.get('active', '#4ECDC4'), label='Active')
                        ax.hist(control_temps, bins=30, alpha=0.5, density=True,
                               color=COLORS.get('control', '#FF6B6B'), label='Control')
                    
                    # Add reference lines
                    ax.axvline(x=EXTREME_COLD, color='blue', linestyle=':', alpha=0.7, linewidth=1)
                    ax.axvline(x=WHO_MIN, color='green', linestyle='--', alpha=0.7, linewidth=1)
                    ax.axvline(x=WHO_MAX, color='green', linestyle='--', alpha=0.7, linewidth=1)
                    ax.axvline(x=EXTREME_HOT, color='red', linestyle=':', alpha=0.7, linewidth=1)
                    
                    # Labels and title
                    ax.set_xlabel('Temperature (°C)')
                    ax.set_ylabel('Density')
                    title = f'{location.capitalize()} - {year} W{week}'
                    if year == 2008 and week == 1:
                        title += '\n(Baseline)'
                    ax.set_title(title, fontsize=10)
                    
                    if loc_idx == 0 and week_idx == 0:
                        ax.legend(fontsize=8, loc='upper right')
                    
                    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Temperature Distribution and Exposure Zones', fontsize=14, y=1.02)
    
    # Add legend for zones
    zone_patches = [
        mpatches.Patch(color='blue', alpha=0.3, label='Extreme cold (<12°C)'),
        mpatches.Patch(color='green', alpha=0.3, label='WHO range (18-24°C)'),
        mpatches.Patch(color='red', alpha=0.3, label='Extreme hot (>28°C)')
    ]
    fig.legend(handles=zone_patches, loc='center', bbox_to_anchor=(0.5, -0.02), ncol=3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'question_04_temperature_distributions.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def create_exposure_comparison_bars(exposure_df: pd.DataFrame):
    """
    Create bar plots comparing exposure metrics between groups.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics_to_plot = [
        ('pct_below_12', 'Time Below 12°C (%)', 'Extreme Cold Exposure'),
        ('pct_18_to_24', 'Time in WHO Range (%)', 'Optimal Temperature Exposure'),
        ('pct_above_24', 'Time Above 24°C (%)', 'Warm Temperature Exposure'),
        ('minutes_below_12', 'Minutes Below 12°C/week', 'Weekly Cold Exposure'),
        ('minutes_in_who', 'Minutes in WHO Range/week', 'Weekly Optimal Exposure'),
        ('extreme_cold_events', 'Number of Events', 'Extreme Cold Events (>30 min)')
    ]
    
    for idx, (metric, ylabel, title) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        
        # Prepare data for plotting
        plot_data = []
        labels = []
        colors_list = []
        
        for year in [2008, 2009]:
            for week in [1, 2]:
                for location in ['living', 'bedroom']:
                    for group in ['active', 'control']:
                        # Filter data
                        mask = (exposure_df['year'] == year) & \
                               (exposure_df['week'] == week) & \
                               (exposure_df['location'] == location) & \
                               (exposure_df['house_group'] == group)
                        
                        group_data = exposure_df[mask]
                        
                        if not group_data.empty and metric in group_data.columns:
                            values = group_data[metric].dropna()
                            if len(values) > 0:
                                plot_data.append(values.mean())
                                label = f'{year%2000}W{week}\n{location[:3].upper()}\n{group[:3].upper()}'
                                labels.append(label)
                                colors_list.append(COLORS.get(group, '#888888'))
        
        if plot_data:
            x_pos = np.arange(len(plot_data))
            bars = ax.bar(x_pos, plot_data, color=colors_list, alpha=0.7, edgecolor='black')
            
            # Customize plot
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, val in zip(bars, plot_data):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=7)
    
    plt.suptitle('Temperature Exposure Metrics Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'question_04_exposure_bars.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_temporal_exposure_plot(temporal_df: pd.DataFrame):
    """
    Create plots showing temporal patterns of exposure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Hourly patterns
    ax = axes[0, 0]
    for group in ['active', 'control']:
        group_data = temporal_df[temporal_df['house_group'] == group]
        if 'coldest_hour' in group_data.columns:
            hourly_cold = group_data.groupby('coldest_hour').size()
            ax.bar(hourly_cold.index + (0.3 if group == 'control' else 0), 
                  hourly_cold.values, width=0.3,
                  label=f'{group.capitalize()} houses',
                  color=COLORS.get(group, '#888888'), alpha=0.7)
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Coldest Hours')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Day vs Night temperature difference
    ax = axes[0, 1]
    if 'day_night_diff' in temporal_df.columns:
        for group in ['active', 'control']:
            group_data = temporal_df[temporal_df['house_group'] == group]['day_night_diff'].dropna()
            if len(group_data) > 0:
                ax.boxplot([group_data], positions=[1 if group == 'active' else 2],
                          widths=0.5, patch_artist=True,
                          boxprops=dict(facecolor=COLORS.get(group, '#888888'), alpha=0.7))
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Active', 'Control'])
        ax.set_ylabel('Temperature Difference (°C)')
        ax.set_title('Day - Night Temperature Difference')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Night exposure to cold
    ax = axes[1, 0]
    if 'night_pct_below_12' in temporal_df.columns:
        plot_data = []
        labels = []
        colors_list = []
        
        for year in [2008, 2009]:
            for week in [1, 2]:
                for group in ['active', 'control']:
                    mask = (temporal_df['year'] == year) & \
                           (temporal_df['week'] == week) & \
                           (temporal_df['house_group'] == group)
                    
                    values = temporal_df[mask]['night_pct_below_12'].dropna()
                    if len(values) > 0:
                        plot_data.append(values.mean())
                        labels.append(f'{year%2000}W{week}\n{group[:3].upper()}')
                        colors_list.append(COLORS.get(group, '#888888'))
        
        if plot_data:
            bars = ax.bar(range(len(plot_data)), plot_data, color=colors_list, alpha=0.7)
            ax.set_xticks(range(len(plot_data)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel('% Time Below 12°C')
            ax.set_title('Nighttime Exposure to Extreme Cold')
            ax.grid(True, alpha=0.3)
    
    # Plot 4: Temperature variability (range)
    ax = axes[1, 1]
    for group in ['active', 'control']:
        group_data = temporal_df[temporal_df['house_group'] == group]
        if 'hourly_range' in group_data.columns:
            values = group_data['hourly_range'].dropna()
            if len(values) > 0:
                ax.scatter(np.random.normal(1 if group == 'active' else 2, 0.04, len(values)),
                          values, alpha=0.5, s=30,
                          color=COLORS.get(group, '#888888'),
                          label=f'{group.capitalize()} houses')
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Active', 'Control'])
    ax.set_ylabel('Daily Temperature Range (°C)')
    ax.set_title('Temperature Variability (Hourly Range)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Temporal Patterns of Temperature Exposure', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'question_04_temporal_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()


def statistical_comparison(exposure_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform statistical comparisons of exposure metrics between groups.
    """
    results = []
    
    metrics = ['pct_below_12', 'pct_18_to_24', 'pct_above_24', 
               'minutes_below_12', 'minutes_in_who', 'extreme_cold_events']
    
    for year in [2008, 2009]:
        for week in [1, 2]:
            for location in ['living', 'bedroom']:
                for metric in metrics:
                    # Get data for both groups
                    active_mask = (exposure_df['year'] == year) & \
                                 (exposure_df['week'] == week) & \
                                 (exposure_df['location'] == location) & \
                                 (exposure_df['house_group'] == 'active')
                    
                    control_mask = (exposure_df['year'] == year) & \
                                  (exposure_df['week'] == week) & \
                                  (exposure_df['location'] == location) & \
                                  (exposure_df['house_group'] == 'control')
                    
                    active_data = exposure_df[active_mask][metric].dropna()
                    control_data = exposure_df[control_mask][metric].dropna()
                    
                    if len(active_data) > 0 and len(control_data) > 0:
                        # Test normality
                        active_norm = normality_test(active_data.values)
                        control_norm = normality_test(control_data.values)
                        
                        # Perform appropriate test
                        if active_norm['is_normal'] and control_norm['is_normal']:
                            test_result = independent_t_test(active_data.values, control_data.values)
                            test_type = 't-test'
                        else:
                            test_result = mann_whitney_u(active_data.values, control_data.values)
                            test_type = 'Mann-Whitney U'
                        
                        results.append({
                            'year': year,
                            'week': week,
                            'location': location,
                            'metric': metric,
                            'active_mean': active_data.mean(),
                            'active_std': active_data.std(),
                            'control_mean': control_data.mean(),
                            'control_std': control_data.std(),
                            'difference': active_data.mean() - control_data.mean(),
                            'test_type': test_type,
                            'p_value': test_result['p_value'],
                            'significant': test_result['p_value'] < 0.05
                        })
    
    return pd.DataFrame(results)


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run the complete Question 4 analysis."""
    
    print("=" * 80)
    print("QUESTION 4: TEMPERATURE EXPOSURE ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing differences in exposure to:")
    print("  - Extreme cold (<12°C)")
    print("  - WHO recommended range (18-24°C)")
    print("  - Temperature variability and extremes")
    print("=" * 80)
    
    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_exposure_data = []
    all_temporal_data = []
    all_temperature_data = {}  # For distribution plots
    
    # Analyze each house
    for year in [8, 9]:
        for week in [1, 2]:
            print(f"\n{'='*60}")
            print(f"Analyzing Year 200{year}, Week {week}")
            if year == 8 and week == 1:
                print("(Baseline - No ventilation units installed)")
            elif year == 8 and week == 2:
                print("(After ventilation installation in active homes)")
            print(f"{'='*60}")
            
            # Collect temperature data for distribution plots
            for location in ['living', 'bedroom']:
                active_temps = []
                control_temps = []
                
                for house_num in range(1, 31):
                    df = load_house_data(week, year, house_num)
                    if df.empty:
                        continue
                    
                    house_group = get_house_group(house_num)
                    temp_col = get_column_name('T', location[0].upper(), week, year)
                    
                    if temp_col in df.columns:
                        temp_values = df[temp_col].dropna().values
                        if house_group == 'active':
                            active_temps.extend(temp_values)
                        else:
                            control_temps.extend(temp_values)
                        
                        # Calculate exposure metrics for overall period
                        overall_metrics = calculate_exposure_metrics(df, temp_col, 'all')
                        
                        if overall_metrics:
                            exposure_record = {
                                'house_num': house_num,
                                'house_group': house_group,
                                'year': 2000 + year,
                                'week': week,
                                'location': location,
                                'period': 'overall',
                                **overall_metrics
                            }
                            all_exposure_data.append(exposure_record)
                        
                        # Calculate day/night specific metrics if solar data available
                        if 'ext__SR' in df.columns:
                            day_df, night_df = split_by_time_period(df, SOLAR_DAY_NIGHT_THRESHOLD)
                            
                            # Daytime metrics
                            if not day_df.empty:
                                day_metrics = calculate_exposure_metrics(day_df, temp_col, 'day')
                                if day_metrics:
                                    day_record = {
                                        'house_num': house_num,
                                        'house_group': house_group,
                                        'year': 2000 + year,
                                        'week': week,
                                        'location': location,
                                        'period': 'day',
                                        **day_metrics
                                    }
                                    all_exposure_data.append(day_record)
                            
                            # Nighttime metrics
                            if not night_df.empty:
                                night_metrics = calculate_exposure_metrics(night_df, temp_col, 'night')
                                if night_metrics:
                                    night_record = {
                                        'house_num': house_num,
                                        'house_group': house_group,
                                        'year': 2000 + year,
                                        'week': week,
                                        'location': location,
                                        'period': 'night',
                                        **night_metrics
                                    }
                                    all_exposure_data.append(night_record)
                        
                        # Analyze temporal patterns
                        temporal_patterns = analyze_temporal_patterns(df, temp_col)
                        if temporal_patterns:
                            temporal_record = {
                                'house_num': house_num,
                                'house_group': house_group,
                                'year': 2000 + year,
                                'week': week,
                                'location': location,
                                **temporal_patterns
                            }
                            all_temporal_data.append(temporal_record)
                
                # Store for distribution plots
                key = f"{2000+year}_week{week}_{location}"
                all_temperature_data[key] = {
                    'active': active_temps,
                    'control': control_temps
                }
    
    # Create DataFrames
    exposure_df = pd.DataFrame(all_exposure_data)
    temporal_df = pd.DataFrame(all_temporal_data)
    
    # Save raw data
    exposure_df.to_csv(TABLES_DIR / 'question_04_exposure_data.csv', index=False)
    temporal_df.to_csv(TABLES_DIR / 'question_04_temporal_data.csv', index=False)
    
    # Perform statistical comparisons
    print("\n" + "="*80)
    print("STATISTICAL COMPARISONS")
    print("="*80)
    
    overall_exposure = exposure_df[exposure_df['period'] == 'overall']
    comparison_results = statistical_comparison(overall_exposure)
    comparison_results.to_csv(TABLES_DIR / 'question_04_statistical_comparisons.csv', index=False)
    
    # Print summary statistics
    print("\n" + "-"*60)
    print("EXTREME COLD EXPOSURE (<12°C)")
    print("-"*60)
    
    for year in [2008, 2009]:
        print(f"\nYear {year}:")
        year_data = overall_exposure[overall_exposure['year'] == year]
        
        for location in ['living', 'bedroom']:
            loc_data = year_data[year_data['location'] == location]
            
            for week in [1, 2]:
                week_data = loc_data[loc_data['week'] == week]
                
                if not week_data.empty:
                    active_pct = week_data[week_data['house_group'] == 'active']['pct_below_12'].mean()
                    control_pct = week_data[week_data['house_group'] == 'control']['pct_below_12'].mean()
                    
                    active_min = week_data[week_data['house_group'] == 'active']['minutes_below_12'].mean()
                    control_min = week_data[week_data['house_group'] == 'control']['minutes_below_12'].mean()
                    
                    print(f"  Week {week} - {location.capitalize()}:")
                    print(f"    Active:  {active_pct:.1f}% ({active_min:.0f} min/week)")
                    print(f"    Control: {control_pct:.1f}% ({control_min:.0f} min/week)")
                    
                    # Statistical test result
                    stat_result = comparison_results[
                        (comparison_results['year'] == year) &
                        (comparison_results['week'] == week) &
                        (comparison_results['location'] == location) &
                        (comparison_results['metric'] == 'pct_below_12')
                    ]
                    
                    if not stat_result.empty:
                        p_val = stat_result.iloc[0]['p_value']
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        print(f"    Difference: {active_pct - control_pct:.1f}% (p={p_val:.4f} {sig})")
    
    print("\n" + "-"*60)
    print("WHO RECOMMENDED RANGE (18-24°C)")
    print("-"*60)
    
    for year in [2008, 2009]:
        print(f"\nYear {year}:")
        year_data = overall_exposure[overall_exposure['year'] == year]
        
        for location in ['living', 'bedroom']:
            loc_data = year_data[year_data['location'] == location]
            
            for week in [1, 2]:
                week_data = loc_data[loc_data['week'] == week]
                
                if not week_data.empty:
                    active_pct = week_data[week_data['house_group'] == 'active']['pct_18_to_24'].mean()
                    control_pct = week_data[week_data['house_group'] == 'control']['pct_18_to_24'].mean()
                    
                    active_min = week_data[week_data['house_group'] == 'active']['minutes_in_who'].mean()
                    control_min = week_data[week_data['house_group'] == 'control']['minutes_in_who'].mean()
                    
                    print(f"  Week {week} - {location.capitalize()}:")
                    print(f"    Active:  {active_pct:.1f}% ({active_min:.0f} min/week)")
                    print(f"    Control: {control_pct:.1f}% ({control_min:.0f} min/week)")
                    
                    # Statistical test result
                    stat_result = comparison_results[
                        (comparison_results['year'] == year) &
                        (comparison_results['week'] == week) &
                        (comparison_results['location'] == location) &
                        (comparison_results['metric'] == 'pct_18_to_24')
                    ]
                    
                    if not stat_result.empty:
                        p_val = stat_result.iloc[0]['p_value']
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        print(f"    Difference: {active_pct - control_pct:.1f}% (p={p_val:.4f} {sig})")
    
    # Nighttime specific analysis
    print("\n" + "-"*60)
    print("NIGHTTIME EXTREME COLD EXPOSURE")
    print("-"*60)
    
    night_exposure = exposure_df[exposure_df['period'] == 'night']
    
    if not night_exposure.empty:
        for year in [2008, 2009]:
            print(f"\nYear {year}:")
            year_data = night_exposure[night_exposure['year'] == year]
            
            for location in ['bedroom']:  # Focus on bedroom at night
                loc_data = year_data[year_data['location'] == location]
                
                for week in [1, 2]:
                    week_data = loc_data[loc_data['week'] == week]
                    
                    if not week_data.empty:
                        active_pct = week_data[week_data['house_group'] == 'active']['pct_below_12'].mean()
                        control_pct = week_data[week_data['house_group'] == 'control']['pct_below_12'].mean()
                        
                        active_events = week_data[week_data['house_group'] == 'active']['extreme_cold_events'].mean()
                        control_events = week_data[week_data['house_group'] == 'control']['extreme_cold_events'].mean()
                        
                        print(f"  Week {week} - {location.capitalize()} (Night only):")
                        print(f"    Active:  {active_pct:.1f}% ({active_events:.1f} events)")
                        print(f"    Control: {control_pct:.1f}% ({control_events:.1f} events)")
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    create_temperature_distribution_plot(all_temperature_data)
    print("✓ Temperature distribution plots created")
    
    create_exposure_comparison_bars(overall_exposure)
    print("✓ Exposure comparison bars created")
    
    create_temporal_exposure_plot(temporal_df)
    print("✓ Temporal pattern plots created")
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # 1. Extreme cold exposure
    print("\n1. EXTREME COLD EXPOSURE (<12°C):")
    cold_sig = comparison_results[
        (comparison_results['metric'] == 'pct_below_12') & 
        (comparison_results['significant'] == True)
    ]
    if not cold_sig.empty:
        print(f"   Found {len(cold_sig)} significant differences:")
        for _, row in cold_sig.iterrows():
            diff = row['difference']
            direction = "more" if diff > 0 else "less"
            print(f"   - {row['year']} W{row['week']} {row['location']}: Active {abs(diff):.1f}% {direction} exposure")
    else:
        print("   No significant differences in extreme cold exposure")
    
    # 2. WHO range exposure
    print("\n2. WHO RECOMMENDED RANGE (18-24°C):")
    who_sig = comparison_results[
        (comparison_results['metric'] == 'pct_18_to_24') & 
        (comparison_results['significant'] == True)
    ]
    if not who_sig.empty:
        print(f"   Found {len(who_sig)} significant differences:")
        for _, row in who_sig.iterrows():
            diff = row['difference']
            direction = "more" if diff > 0 else "less"
            print(f"   - {row['year']} W{row['week']} {row['location']}: Active {abs(diff):.1f}% {direction} time in range")
    
    # 3. Temperature variability
    print("\n3. TEMPERATURE VARIABILITY:")
    if 'range' in overall_exposure.columns:
        active_range = overall_exposure[overall_exposure['house_group'] == 'active']['range'].mean()
        control_range = overall_exposure[overall_exposure['house_group'] == 'control']['range'].mean()
        print(f"   Average temperature range:")
        print(f"   - Active houses:  {active_range:.1f}°C")
        print(f"   - Control houses: {control_range:.1f}°C")
    
    # 4. Extreme events
    print("\n4. EXTREME TEMPERATURE EVENTS:")
    if 'extreme_cold_events' in overall_exposure.columns:
        active_events = overall_exposure[overall_exposure['house_group'] == 'active']['extreme_cold_events'].mean()
        control_events = overall_exposure[overall_exposure['house_group'] == 'control']['extreme_cold_events'].mean()
        print(f"   Average cold events (>30 min below 12°C):")
        print(f"   - Active houses:  {active_events:.1f} events/week")
        print(f"   - Control houses: {control_events:.1f} events/week")
    
    # 5. Special note about 2008 Week 1
    baseline_data = overall_exposure[(overall_exposure['year'] == 2008) & (overall_exposure['week'] == 1)]
    if not baseline_data.empty:
        print("\n5. BASELINE COMPARISON (2008 Week 1 - No ventilation):")
        
        for metric, description in [('pct_below_12', 'below 12°C'), ('pct_18_to_24', 'in WHO range')]:
            active_val = baseline_data[baseline_data['house_group'] == 'active'][metric].mean()
            control_val = baseline_data[baseline_data['house_group'] == 'control'][metric].mean()
            print(f"   Time {description}:")
            print(f"   - Future active houses:  {active_val:.1f}%")
            print(f"   - Control houses:        {control_val:.1f}%")
            print(f"   - Difference:            {active_val - control_val:.1f}%")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*80)
    
    return exposure_df, temporal_df, comparison_results


if __name__ == "__main__":
    exposure_df, temporal_df, comparison_results = main()