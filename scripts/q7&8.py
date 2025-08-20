#!/usr/bin/env python
"""
Questions 7 & 8 ENHANCED: Comprehensive Relative Humidity Analysis

Question 7: Was the relative humidity level in the active homes lower than in the control homes for the whole day?
Question 8: Was the relative humidity level lower in the active homes than in the control homes for part of the day?

Analysis includes:
- Full day (24-hour) RH comparisons
- Solar-based daytime vs nighttime RH comparisons
- Living room and bedroom separate analyses
- Dehumidifier usage impact analysis
- Humidity ratio calculations (absolute humidity)
- Histograms, box plots, time series
- Statistical tests with normality checking
- Effect of ventilation system on moisture levels
- Year-to-year comparisons
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from scipy import stats
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
        calculate_weekly_average,
        calculate_humidity_ratio,
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
        RH_THRESHOLDS
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

# ============================================================================
# CONSTANTS
# ============================================================================

# Dehumidifier usage (from report)
DEHUMIDIFIER_USERS_2008 = {
    'active_before': 13,  # Out of 20 active homes before installation
    'active_after': 0,    # After installation
    'control': 8          # Out of 10 control homes
}

# Heavy dehumidifier users in control group (from Figure 5.22)
HEAVY_DEHUMIDIFIER_USERS = ['C4', 'C5', 'C6', 'C8']
MODERATE_DEHUMIDIFIER_USERS = ['C1']
LOW_DEHUMIDIFIER_USERS = ['C2', 'C10']
NO_DEHUMIDIFIER_USERS = ['C3', 'C7', 'C9']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_house_group(house_num: int) -> str:
    """Get the house group (control or active) based on house number."""
    return 'control' if house_num in CONTROL_HOUSES else 'active'


def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval with normality check."""
    clean_data = np.array([x for x in data if not np.isnan(x)])
    
    if len(clean_data) < 3:
        return np.nan, np.nan, np.nan, False
    
    mean = np.mean(clean_data)
    
    # Test normality
    norm_test = normality_test(clean_data)
    is_normal = norm_test['is_normal']
    
    if is_normal and len(clean_data) >= 30:
        # Use parametric method
        sem = stats.sem(clean_data)
        ci = stats.t.interval(confidence, len(clean_data)-1, loc=mean, scale=sem)
    else:
        # Use bootstrap for non-normal or small samples
        n_bootstrap = 1000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(clean_data, size=len(clean_data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        ci = np.percentile(bootstrap_means, [alpha/2 * 100, (1 - alpha/2) * 100])
    
    return mean, ci[0], ci[1], is_normal


def classify_dehumidifier_usage(house_num: int) -> str:
    """Classify dehumidifier usage for control homes."""
    if house_num not in CONTROL_HOUSES:
        return 'N/A'
    
    house_code = f'C{CONTROL_HOUSES.index(house_num) + 1}'
    
    if house_code in HEAVY_DEHUMIDIFIER_USERS:
        return 'heavy'
    elif house_code in MODERATE_DEHUMIDIFIER_USERS:
        return 'moderate'
    elif house_code in LOW_DEHUMIDIFIER_USERS:
        return 'low'
    elif house_code in NO_DEHUMIDIFIER_USERS:
        return 'none'
    else:
        return 'unknown'


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_rh_by_location_and_period(week: int, year: int, house_num: int) -> dict:
    """
    Analyze relative humidity for a house by location (living/bedroom) and time period.
    Also calculates humidity ratio (absolute humidity).
    """
    df = load_house_data(week, year, house_num)
    
    if df.empty:
        return {}
    
    results = {
        'house_num': house_num,
        'week': week,
        'year': 2000 + year,
        'house_group': get_house_group(house_num),
        'recording_status': classify_house_status(house_num, week, year),
        'dehumidifier_usage': classify_dehumidifier_usage(house_num) if house_num in CONTROL_HOUSES else 'N/A'
    }
    
    # Analyze each location
    for location, location_name in [('L', 'living'), ('B', 'bedroom')]:
        rh_col = get_column_name('RH', location, week, year)
        temp_col = get_column_name('T', location, week, year)
        
        if rh_col not in df.columns:
            continue
        
        # Overall (24-hour) RH statistics
        rh_data = df[rh_col].dropna()
        if len(rh_data) > 0:
            mean, ci_lower, ci_upper, is_normal = calculate_confidence_interval(rh_data.values)
            results[f'{location_name}_overall_rh_mean'] = mean
            results[f'{location_name}_overall_rh_std'] = rh_data.std()
            results[f'{location_name}_overall_rh_median'] = rh_data.median()
            results[f'{location_name}_overall_rh_min'] = rh_data.min()
            results[f'{location_name}_overall_rh_max'] = rh_data.max()
            results[f'{location_name}_overall_rh_q25'] = rh_data.quantile(0.25)
            results[f'{location_name}_overall_rh_q75'] = rh_data.quantile(0.75)
            results[f'{location_name}_overall_rh_n'] = len(rh_data)
            results[f'{location_name}_overall_rh_ci_lower'] = ci_lower
            results[f'{location_name}_overall_rh_ci_upper'] = ci_upper
            results[f'{location_name}_overall_rh_is_normal'] = is_normal
            
            # Time in ASHRAE comfort range (30-60%)
            in_comfort = ((rh_data >= RH_THRESHOLDS['ashrae_min']) & 
                         (rh_data <= RH_THRESHOLDS['ashrae_max']))
            results[f'{location_name}_pct_comfort_rh'] = in_comfort.mean() * 100
            
            # Time above 60% (potential condensation risk)
            above_60 = rh_data > RH_THRESHOLDS['ashrae_max']
            results[f'{location_name}_pct_above_60'] = above_60.mean() * 100
            
            # Time above 70% (high condensation risk)
            above_70 = rh_data > 70
            results[f'{location_name}_pct_above_70'] = above_70.mean() * 100
        
        # Calculate humidity ratio if temperature is available
        if temp_col in df.columns and rh_col in df.columns:
            temp_data = df[temp_col]
            rh_for_calc = df[rh_col]
            
            # Calculate humidity ratio for valid pairs
            valid_mask = ~(temp_data.isna() | rh_for_calc.isna())
            if valid_mask.sum() > 0:
                humidity_ratios = []
                for t, rh in zip(temp_data[valid_mask], rh_for_calc[valid_mask]):
                    hr = calculate_humidity_ratio(t, rh)
                    humidity_ratios.append(hr)
                
                results[f'{location_name}_overall_hr_mean'] = np.mean(humidity_ratios)
                results[f'{location_name}_overall_hr_std'] = np.std(humidity_ratios)
        
        # Split by solar irradiance if available
        if 'ext__SR' in df.columns:
            daytime_df, nighttime_df = split_by_time_period(df, SOLAR_DAY_NIGHT_THRESHOLD)
            
            # Daytime RH statistics
            if not daytime_df.empty and rh_col in daytime_df.columns:
                day_rh = daytime_df[rh_col].dropna()
                if len(day_rh) > 0:
                    mean, ci_lower, ci_upper, is_normal = calculate_confidence_interval(day_rh.values)
                    results[f'{location_name}_day_rh_mean'] = mean
                    results[f'{location_name}_day_rh_std'] = day_rh.std()
                    results[f'{location_name}_day_rh_median'] = day_rh.median()
                    results[f'{location_name}_day_rh_min'] = day_rh.min()
                    results[f'{location_name}_day_rh_max'] = day_rh.max()
                    results[f'{location_name}_day_rh_n'] = len(day_rh)
                    results[f'{location_name}_day_rh_hours'] = len(day_rh) * 2 / 60
                    results[f'{location_name}_day_rh_is_normal'] = is_normal
                    
                    # Comfort range during day
                    in_comfort_day = ((day_rh >= RH_THRESHOLDS['ashrae_min']) & 
                                     (day_rh <= RH_THRESHOLDS['ashrae_max']))
                    results[f'{location_name}_day_pct_comfort'] = in_comfort_day.mean() * 100
                    
                    # Calculate daytime humidity ratio
                    if temp_col in daytime_df.columns:
                        day_temp = daytime_df[temp_col].dropna()
                        if len(day_temp) > 0 and len(day_rh) > 0:
                            # Match indices
                            common_idx = day_temp.index.intersection(day_rh.index)
                            if len(common_idx) > 0:
                                day_hrs = []
                                for idx in common_idx:
                                    hr = calculate_humidity_ratio(day_temp[idx], day_rh[idx])
                                    day_hrs.append(hr)
                                results[f'{location_name}_day_hr_mean'] = np.mean(day_hrs)
            
            # Nighttime RH statistics
            if not nighttime_df.empty and rh_col in nighttime_df.columns:
                night_rh = nighttime_df[rh_col].dropna()
                if len(night_rh) > 0:
                    mean, ci_lower, ci_upper, is_normal = calculate_confidence_interval(night_rh.values)
                    results[f'{location_name}_night_rh_mean'] = mean
                    results[f'{location_name}_night_rh_std'] = night_rh.std()
                    results[f'{location_name}_night_rh_median'] = night_rh.median()
                    results[f'{location_name}_night_rh_min'] = night_rh.min()
                    results[f'{location_name}_night_rh_max'] = night_rh.max()
                    results[f'{location_name}_night_rh_n'] = len(night_rh)
                    results[f'{location_name}_night_rh_hours'] = len(night_rh) * 2 / 60
                    results[f'{location_name}_night_rh_is_normal'] = is_normal
                    
                    # Above 60% at night
                    above_60_night = night_rh > RH_THRESHOLDS['ashrae_max']
                    results[f'{location_name}_night_pct_above_60'] = above_60_night.mean() * 100
                    
                    # Above 70% at night (high risk)
                    above_70_night = night_rh > 70
                    results[f'{location_name}_night_pct_above_70'] = above_70_night.mean() * 100
                    
                    # Calculate nighttime humidity ratio
                    if temp_col in nighttime_df.columns:
                        night_temp = nighttime_df[temp_col].dropna()
                        if len(night_temp) > 0 and len(night_rh) > 0:
                            common_idx = night_temp.index.intersection(night_rh.index)
                            if len(common_idx) > 0:
                                night_hrs = []
                                for idx in common_idx:
                                    hr = calculate_humidity_ratio(night_temp[idx], night_rh[idx])
                                    night_hrs.append(hr)
                                results[f'{location_name}_night_hr_mean'] = np.mean(night_hrs)
    
    return results


def compare_groups_rh(active_data: pd.DataFrame, control_data: pd.DataFrame,
                      location: str, period: str, metric: str, 
                      week: int, year: int) -> dict:
    """
    Compare RH or humidity ratio between active and control groups.
    
    Parameters:
    -----------
    metric : str
        'rh' for relative humidity or 'hr' for humidity ratio
    """
    # Construct column name
    if metric == 'rh':
        col_name = f'{location}_{period}_rh_mean'
    elif metric == 'hr':
        col_name = f'{location}_{period}_hr_mean'
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    active_values = active_data[col_name].dropna().values if col_name in active_data.columns else np.array([])
    control_values = control_data[col_name].dropna().values if col_name in control_data.columns else np.array([])
    
    if len(active_values) == 0 or len(control_values) == 0:
        return {'error': 'Insufficient data'}
    
    results = {
        'location': location,
        'period': period,
        'metric': metric,
        'week': week,
        'year': 2000 + year,
        'active_mean': np.mean(active_values),
        'active_std': np.std(active_values, ddof=1),
        'active_median': np.median(active_values),
        'active_n': len(active_values),
        'control_mean': np.mean(control_values),
        'control_std': np.std(control_values, ddof=1),
        'control_median': np.median(control_values),
        'control_n': len(control_values),
        'mean_difference': np.mean(active_values) - np.mean(control_values)
    }
    
    # Test assumptions
    active_norm = normality_test(active_values)
    control_norm = normality_test(control_values)
    variance_test = levene_test([active_values, control_values])
    
    results['active_normal'] = active_norm['is_normal']
    results['control_normal'] = control_norm['is_normal']
    results['equal_variance'] = variance_test['equal_variance']
    
    # Perform appropriate test
    if active_norm['is_normal'] and control_norm['is_normal']:
        test_result = independent_t_test(
            active_values, control_values,
            equal_var=variance_test['equal_variance']
        )
        results['test_type'] = 'Independent t-test' if variance_test['equal_variance'] else "Welch's t-test"
    else:
        test_result = mann_whitney_u(active_values, control_values)
        results['test_type'] = 'Mann-Whitney U test'
    
    results.update(test_result)
    
    # Interpret significance
    results['significant'] = test_result['p_value'] < 0.05
    results['significance_level'] = (
        '***' if test_result['p_value'] < 0.001 else
        '**' if test_result['p_value'] < 0.01 else
        '*' if test_result['p_value'] < 0.05 else
        'ns'
    )
    
    return results


def analyze_dehumidifier_impact(house_df: pd.DataFrame):
    """Analyze the impact of dehumidifier usage on RH levels in control homes."""
    print("\n" + "="*80)
    print("DEHUMIDIFIER USAGE IMPACT ANALYSIS")
    print("="*80)
    
    # Filter control homes
    control_df = house_df[house_df['house_group'] == 'control'].copy()
    
    if control_df.empty:
        print("No control home data available")
        return pd.DataFrame()
    
    # Group by dehumidifier usage
    usage_groups = {}
    for usage_type in ['heavy', 'moderate', 'low', 'none']:
        group_data = control_df[control_df['dehumidifier_usage'] == usage_type]
        if not group_data.empty:
            usage_groups[usage_type] = group_data
    
    # Analyze each location and period
    results = []
    for location in ['living', 'bedroom']:
        for period in ['overall', 'day', 'night']:
            col_name = f'{location}_{period}_rh_mean'
            
            if col_name not in control_df.columns:
                continue
            
            row = {
                'Location': location.capitalize(),
                'Period': period.capitalize()
            }
            
            for usage_type, group_data in usage_groups.items():
                if col_name in group_data.columns:
                    values = group_data[col_name].dropna()
                    if len(values) > 0:
                        row[f'{usage_type.capitalize()} (mean)'] = values.mean()
                        row[f'{usage_type.capitalize()} (n)'] = len(values)
            
            results.append(row)
    
    if results:
        results_df = pd.DataFrame(results)
        print("\nRelative Humidity by Dehumidifier Usage (Control Homes):")
        print(results_df.to_string(index=False))
        
        # Calculate overall impact
        print("\n" + "-"*60)
        print("Overall Impact:")
        
        for location in ['living', 'bedroom']:
            col_name = f'{location}_overall_rh_mean'
            if col_name in control_df.columns:
                heavy_users = control_df[control_df['dehumidifier_usage'] == 'heavy'][col_name].mean()
                no_users = control_df[control_df['dehumidifier_usage'] == 'none'][col_name].mean()
                
                if not np.isnan(heavy_users) and not np.isnan(no_users):
                    diff = no_users - heavy_users
                    print(f"  {location.capitalize()}: Heavy users {diff:.1f}% lower RH than non-users")
        
        return results_df
    
    return pd.DataFrame()


def create_rh_histograms(house_df: pd.DataFrame):
    """Create RH distribution histograms."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    
    for year_idx, year in enumerate([2008, 2009]):
        for location_idx, location in enumerate(['living', 'bedroom']):
            row_idx = year_idx * 2 + location_idx
            
            for period_idx, period in enumerate(['overall', 'day', 'night']):
                ax = axes[row_idx, period_idx]
                
                # Get data for this combination
                year_data = house_df[house_df['year'] == year]
                col_name = f'{location}_{period}_rh_mean'
                
                if col_name in year_data.columns:
                    active_data = year_data[year_data['house_group'] == 'active'][col_name].dropna()
                    control_data = year_data[year_data['house_group'] == 'control'][col_name].dropna()
                    
                    if len(active_data) > 0 and len(control_data) > 0:
                        # Determine bin edges
                        all_rh = pd.concat([active_data, control_data])
                        bins = np.linspace(all_rh.min(), all_rh.max(), 20)
                        
                        # Plot histograms
                        ax.hist(active_data, bins=bins, alpha=0.6, label='Active houses', 
                               color=COLORS.get('active', '#4ECDC4'), density=True)
                        ax.hist(control_data, bins=bins, alpha=0.6, label='Control houses', 
                               color=COLORS.get('control', '#FF6B6B'), density=True)
                        
                        # Add vertical lines for means
                        ax.axvline(active_data.mean(), color=COLORS.get('active', '#4ECDC4'), 
                                  linestyle='--', linewidth=2, alpha=0.8)
                        ax.axvline(control_data.mean(), color=COLORS.get('control', '#FF6B6B'), 
                                  linestyle='--', linewidth=2, alpha=0.8)
                        
                        # Add ASHRAE comfort thresholds
                        ax.axvline(RH_THRESHOLDS['ashrae_min'], color='green', 
                                  linestyle=':', linewidth=1, alpha=0.5)
                        ax.axvline(RH_THRESHOLDS['ashrae_max'], color='green', 
                                  linestyle=':', linewidth=1, alpha=0.5)
                        
                        ax.set_xlabel('Relative Humidity (%)')
                        ax.set_ylabel('Density')
                        ax.set_title(f'{year} - {location.capitalize()} - {period.capitalize()}')
                        if row_idx == 0 and period_idx == 0:
                            ax.legend(fontsize=8)
                        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Relative Humidity Distributions: Active vs Control Homes', fontsize=14, y=1.01)
    plt.tight_layout()
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'question_07_08_rh_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_comprehensive_rh_boxplots(house_df: pd.DataFrame):
    """Create comprehensive box plots for RH."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    for year_idx, year in enumerate([2008, 2009]):
        year_data = house_df[house_df['year'] == year]
        
        for week_idx, week in enumerate([1, 2]):
            week_data = year_data[year_data['week'] == week]
            
            if week_data.empty:
                continue
            
            for loc_idx, location in enumerate(['living', 'bedroom']):
                ax = axes[loc_idx, year_idx * 2 + week_idx]
                
                # Prepare data for box plot
                plot_data = []
                labels = []
                positions = []
                colors = []
                pos = 1
                
                for period in ['overall', 'day', 'night']:
                    for house_group in ['active', 'control']:
                        col_name = f'{location}_{period}_rh_mean'
                        if col_name in week_data.columns:
                            group_data = week_data[week_data['house_group'] == house_group][col_name].dropna()
                            if len(group_data) > 0:
                                plot_data.append(group_data.values)
                                labels.append(f"{house_group[:3].upper()}\n{period[:3]}")
                                positions.append(pos)
                                colors.append(COLORS.get(house_group, '#888888'))
                                pos += 1
                    pos += 0.5  # Space between periods
                
                if plot_data:
                    bp = ax.boxplot(plot_data, positions=positions, widths=0.6, 
                                   patch_artist=True, showmeans=True,
                                   meanprops=dict(marker='D', markerfacecolor='white', 
                                                markeredgecolor='black', markersize=5))
                    
                    # Color the boxes
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    # Add ASHRAE range reference lines
                    ax.axhline(y=RH_THRESHOLDS['ashrae_min'], color='green', 
                              linestyle=':', linewidth=1, alpha=0.5)
                    ax.axhline(y=RH_THRESHOLDS['ashrae_max'], color='green', 
                              linestyle=':', linewidth=1, alpha=0.5)
                    ax.axhline(y=70, color='red', linestyle=':', linewidth=1, alpha=0.5)
                    
                    # Set labels
                    ax.set_xticks(positions)
                    ax.set_xticklabels(labels, fontsize=7)
                    ax.set_ylabel('Relative Humidity (%)')
                    
                    # Title with clarification
                    title = f'{location.capitalize()} - {year} W{week}'
                    if year == 2008 and week == 1:
                        title += '\n(Baseline)'
                    elif year == 2008 and week == 2:
                        title += '\n(Ventilation installed)'
                    ax.set_title(title, fontsize=10)
                    
                    # Fix y-axis
                    all_data = [item for sublist in plot_data for item in sublist]
                    ax.set_ylim(bottom=min(all_data) - 2, top=min(max(all_data) + 2, 100))
                    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Relative Humidity by Location, Period, and House Group', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'question_07_08_rh_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_humidity_ratio_comparison(house_df: pd.DataFrame):
    """Create plots comparing humidity ratio (absolute humidity) between groups."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for year_idx, year in enumerate([2008, 2009]):
        year_data = house_df[house_df['year'] == year]
        
        for location_idx, location in enumerate(['living', 'bedroom']):
            ax = axes[year_idx, location_idx]
            
            # Collect humidity ratio data
            hr_overall = []
            hr_day = []
            hr_night = []
            groups = []
            
            for house_group in ['active', 'control']:
                group_data = year_data[year_data['house_group'] == house_group]
                
                # Overall humidity ratio
                col_overall = f'{location}_overall_hr_mean'
                if col_overall in group_data.columns:
                    values = group_data[col_overall].dropna()
                    if len(values) > 0:
                        hr_overall.extend(values)
                        groups.extend([house_group] * len(values))
                
                # Day/night if available
                col_day = f'{location}_day_hr_mean'
                col_night = f'{location}_night_hr_mean'
                
                if col_day in group_data.columns:
                    day_values = group_data[col_day].dropna()
                    if len(day_values) > 0:
                        hr_day.extend(day_values)
                
                if col_night in group_data.columns:
                    night_values = group_data[col_night].dropna()
                    if len(night_values) > 0:
                        hr_night.extend(night_values)
            
            if hr_overall:
                # Create DataFrame for plotting
                plot_df = pd.DataFrame({
                    'Humidity Ratio (g/kg)': hr_overall,
                    'Group': groups
                })
                
                # Box plot
                bp = ax.boxplot([plot_df[plot_df['Group'] == 'active']['Humidity Ratio (g/kg)'].values,
                                 plot_df[plot_df['Group'] == 'control']['Humidity Ratio (g/kg)'].values],
                                labels=['Active', 'Control'],
                                patch_artist=True,
                                showmeans=True)
                
                # Color the boxes
                bp['boxes'][0].set_facecolor(COLORS.get('active', '#4ECDC4'))
                bp['boxes'][1].set_facecolor(COLORS.get('control', '#FF6B6B'))
                
                for box in bp['boxes']:
                    box.set_alpha(0.7)
                
                ax.set_ylabel('Humidity Ratio (g water/kg dry air)')
                ax.set_title(f'{year} - {location.capitalize()} Room')
                ax.grid(True, alpha=0.3)
    
    plt.suptitle('Absolute Humidity (Humidity Ratio) Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'question_07_08_humidity_ratio.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_ventilation_effect_rh(house_df: pd.DataFrame):
    """Analyze the effect of ventilation installation on RH (Week 1 vs Week 2 in 2008)."""
    print("\n" + "="*80)
    print("VENTILATION SYSTEM EFFECT ON RELATIVE HUMIDITY (2008 Week 1 vs Week 2)")
    print("="*80)
    
    # Get active houses data for 2008
    active_2008 = house_df[(house_df['year'] == 2008) & (house_df['house_group'] == 'active')]
    
    week1 = active_2008[active_2008['week'] == 1]
    week2 = active_2008[active_2008['week'] == 2]
    
    # Match houses present in both weeks
    common_houses = set(week1['house_num']) & set(week2['house_num'])
    
    results = []
    for location in ['living', 'bedroom']:
        for period in ['overall', 'day', 'night']:
            # Test RH change
            rh_col = f'{location}_{period}_rh_mean'
            
            if rh_col not in week1.columns:
                continue
            
            # Get paired data
            w1_rh = []
            w2_rh = []
            
            for house in common_houses:
                w1_val = week1[week1['house_num'] == house][rh_col].values
                w2_val = week2[week2['house_num'] == house][rh_col].values
                
                if len(w1_val) > 0 and len(w2_val) > 0:
                    w1_rh.append(w1_val[0])
                    w2_rh.append(w2_val[0])
            
            if len(w1_rh) >= 3:
                # Perform paired t-test
                test_result = paired_t_test(w1_rh, w2_rh)
                
                results.append({
                    'Location': location.capitalize(),
                    'Period': period.capitalize(),
                    'Metric': 'RH (%)',
                    'Week 1 Mean': np.mean(w1_rh),
                    'Week 2 Mean': np.mean(w2_rh),
                    'Change': np.mean(w2_rh) - np.mean(w1_rh),
                    'P-value': test_result['p_value'],
                    'Significant': test_result['p_value'] < 0.05,
                    'N pairs': len(w1_rh)
                })
            
            # Test humidity ratio change if available
            hr_col = f'{location}_{period}_hr_mean'
            if hr_col in week1.columns:
                w1_hr = []
                w2_hr = []
                
                for house in common_houses:
                    w1_val = week1[week1['house_num'] == house][hr_col].values
                    w2_val = week2[week2['house_num'] == house][hr_col].values
                    
                    if len(w1_val) > 0 and len(w2_val) > 0:
                        w1_hr.append(w1_val[0])
                        w2_hr.append(w2_val[0])
                
                if len(w1_hr) >= 3:
                    test_result = paired_t_test(w1_hr, w2_hr)
                    
                    results.append({
                        'Location': location.capitalize(),
                        'Period': period.capitalize(),
                        'Metric': 'HR (g/kg)',
                        'Week 1 Mean': np.mean(w1_hr),
                        'Week 2 Mean': np.mean(w2_hr),
                        'Change': np.mean(w2_hr) - np.mean(w1_hr),
                        'P-value': test_result['p_value'],
                        'Significant': test_result['p_value'] < 0.05,
                        'N pairs': len(w1_hr)
                    })
    
    if results:
        results_df = pd.DataFrame(results)
        print("\nPaired Analysis - Before vs After Ventilation Installation (Active Houses):")
        print(results_df.to_string(index=False))
        
        # Save results
        results_df.to_csv(TABLES_DIR / 'ventilation_effect_rh_analysis.csv', index=False)
        
        # Summary
        sig_results = results_df[results_df['Significant']]
        if not sig_results.empty:
            print(f"\n{len(sig_results)} significant changes found after ventilation installation:")
            for _, row in sig_results.iterrows():
                direction = "increased" if row['Change'] > 0 else "decreased"
                print(f"  - {row['Location']} {row['Period']} {row['Metric']}: "
                      f"{direction} by {abs(row['Change']):.2f} (p={row['P-value']:.4f})")
    
    return results_df if results else pd.DataFrame()


def print_comprehensive_rh_summary(comparison_results: pd.DataFrame, house_df: pd.DataFrame):
    """Print comprehensive summary of RH results."""
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY - QUESTIONS 7 & 8")
    print("="*80)
    
    # Question 7: Full day RH comparison
    print("\n" + "-"*60)
    print("QUESTION 7: FULL DAY (24-HOUR) RELATIVE HUMIDITY COMPARISON")
    print("-"*60)
    
    overall_rh = comparison_results[(comparison_results['period'] == 'overall') & 
                                    (comparison_results['metric'] == 'rh')]
    
    for year in [2008, 2009]:
        year_data = overall_rh[overall_rh['year'] == year]
        if year_data.empty:
            continue
        
        print(f"\nYear {year}:")
        for location in ['living', 'bedroom']:
            loc_data = year_data[year_data['location'] == location]
            
            for week in [1, 2]:
                week_data = loc_data[loc_data['week'] == week]
                if not week_data.empty:
                    row = week_data.iloc[0]
                    print(f"  Week {week} - {location.capitalize()}:")
                    print(f"    Active:  {row['active_mean']:.1f} ± {row['active_std']:.1f}%")
                    print(f"    Control: {row['control_mean']:.1f} ± {row['control_std']:.1f}%")
                    print(f"    Difference: {row['mean_difference']:.1f}% "
                          f"(Active {'lower' if row['mean_difference'] < 0 else 'higher'})")
                    print(f"    P-value: {row['p_value']:.4f} {row['significance_level']}")
    
    # Question 8: Time period RH comparison
    print("\n" + "-"*60)
    print("QUESTION 8: TIME PERIOD RELATIVE HUMIDITY COMPARISON")
    print("-"*60)
    
    period_rh = comparison_results[(comparison_results['period'].isin(['day', 'night'])) & 
                                   (comparison_results['metric'] == 'rh')]
    
    for year in [2008, 2009]:
        year_data = period_rh[period_rh['year'] == year]
        if year_data.empty:
            continue
        
        print(f"\nYear {year}:")
        
        for location in ['living', 'bedroom']:
            loc_data = year_data[year_data['location'] == location]
            
            day_data = loc_data[loc_data['period'] == 'day']
            night_data = loc_data[loc_data['period'] == 'night']
            
            if not day_data.empty and not night_data.empty:
                print(f"\n  {location.capitalize()} Room:")
                
                for week in [1, 2]:
                    day_week = day_data[day_data['week'] == week]
                    night_week = night_data[night_data['week'] == week]
                    
                    if not day_week.empty and not night_week.empty:
                        day_row = day_week.iloc[0]
                        night_row = night_week.iloc[0]
                        
                        print(f"    Week {week}:")
                        print(f"      Daytime:   Active vs Control = {day_row['mean_difference']:.1f}% "
                              f"(p={day_row['p_value']:.4f})")
                        print(f"      Nighttime: Active vs Control = {night_row['mean_difference']:.1f}% "
                              f"(p={night_row['p_value']:.4f})")
                        
                        # Note significant differences
                        if day_row['significant'] or night_row['significant']:
                            sig_period = []
                            if day_row['significant']:
                                sig_period.append('daytime')
                            if night_row['significant']:
                                sig_period.append('nighttime')
                            print(f"      → Significant difference during {' and '.join(sig_period)}")
    
    # Humidity ratio comparison
    print("\n" + "-"*60)
    print("ABSOLUTE HUMIDITY (HUMIDITY RATIO) COMPARISON")
    print("-"*60)
    
    hr_results = comparison_results[comparison_results['metric'] == 'hr']
    
    if not hr_results.empty:
        for year in [2008, 2009]:
            year_data = hr_results[hr_results['year'] == year]
            if year_data.empty:
                continue
            
            print(f"\nYear {year} - Humidity Ratio (g water/kg dry air):")
            
            for location in ['living', 'bedroom']:
                loc_data = year_data[year_data['location'] == location]
                
                if not loc_data.empty:
                    overall_hr = loc_data[loc_data['period'] == 'overall']
                    if not overall_hr.empty:
                        row = overall_hr.iloc[0]
                        print(f"  {location.capitalize()}:")
                        print(f"    Active:  {row['active_mean']:.2f} g/kg")
                        print(f"    Control: {row['control_mean']:.2f} g/kg")
                        print(f"    Difference: {row['mean_difference']:.2f} g/kg")
    
    # Comfort zone analysis
    print("\n" + "-"*60)
    print("ASHRAE COMFORT ZONE (30-60% RH) COMPLIANCE")
    print("-"*60)
    
    for year in [2008, 2009]:
        year_data = house_df[house_df['year'] == year]
        if year_data.empty:
            continue
        
        print(f"\nYear {year}:")
        
        for location in ['living', 'bedroom']:
            comfort_col = f'{location}_pct_comfort_rh'
            above_60_col = f'{location}_pct_above_60'
            above_70_col = f'{location}_pct_above_70'
            
            if comfort_col in year_data.columns:
                active_comfort = year_data[year_data['house_group'] == 'active'][comfort_col].mean()
                control_comfort = year_data[year_data['house_group'] == 'control'][comfort_col].mean()
                
                print(f"\n  {location.capitalize()} - % time in comfort zone (30-60%):")
                print(f"    Active houses:  {active_comfort:.1f}%")
                print(f"    Control houses: {control_comfort:.1f}%")
            
            if above_60_col in year_data.columns:
                active_above = year_data[year_data['house_group'] == 'active'][above_60_col].mean()
                control_above = year_data[year_data['house_group'] == 'control'][above_60_col].mean()
                
                print(f"  {location.capitalize()} - % time above 60%:")
                print(f"    Active houses:  {active_above:.1f}%")
                print(f"    Control houses: {control_above:.1f}%")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run the complete Questions 7 & 8 analysis."""
    
    print("=" * 80)
    print("QUESTIONS 7 & 8: COMPREHENSIVE RELATIVE HUMIDITY ANALYSIS")
    print("=" * 80)
    print("\nQuestion 7: Was RH lower in active homes than control homes for the whole day?")
    print("Question 8: Was RH lower in active homes than control homes for part of the day?")
    print("=" * 80)
    
    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_house_data = []
    all_comparisons = []
    
    # Analyze each house
    for year in [8, 9]:
        for week in [1, 2]:
            print(f"\n{'='*60}")
            print(f"Analyzing Year 200{year}, Week {week}")
            if year == 8 and week == 1:
                print("(Baseline - No ventilation units installed)")
                print("Note: 13/20 active homes using dehumidifiers")
            elif year == 8 and week == 2:
                print("(After ventilation installation in active homes)")
                print("Note: 0/20 active homes using dehumidifiers")
            print(f"Note: {8}/10 control homes using dehumidifiers throughout study")
            print(f"{'='*60}")
            
            # Analyze each house
            for house_num in range(1, 31):
                house_results = analyze_rh_by_location_and_period(week, year, house_num)
                if house_results:
                    all_house_data.append(house_results)
            
            # Compare groups for this week
            week_df = pd.DataFrame([h for h in all_house_data 
                                   if h['week'] == week and h['year'] == 2000 + year])
            
            if not week_df.empty:
                active_df = week_df[week_df['house_group'] == 'active']
                control_df = week_df[week_df['house_group'] == 'control']
                
                # Compare each location, period, and metric
                for location in ['living', 'bedroom']:
                    for period in ['overall', 'day', 'night']:
                        # Compare relative humidity
                        rh_comparison = compare_groups_rh(
                            active_df, control_df, location, period, 'rh', week, year
                        )
                        if 'error' not in rh_comparison:
                            all_comparisons.append(rh_comparison)
                            
                            # Print immediate results
                            if period == 'overall':
                                print(f"\n{location.upper()} - FULL DAY RH:")
                            else:
                                print(f"\n{location.upper()} - {period.upper()}TIME RH:")
                            
                            print(f"  Active:  {rh_comparison['active_mean']:.1f} ± "
                                  f"{rh_comparison['active_std']:.1f}% "
                                  f"(median={rh_comparison['active_median']:.1f}, "
                                  f"n={rh_comparison['active_n']})")
                            print(f"  Control: {rh_comparison['control_mean']:.1f} ± "
                                  f"{rh_comparison['control_std']:.1f}% "
                                  f"(median={rh_comparison['control_median']:.1f}, "
                                  f"n={rh_comparison['control_n']})")
                            print(f"  Difference: {rh_comparison['mean_difference']:.1f}% "
                                  f"(Active {'lower' if rh_comparison['mean_difference'] < 0 else 'higher'})")
                            print(f"  Test: {rh_comparison['test_type']}, "
                                  f"p={rh_comparison['p_value']:.4f} "
                                  f"{rh_comparison['significance_level']}")
                        
                        # Compare humidity ratio
                        hr_comparison = compare_groups_rh(
                            active_df, control_df, location, period, 'hr', week, year
                        )
                        if 'error' not in hr_comparison:
                            all_comparisons.append(hr_comparison)
    
    # Create DataFrames
    house_df = pd.DataFrame(all_house_data)
    comparison_df = pd.DataFrame(all_comparisons)
    
    # Save results
    house_df.to_csv(TABLES_DIR / 'question_07_08_house_data.csv', index=False)
    comparison_df.to_csv(TABLES_DIR / 'question_07_08_comparisons.csv', index=False)
    
    # Analyze dehumidifier impact
    dehumidifier_df = analyze_dehumidifier_impact(house_df)
    if not dehumidifier_df.empty:
        dehumidifier_df.to_csv(TABLES_DIR / 'dehumidifier_impact_analysis.csv', index=False)
    
    # Analyze ventilation effect
    ventilation_effect_df = analyze_ventilation_effect_rh(house_df)
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    create_rh_histograms(house_df)
    print("✓ RH histograms created")
    
    create_comprehensive_rh_boxplots(house_df)
    print("✓ Comprehensive RH box plots created")
    
    create_humidity_ratio_comparison(house_df)
    print("✓ Humidity ratio comparison plots created")
    
    # Print comprehensive summary
    print_comprehensive_rh_summary(comparison_df, house_df)
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # 1. Overall RH differences
    print("\n1. OVERALL RELATIVE HUMIDITY (Question 7):")
    overall_sig = comparison_df[(comparison_df['period'] == 'overall') & 
                                (comparison_df['metric'] == 'rh') &
                                (comparison_df['significant'] == True)]
    if not overall_sig.empty:
        print(f"   Found {len(overall_sig)} significant differences in 24-hour RH:")
        for _, row in overall_sig.iterrows():
            print(f"   - {row['year']} Week {row['week']} {row['location']}: "
                  f"Active {'lower' if row['mean_difference'] < 0 else 'higher'} by "
                  f"{abs(row['mean_difference']):.1f}%")
    else:
        print("   No significant differences in 24-hour average RH")
        print("   Note: Large variation in control group due to different dehumidifier usage")
    
    # 2. Time period differences
    print("\n2. TIME PERIOD RH DIFFERENCES (Question 8):")
    period_sig = comparison_df[(comparison_df['period'].isin(['day', 'night'])) & 
                              (comparison_df['metric'] == 'rh') &
                              (comparison_df['significant'] == True)]
    if not period_sig.empty:
        print(f"   Found {len(period_sig)} significant differences by time period:")
        for _, row in period_sig.iterrows():
            print(f"   - {row['year']} Week {row['week']} {row['location']} {row['period']}: "
                  f"Active {'lower' if row['mean_difference'] < 0 else 'higher'} by "
                  f"{abs(row['mean_difference']):.1f}%")
    
    # 3. Ventilation effect on RH
    if not ventilation_effect_df.empty:
        sig_vent = ventilation_effect_df[ventilation_effect_df['Significant'] == True]
        if not sig_vent.empty:
            print(f"\n3. VENTILATION INSTALLATION EFFECT ON RH (2008 Week 1 vs 2):")
            print(f"   Found {len(sig_vent)} significant changes:")
            for _, row in sig_vent.iterrows():
                direction = "increased" if row['Change'] > 0 else "decreased"
                print(f"   - {row['Location']} {row['Period']} {row['Metric']}: "
                      f"{direction} by {abs(row['Change']):.2f}")
        else:
            print(f"\n3. VENTILATION EFFECT: No significant changes in RH levels")
            print("   Note: Active homes stopped using dehumidifiers after installation")
    
    # 4. Above comfort thresholds
    print("\n4. HUMIDITY ABOVE COMFORT THRESHOLDS:")
    
    # Calculate average time above 60%
    for location in ['living', 'bedroom']:
        col_name = f'{location}_pct_above_60'
        if col_name in house_df.columns:
            for year in [2008, 2009]:
                year_data = house_df[house_df['year'] == year]
                active_above = year_data[year_data['house_group'] == 'active'][col_name].mean()
                control_above = year_data[year_data['house_group'] == 'control'][col_name].mean()
                
                if not np.isnan(active_above) and not np.isnan(control_above):
                    print(f"   {location.capitalize()} above 60% RH ({year}): "
                          f"Active {active_above:.1f}% vs Control {control_above:.1f}%")
    
    # 5. Dehumidifier impact
    print("\n5. DEHUMIDIFIER USAGE IMPACT (Control Homes):")
    if not dehumidifier_df.empty:
        print("   Heavy users achieved lower RH than non-users")
        print("   This variation explains large confidence intervals in control group")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*80)
    
    return house_df, comparison_df


if __name__ == "__main__":
    house_df, comparison_df = main()