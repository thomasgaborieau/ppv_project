#!/usr/bin/env python
"""
Question 1 ENHANCED: Comprehensive outdoor temperature analysis
- Histograms of temperature distributions
- Fixed box plots (comparing house groups, not recording status)
- Aggregated results display with median and std
- Year-to-year comparisons with statistical tests
- Normality testing before confidence intervals
- Overall, daytime, and nighttime analyses
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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
        calculate_weekly_average,
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
        COLORS
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
# HELPER FUNCTIONS
# ============================================================================

def get_house_group(house_num: int) -> str:
    """
    Get the house group (control or active) based on house number.
    This is independent of recording status.
    """
    return 'control' if house_num in CONTROL_HOUSES else 'active'


def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval with normality check.
    If data is not normal, use bootstrap method.
    """
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


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def extract_outdoor_temperature(week: int, year: int, house_num: int) -> pd.DataFrame:
    """Extract outdoor temperature and solar irradiance for a specific house."""
    df = load_house_data(week, year, house_num)
    
    if df.empty:
        return pd.DataFrame()
    
    outdoor_cols = ['ext__T', 'ext__SR']
    available_cols = [col for col in outdoor_cols if col in df.columns]
    
    if not available_cols:
        return pd.DataFrame()
    
    outdoor_df = df[available_cols].copy()
    outdoor_df['house_num'] = house_num
    outdoor_df['week'] = week
    outdoor_df['year'] = 2000 + year
    outdoor_df['house_group'] = get_house_group(house_num)  # Use house group instead of status
    outdoor_df['recording_status'] = classify_house_status(house_num, week, year)
    
    return outdoor_df


def analyze_temperature_by_period(data: pd.DataFrame, 
                                 solar_threshold: float = SOLAR_DAY_NIGHT_THRESHOLD) -> dict:
    """Analyze temperature for overall, daytime, and nighttime periods with normality testing."""
    results = {}
    
    # Overall statistics
    temp_data = data['ext__T'].dropna()
    if len(temp_data) > 0:
        mean, ci_lower, ci_upper, is_normal = calculate_confidence_interval(temp_data.values)
        results['overall'] = {
            'mean': mean,
            'std': temp_data.std(),
            'median': temp_data.median(),
            'min': temp_data.min(),
            'max': temp_data.max(),
            'q25': temp_data.quantile(0.25),
            'q75': temp_data.quantile(0.75),
            'n': len(temp_data),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'is_normal': is_normal,
            'data': temp_data.values
        }
    
    # Split by solar irradiance if available
    if 'ext__SR' in data.columns:
        daytime_df, nighttime_df = split_by_time_period(data, solar_threshold)
        
        # Daytime statistics
        if not daytime_df.empty:
            day_temp = daytime_df['ext__T'].dropna()
            if len(day_temp) > 0:
                mean, ci_lower, ci_upper, is_normal = calculate_confidence_interval(day_temp.values)
                results['daytime'] = {
                    'mean': mean,
                    'std': day_temp.std(),
                    'median': day_temp.median(),
                    'min': day_temp.min(),
                    'max': day_temp.max(),
                    'q25': day_temp.quantile(0.25),
                    'q75': day_temp.quantile(0.75),
                    'n': len(day_temp),
                    'hours': len(day_temp) * 2 / 60,
                    'avg_solar': daytime_df['ext__SR'].mean(),
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'is_normal': is_normal,
                    'data': day_temp.values
                }
        
        # Nighttime statistics
        if not nighttime_df.empty:
            night_temp = nighttime_df['ext__T'].dropna()
            if len(night_temp) > 0:
                mean, ci_lower, ci_upper, is_normal = calculate_confidence_interval(night_temp.values)
                results['nighttime'] = {
                    'mean': mean,
                    'std': night_temp.std(),
                    'median': night_temp.median(),
                    'min': night_temp.min(),
                    'max': night_temp.max(),
                    'q25': night_temp.quantile(0.25),
                    'q75': night_temp.quantile(0.75),
                    'n': len(night_temp),
                    'hours': len(night_temp) * 2 / 60,
                    'avg_solar': nighttime_df['ext__SR'].mean(),
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'is_normal': is_normal,
                    'data': night_temp.values
                }
    
    return results


def compare_groups(active_data: list, control_data: list, 
                  period_name: str, week: int, year: int) -> dict:
    """Compare temperature between active and control house groups."""
    active_clean = [x for x in active_data if not np.isnan(x)]
    control_clean = [x for x in control_data if not np.isnan(x)]
    
    if not active_clean or not control_clean:
        return {'error': 'Insufficient data'}
    
    results = {
        'period': period_name,
        'week': week,
        'year': 2000 + year,
        'active_mean': np.mean(active_clean),
        'active_std': np.std(active_clean, ddof=1),
        'active_median': np.median(active_clean),
        'active_n': len(active_clean),
        'control_mean': np.mean(control_clean),
        'control_std': np.std(control_clean, ddof=1),
        'control_median': np.median(control_clean),
        'control_n': len(control_clean),
        'mean_difference': np.mean(control_clean) - np.mean(active_clean)
    }
    
    # Test assumptions
    active_norm = normality_test(active_clean)
    control_norm = normality_test(control_clean)
    variance_test = levene_test([active_clean, control_clean])
    
    results['active_normal'] = active_norm['is_normal']
    results['control_normal'] = control_norm['is_normal']
    results['equal_variance'] = variance_test['equal_variance']
    
    # Perform appropriate test
    if active_norm['is_normal'] and control_norm['is_normal']:
        test_result = independent_t_test(
            active_clean, control_clean,
            equal_var=variance_test['equal_variance']
        )
        results['test_type'] = 'Independent t-test' if variance_test['equal_variance'] else "Welch's t-test"
    else:
        test_result = mann_whitney_u(active_clean, control_clean)
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


def compare_years(data_2008: list, data_2009: list, description: str) -> dict:
    """Compare temperatures between 2008 and 2009."""
    clean_2008 = [x for x in data_2008 if not np.isnan(x)]
    clean_2009 = [x for x in data_2009 if not np.isnan(x)]
    
    if not clean_2008 or not clean_2009:
        return {'error': 'Insufficient data'}
    
    # Basic statistics
    results = {
        'comparison': description,
        'mean_2008': np.mean(clean_2008),
        'std_2008': np.std(clean_2008, ddof=1),
        'median_2008': np.median(clean_2008),
        'n_2008': len(clean_2008),
        'mean_2009': np.mean(clean_2009),
        'std_2009': np.std(clean_2009, ddof=1),
        'median_2009': np.median(clean_2009),
        'n_2009': len(clean_2009),
        'mean_difference': np.mean(clean_2008) - np.mean(clean_2009)
    }
    
    # Test normality
    norm_2008 = normality_test(clean_2008)
    norm_2009 = normality_test(clean_2009)
    results['normal_2008'] = norm_2008['is_normal']
    results['normal_2009'] = norm_2009['is_normal']
    
    # Statistical test
    if norm_2008['is_normal'] and norm_2009['is_normal']:
        test_result = independent_t_test(clean_2008, clean_2009)
        results['test_type'] = 't-test'
    else:
        test_result = mann_whitney_u(clean_2008, clean_2009)
        results['test_type'] = 'Mann-Whitney U'
    
    results.update(test_result)
    results['significant'] = test_result['p_value'] < 0.05
    
    return results


def create_histograms(all_data: dict):
    """Create temperature distribution histograms."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for year_idx, year in enumerate([2008, 2009]):
        for period_idx, period in enumerate(['overall', 'daytime', 'nighttime']):
            ax = axes[year_idx, period_idx]
            
            # Get data for this year and period
            active_data = []
            control_data = []
            
            for week in [1, 2]:
                key = f"{year}_week{week}_{period}"
                if key in all_data:
                    if 'active' in all_data[key]:
                        active_data.extend(all_data[key]['active'])
                    if 'control' in all_data[key]:
                        control_data.extend(all_data[key]['control'])
            
            if active_data and control_data:
                # Determine bin edges
                all_temps = active_data + control_data
                bins = np.linspace(min(all_temps), max(all_temps), 30)
                
                # Plot histograms
                ax.hist(active_data, bins=bins, alpha=0.6, label='Active houses', 
                       color=COLORS.get('active', '#4ECDC4'), density=True)
                ax.hist(control_data, bins=bins, alpha=0.6, label='Control houses', 
                       color=COLORS.get('control', '#FF6B6B'), density=True)
                
                # Add vertical lines for means
                ax.axvline(np.mean(active_data), color=COLORS.get('active', '#4ECDC4'), 
                          linestyle='--', linewidth=2, label=f'Active mean: {np.mean(active_data):.1f}°C')
                ax.axvline(np.mean(control_data), color=COLORS.get('control', '#FF6B6B'), 
                          linestyle='--', linewidth=2, label=f'Control mean: {np.mean(control_data):.1f}°C')
                
                ax.set_xlabel('Temperature (°C)')
                ax.set_ylabel('Density')
                ax.set_title(f'{year} - {period.capitalize()}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
    
    plt.suptitle('Temperature Distribution: Active vs Control Houses', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'question_01_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_fixed_boxplots(house_df: pd.DataFrame):
    """Create properly formatted box plots comparing house groups (not recording status)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for year_idx, year in enumerate([8, 9]):
        for week_idx, week in enumerate([1, 2]):
            ax = axes[year_idx, week_idx]
            
            week_data = house_df[(house_df['year'] == 2000 + year) & (house_df['week'] == week)]
            
            if not week_data.empty:
                # Prepare data for box plot
                plot_data = []
                labels = []
                positions = []
                colors = []
                pos = 1
                
                for period in ['overall', 'daytime', 'nighttime']:
                    for house_group in ['active', 'control']:
                        col_name = f'{period}_mean'
                        if col_name in week_data.columns:
                            # Use house_group instead of recording_status
                            group_data = week_data[week_data['house_group'] == house_group][col_name].dropna()
                            if len(group_data) > 0:
                                plot_data.append(group_data.values)
                                labels.append(f"{house_group.capitalize()}\nhouses\n{period[:3]}")
                                positions.append(pos)
                                colors.append(COLORS.get(house_group, '#888888'))
                                pos += 1
                    pos += 0.5  # Space between periods
                
                if plot_data:
                    bp = ax.boxplot(plot_data, positions=positions, widths=0.6, 
                                   patch_artist=True, showmeans=True,
                                   meanprops=dict(marker='D', markerfacecolor='white', 
                                                markeredgecolor='black', markersize=6))
                    
                    # Color the boxes
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    # Style whiskers and caps
                    for whisker in bp['whiskers']:
                        whisker.set(color='#8B8B8B', linewidth=1.25)
                    for cap in bp['caps']:
                        cap.set(color='#8B8B8B', linewidth=1.25)
                    
                    # Set labels
                    ax.set_xticks(positions)
                    ax.set_xticklabels(labels, fontsize=8)
                    ax.set_ylabel('Temperature (°C)')
                    
                    # Add title with clarification for Week 1, 2008
                    if year == 8 and week == 1:
                        ax.set_title(f'Year 200{year}, Week {week}\n(Baseline - no ventilation units installed)')
                    else:
                        ax.set_title(f'Year 200{year}, Week {week}')
                    
                    # Fix y-axis to show full range
                    all_data = [item for sublist in plot_data for item in sublist]
                    ax.set_ylim(bottom=min(all_data) - 1, top=max(all_data) + 1)
                    
                    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Temperature Distribution by House Groups (Active vs Control Houses)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'question_01_boxplots_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_aggregated_results(results_df: pd.DataFrame):
    """Print aggregated results with median and std in parentheses."""
    print("\n" + "="*80)
    print("AGGREGATED RESULTS SUMMARY")
    print("="*80)
    
    # Group by year and period
    for year in [2008, 2009]:
        print(f"\n{'='*60}")
        print(f"YEAR {year}")
        print(f"{'='*60}")
        
        year_data = results_df[results_df['year'] == year]
        
        if year_data.empty:
            print("No data available")
            continue
        
        # Create summary table
        summary = []
        for period in ['overall', 'daytime', 'nighttime']:
            period_data = year_data[year_data['period'] == period]
            
            for week in [1, 2]:
                week_data = period_data[period_data['week'] == week]
                if not week_data.empty:
                    row = week_data.iloc[0]
                    
                    # Format with median and std in parentheses
                    active_str = f"{row['active_mean']:.2f} (σ={row['active_std']:.2f}, med={row['active_median']:.2f})"
                    control_str = f"{row['control_mean']:.2f} (σ={row['control_std']:.2f}, med={row['control_median']:.2f})"
                    
                    summary.append({
                        'Week': week,
                        'Period': period.capitalize(),
                        'Active Houses': active_str,
                        'Control Houses': control_str,
                        'Diff': f"{row['mean_difference']:.2f}°C",
                        'P-val': f"{row['p_value']:.4f}",
                        'Sig': row['significance_level'],
                        'Test': row['test_type'][:10],  # Abbreviated test name
                        'Norm': 'Y/Y' if row['active_normal'] and row['control_normal'] else 
                               'Y/N' if row['active_normal'] else 
                               'N/Y' if row['control_normal'] else 'N/N'
                    })
        
        if summary:
            summary_df = pd.DataFrame(summary)
            print("\nTemperature (°C): mean (σ=std, med=median)")
            print(summary_df.to_string(index=False))
        
        # Statistical summary
        print(f"\nStatistical Summary for {year}:")
        sig_results = year_data[year_data['significant']]
        if not sig_results.empty:
            print(f"  - {len(sig_results)} significant differences found")
            for _, row in sig_results.iterrows():
                print(f"    * Week {row['week']} {row['period']}: "
                      f"Control houses {row['mean_difference']:.2f}°C {'warmer' if row['mean_difference'] > 0 else 'cooler'} "
                      f"(p={row['p_value']:.4f}, {row['test_type']})")
        else:
            print("  - No significant differences found")
        
        # Note about normality
        non_normal = year_data[(~year_data['active_normal']) | (~year_data['control_normal'])]
        if not non_normal.empty:
            print(f"\n  Note: {len(non_normal)} comparisons used non-parametric tests due to non-normality")


def analyze_year_differences(all_data: dict):
    """Analyze temperature differences between 2008 and 2009."""
    print("\n" + "="*80)
    print("YEAR-TO-YEAR COMPARISON (2008 vs 2009)")
    print("="*80)
    
    comparisons = []
    
    # Compare overall temperatures
    for group in ['all', 'active', 'control']:
        for period in ['overall', 'daytime', 'nighttime']:
            # Collect data
            data_2008 = []
            data_2009 = []
            
            for week in [1, 2]:
                key_2008 = f"2008_week{week}_{period}"
                key_2009 = f"2009_week{week}_{period}"
                
                if group == 'all':
                    if key_2008 in all_data:
                        if 'active' in all_data[key_2008]:
                            data_2008.extend(all_data[key_2008]['active'])
                        if 'control' in all_data[key_2008]:
                            data_2008.extend(all_data[key_2008]['control'])
                    if key_2009 in all_data:
                        if 'active' in all_data[key_2009]:
                            data_2009.extend(all_data[key_2009]['active'])
                        if 'control' in all_data[key_2009]:
                            data_2009.extend(all_data[key_2009]['control'])
                else:
                    if key_2008 in all_data and group in all_data[key_2008]:
                        data_2008.extend(all_data[key_2008][group])
                    if key_2009 in all_data and group in all_data[key_2009]:
                        data_2009.extend(all_data[key_2009][group])
            
            if data_2008 and data_2009:
                result = compare_years(data_2008, data_2009, f"{group.capitalize()} houses - {period.capitalize()}")
                comparisons.append(result)
    
    # Print results
    print("\nYear-to-Year Temperature Differences:")
    print("-" * 60)
    
    comparison_df = pd.DataFrame(comparisons)
    for _, row in comparison_df.iterrows():
        if 'error' not in row:
            print(f"\n{row['comparison']}:")
            print(f"  2008: {row['mean_2008']:.2f} ± {row['std_2008']:.2f}°C (median={row['median_2008']:.2f}, n={row['n_2008']})")
            print(f"  2009: {row['mean_2009']:.2f} ± {row['std_2009']:.2f}°C (median={row['median_2009']:.2f}, n={row['n_2009']})")
            print(f"  Difference: {row['mean_difference']:.2f}°C")
            
            # Show normality and test type
            norm_str = f"Normality: 2008={'Y' if row['normal_2008'] else 'N'}, 2009={'Y' if row['normal_2009'] else 'N'}"
            print(f"  {norm_str} → {row['test_type']} test")
            print(f"  P-value: {row['p_value']:.4f} {'(Significant)' if row['significant'] else '(Not significant)'}")
            
            if 'cohen_d' in row and not np.isnan(row.get('cohen_d', np.nan)):
                print(f"  Effect size (Cohen's d): {row['cohen_d']:.3f}")
    
    # Save comparison results
    comparison_df.to_csv(TABLES_DIR / 'question_01_year_comparison.csv', index=False)
    
    return comparison_df


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run the complete enhanced Question 1 analysis."""
    
    print("=" * 80)
    print("QUESTION 1 ENHANCED: COMPREHENSIVE OUTDOOR TEMPERATURE ANALYSIS")
    print("=" * 80)
    print()
    
    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_results = []
    house_level_data = []
    all_temperature_data = {}  # For histograms and year comparisons
    
    # Analyze each week and year
    for year in [8, 9]:
        for week in [1, 2]:
            print(f"\n{'='*60}")
            print(f"Analyzing Year 200{year}, Week {week}")
            if year == 8 and week == 1:
                print("(Baseline week - no ventilation units installed)")
            print(f"{'='*60}")
            
            # Collect data for all houses BY HOUSE GROUP (not recording status)
            active_temps = {'overall': [], 'daytime': [], 'nighttime': []}
            control_temps = {'overall': [], 'daytime': [], 'nighttime': []}
            
            # Process each house
            for house_num in range(1, 31):
                outdoor_df = extract_outdoor_temperature(week, year, house_num)
                
                if outdoor_df.empty:
                    continue
                
                # Analyze periods
                period_stats = analyze_temperature_by_period(outdoor_df)
                
                # Get house group (active or control) - NOT recording status
                house_group = get_house_group(house_num)
                
                # Store house-level data
                house_data = {
                    'house_num': house_num,
                    'week': week,
                    'year': 2000 + year,
                    'house_group': house_group,  # This is the house classification
                    'recording_status': classify_house_status(house_num, week, year)  # This is the recording status
                }
                
                # Add statistics for each period
                for period in ['overall', 'daytime', 'nighttime']:
                    if period in period_stats:
                        house_data[f'{period}_mean'] = period_stats[period]['mean']
                        house_data[f'{period}_std'] = period_stats[period]['std']
                        house_data[f'{period}_median'] = period_stats[period]['median']
                        house_data[f'{period}_is_normal'] = period_stats[period]['is_normal']
                        if period != 'overall':
                            house_data[f'{period}_hours'] = period_stats[period].get('hours', np.nan)
                
                house_level_data.append(house_data)
                
                # Collect temperatures by HOUSE GROUP (not recording status)
                for period in ['overall', 'daytime', 'nighttime']:
                    if period in period_stats and 'data' in period_stats[period]:
                        if house_group == 'active':
                            active_temps[period].extend(period_stats[period]['data'])
                        else:
                            control_temps[period].extend(period_stats[period]['data'])
            
            # Store data for histograms and year comparisons
            for period in ['overall', 'daytime', 'nighttime']:
                key = f"{2000+year}_week{week}_{period}"
                all_temperature_data[key] = {
                    'active': active_temps[period],
                    'control': control_temps[period]
                }
            
            # Compare groups for each period
            for period in ['overall', 'daytime', 'nighttime']:
                if active_temps[period] and control_temps[period]:
                    comparison = compare_groups(
                        active_temps[period], control_temps[period], 
                        period, week, year
                    )
                    all_results.append(comparison)
                    
                    # Print immediate results
                    print(f"\n{period.upper()} PERIOD:")
                    print(f"  Active houses:  {comparison['active_mean']:.2f} ± {comparison['active_std']:.2f}°C "
                          f"(median={comparison['active_median']:.2f}, n={comparison['active_n']})")
                    print(f"  Control houses: {comparison['control_mean']:.2f} ± {comparison['control_std']:.2f}°C "
                          f"(median={comparison['control_median']:.2f}, n={comparison['control_n']})")
                    print(f"  Difference: {comparison['mean_difference']:.2f}°C")
                    print(f"  Normality: Active={'Normal' if comparison['active_normal'] else 'Non-normal'}, "
                          f"Control={'Normal' if comparison['control_normal'] else 'Non-normal'}")
                    print(f"  Test: {comparison['test_type']}")
                    print(f"  P-value: {comparison['p_value']:.4f} {comparison['significance_level']}")
    
    # Create DataFrames
    results_df = pd.DataFrame(all_results)
    house_df = pd.DataFrame(house_level_data)
    
    # Save raw results
    results_df.to_csv(TABLES_DIR / 'question_01_results.csv', index=False)
    house_df.to_csv(TABLES_DIR / 'question_01_house_data.csv', index=False)
    
    # Print aggregated results
    print_aggregated_results(results_df)
    
    # Analyze year-to-year differences
    year_comparison_df = analyze_year_differences(all_temperature_data)
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Create all visualizations
    create_histograms(all_temperature_data)
    print("✓ Histograms created")
    
    create_fixed_boxplots(house_df)
    print("✓ Fixed box plots created")
    
    # Create summary statistics table
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    summary_stats = results_df.groupby(['year', 'period']).agg({
        'active_mean': 'mean',
        'control_mean': 'mean',
        'mean_difference': 'mean',
        'p_value': 'mean'
    }).round(3)
    
    print("\nAverage values across weeks:")
    print(summary_stats)
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Week 1, 2008 - Baseline comparison
    baseline = results_df[(results_df['year'] == 2008) & (results_df['week'] == 1)]
    if not baseline.empty:
        print("\n1. BASELINE COMPARISON (Week 1, 2008 - No ventilation units):")
        for _, row in baseline.iterrows():
            print(f"   {row['period'].capitalize()}: Active vs Control houses differ by {row['mean_difference']:.2f}°C "
                  f"(p={row['p_value']:.4f} {row['significance_level']})")
    
    # 2008 vs 2009 overall
    overall_2008 = results_df[(results_df['year'] == 2008) & (results_df['period'] == 'overall')]
    overall_2009 = results_df[(results_df['year'] == 2009) & (results_df['period'] == 'overall')]
    
    if not overall_2008.empty and not overall_2009.empty:
        avg_diff_2008 = overall_2008['mean_difference'].mean()
        avg_diff_2009 = overall_2009['mean_difference'].mean()
        
        print(f"\n2. SAMPLING BIAS:")
        print(f"   2008: Control houses {avg_diff_2008:.2f}°C {'warmer' if avg_diff_2008 > 0 else 'cooler'} on average")
        print(f"   2009: Control houses {avg_diff_2009:.2f}°C {'warmer' if avg_diff_2009 > 0 else 'cooler'} on average")
        
    # Significant differences
    sig_results = results_df[results_df['significant']]
    print(f"\n3. SIGNIFICANT DIFFERENCES:")
    print(f"   Found {len(sig_results)} significant differences out of {len(results_df)} comparisons")
    
    if not sig_results.empty:
        print("\n   Details:")
        for _, row in sig_results.iterrows():
            print(f"   - {row['year']} Week {row['week']} {row['period']}: "
                  f"p={row['p_value']:.4f}, diff={row['mean_difference']:.2f}°C")
    
    # Year comparison highlights
    sig_year_comp = year_comparison_df[year_comparison_df['significant'] == True]
    if not sig_year_comp.empty:
        print(f"\n4. YEAR-TO-YEAR DIFFERENCES:")
        for _, row in sig_year_comp.iterrows():
            if 'comparison' in row:
                print(f"   - {row['comparison']}: 2008 was {abs(row['mean_difference']):.2f}°C "
                      f"{'warmer' if row['mean_difference'] > 0 else 'cooler'} than 2009 (p={row['p_value']:.4f})")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()