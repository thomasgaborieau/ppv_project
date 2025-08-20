#!/usr/bin/env python
"""
Question 10: How often was the air coming from the roof cavity warmer and drier 
than the air in the occupied area (living room/bedroom)?

This analysis compares roof cavity conditions with living spaces to determine
when the ventilation system would be drawing in beneficial air (warmer and drier).
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
    from utils import (
        load_house_data,
        get_column_name,
        calculate_humidity_ratio,
        CONTROL_HOUSES,
        ACTIVE_HOUSES
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
    from src.config import *

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_humidity_ratio_vectorized(temp: pd.Series, rh: pd.Series, 
                                       pressure: float = 101325) -> pd.Series:
    """
    Calculate humidity ratio for series of temperature and RH values.
    
    Parameters:
    -----------
    temp : pd.Series
        Temperature in Celsius
    rh : pd.Series
        Relative humidity in percentage
    pressure : float
        Atmospheric pressure in Pa
    
    Returns:
    --------
    pd.Series
        Humidity ratio in g/kg
    """
    # Saturation vapor pressure (Pa)
    p_sat = 611.21 * np.exp((18.678 - temp/234.5) * (temp/(257.14 + temp)))
    
    # Partial pressure of water vapor
    p_vapor = (rh / 100) * p_sat
    
    # Humidity ratio
    w = 622 * p_vapor / (pressure - p_vapor)  # g/kg
    
    return w


def analyze_roof_cavity_conditions(week: int, year: int, house_num: int) -> dict:
    """
    Analyze when roof cavity is warmer and drier than living spaces.
    
    Parameters:
    -----------
    week : int
        Week number (1 or 2)
    year : int
        Year (8 or 9)
    house_num : int
        House number (1-30)
    
    Returns:
    --------
    dict
        Analysis results
    """
    df = load_house_data(week, year, house_num)
    
    if df.empty:
        return {}
    
    results = {
        'house_num': house_num,
        'week': week,
        'year': 2000 + year,
        'house_group': 'control' if house_num in CONTROL_HOUSES else 'active'
    }
    
    # Get column names
    roof_temp_col = get_column_name('T', 'R', week, year)
    roof_rh_col = get_column_name('RH', 'R', week, year)
    living_temp_col = get_column_name('T', 'L', week, year)
    living_rh_col = get_column_name('RH', 'L', week, year)
    bedroom_temp_col = get_column_name('T', 'B', week, year)
    bedroom_rh_col = get_column_name('RH', 'B', week, year)
    
    # Check if all required columns exist
    required_cols = [roof_temp_col, roof_rh_col]
    if not all(col in df.columns for col in required_cols):
        return results
    
    # Calculate humidity ratios for roof cavity
    roof_humidity_ratio = calculate_humidity_ratio_vectorized(
        df[roof_temp_col], df[roof_rh_col]
    )
    
    # Analyze vs Living Room
    if living_temp_col in df.columns and living_rh_col in df.columns:
        living_humidity_ratio = calculate_humidity_ratio_vectorized(
            df[living_temp_col], df[living_rh_col]
        )
        
        # Conditions where roof is better (warmer AND drier)
        roof_warmer_than_living = df[roof_temp_col] > df[living_temp_col]
        roof_drier_than_living = roof_humidity_ratio < living_humidity_ratio
        roof_better_than_living = roof_warmer_than_living & roof_drier_than_living
        
        # Calculate percentages for different time periods
        results['living_overall_pct'] = (roof_better_than_living.sum() / len(df)) * 100
        
        # Daytime analysis (9am-5pm based on solar or time)
        if 'ext__SR' in df.columns:
            daytime_mask = df['ext__SR'] >= 0.005
        else:
            times = df.index.time
            daytime_mask = (df.index.hour >= 9) & (df.index.hour < 17)
        
        if daytime_mask.sum() > 0:
            results['living_daytime_pct'] = (
                (roof_better_than_living & daytime_mask).sum() / daytime_mask.sum()
            ) * 100
            results['living_daytime_hours'] = daytime_mask.sum() * 2 / 60
        
        # Evening/night analysis
        evening_mask = ~daytime_mask
        if evening_mask.sum() > 0:
            results['living_evening_pct'] = (
                (roof_better_than_living & evening_mask).sum() / evening_mask.sum()
            ) * 100
            results['living_evening_hours'] = evening_mask.sum() * 2 / 60
        
        # Temperature and humidity differences
        results['living_temp_diff_mean'] = (df[roof_temp_col] - df[living_temp_col]).mean()
        results['living_humidity_diff_mean'] = (roof_humidity_ratio - living_humidity_ratio).mean()
        
        # Store detailed data for visualization
        results['living_warmer_pct'] = (roof_warmer_than_living.sum() / len(df)) * 100
        results['living_drier_pct'] = (roof_drier_than_living.sum() / len(df)) * 100
    
    # Analyze vs Bedroom
    if bedroom_temp_col in df.columns and bedroom_rh_col in df.columns:
        bedroom_humidity_ratio = calculate_humidity_ratio_vectorized(
            df[bedroom_temp_col], df[bedroom_rh_col]
        )
        
        # Conditions where roof is better
        roof_warmer_than_bedroom = df[roof_temp_col] > df[bedroom_temp_col]
        roof_drier_than_bedroom = roof_humidity_ratio < bedroom_humidity_ratio
        roof_better_than_bedroom = roof_warmer_than_bedroom & roof_drier_than_bedroom
        
        # Calculate percentages
        results['bedroom_overall_pct'] = (roof_better_than_bedroom.sum() / len(df)) * 100
        
        if 'ext__SR' in df.columns:
            daytime_mask = df['ext__SR'] >= 0.005
        else:
            times = df.index.time
            daytime_mask = (df.index.hour >= 9) & (df.index.hour < 17)
        
        if daytime_mask.sum() > 0:
            results['bedroom_daytime_pct'] = (
                (roof_better_than_bedroom & daytime_mask).sum() / daytime_mask.sum()
            ) * 100
        
        evening_mask = ~daytime_mask
        if evening_mask.sum() > 0:
            results['bedroom_evening_pct'] = (
                (roof_better_than_bedroom & evening_mask).sum() / evening_mask.sum()
            ) * 100
        
        # Temperature and humidity differences
        results['bedroom_temp_diff_mean'] = (df[roof_temp_col] - df[bedroom_temp_col]).mean()
        results['bedroom_humidity_diff_mean'] = (roof_humidity_ratio - bedroom_humidity_ratio).mean()
        
        # Store detailed data
        results['bedroom_warmer_pct'] = (roof_warmer_than_bedroom.sum() / len(df)) * 100
        results['bedroom_drier_pct'] = (roof_drier_than_bedroom.sum() / len(df)) * 100
    
    # Solar irradiance analysis
    if 'ext__SR' in df.columns:
        results['avg_solar'] = df['ext__SR'].mean()
        results['max_solar'] = df['ext__SR'].max()
        
        # High solar periods (>400 W/m²)
        high_solar_mask = df['ext__SR'] > 400
        if high_solar_mask.sum() > 0 and living_temp_col in df.columns:
            results['living_high_solar_pct'] = (
                (roof_better_than_living & high_solar_mask).sum() / high_solar_mask.sum()
            ) * 100
    
    return results


def create_time_series_plot(week: int, year: int, house_num: int):
    """
    Create time series plot showing when roof cavity is beneficial.
    """
    df = load_house_data(week, year, house_num)
    
    if df.empty:
        return
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Get column names
    roof_temp = get_column_name('T', 'R', week, year)
    roof_rh = get_column_name('RH', 'R', week, year)
    living_temp = get_column_name('T', 'L', week, year)
    living_rh = get_column_name('RH', 'L', week, year)
    
    if not all(col in df.columns for col in [roof_temp, roof_rh, living_temp, living_rh]):
        return
    
    # Calculate humidity ratios
    roof_hr = calculate_humidity_ratio_vectorized(df[roof_temp], df[roof_rh])
    living_hr = calculate_humidity_ratio_vectorized(df[living_temp], df[living_rh])
    
    # Plot temperatures
    axes[0].plot(df.index, df[roof_temp], label='Roof Cavity', color='red', alpha=0.7)
    axes[0].plot(df.index, df[living_temp], label='Living Room', color='blue', alpha=0.7)
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'House {house_num} - Year 200{year}, Week {week}')
    
    # Plot humidity ratios
    axes[1].plot(df.index, roof_hr, label='Roof Cavity', color='red', alpha=0.7)
    axes[1].plot(df.index, living_hr, label='Living Room', color='blue', alpha=0.7)
    axes[1].set_ylabel('Humidity Ratio (g/kg)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot temperature difference
    temp_diff = df[roof_temp] - df[living_temp]
    axes[2].plot(df.index, temp_diff, color='purple', alpha=0.7)
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].fill_between(df.index, 0, temp_diff, where=(temp_diff > 0), 
                        color='red', alpha=0.3, label='Roof warmer')
    axes[2].fill_between(df.index, 0, temp_diff, where=(temp_diff <= 0), 
                        color='blue', alpha=0.3, label='Living warmer')
    axes[2].set_ylabel('Temp Diff (°C)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot beneficial periods
    roof_warmer = df[roof_temp] > df[living_temp]
    roof_drier = roof_hr < living_hr
    beneficial = roof_warmer & roof_drier
    
    axes[3].fill_between(df.index, 0, 1, where=beneficial, 
                        color='green', alpha=0.5, label='Beneficial')
    axes[3].set_ylabel('Beneficial Period')
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Format x-axis
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.xticks(rotation=45)
    axes[3].set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'q10_timeseries_h{house_num}_w{week}_y{year}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()


def create_summary_boxplots(all_results: pd.DataFrame):
    """
    Create box plots comparing active and control homes.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for loc_idx, location in enumerate(['living', 'bedroom']):
        for period_idx, (period, period_name) in enumerate([
            ('overall', 'Overall (24h)'),
            ('daytime', 'Daytime'),
            ('evening', 'Evening/Night')
        ]):
            ax = axes[loc_idx, period_idx]
            
            col_name = f'{location}_{period}_pct'
            
            if col_name in all_results.columns:
                # Prepare data for box plot
                active_data = all_results[all_results['house_group'] == 'active'][col_name].dropna()
                control_data = all_results[all_results['house_group'] == 'control'][col_name].dropna()
                
                if len(active_data) > 0 and len(control_data) > 0:
                    bp = ax.boxplot([active_data, control_data], 
                                   labels=['Active', 'Control'],
                                   patch_artist=True, showmeans=True,
                                   meanprops=dict(marker='D', markerfacecolor='white', 
                                                markeredgecolor='black', markersize=8))
                    
                    # Color the boxes
                    bp['boxes'][0].set_facecolor(COLORS.get('active', '#4ECDC4'))
                    bp['boxes'][1].set_facecolor(COLORS.get('control', '#FF6B6B'))
                    
                    for box in bp['boxes']:
                        box.set_alpha(0.7)
                    
                    ax.set_ylabel('% Time Beneficial')
                    ax.set_title(f'{location.capitalize()} - {period_name}')
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Add mean values as text
                    ax.text(1, active_data.mean() + 1, f'μ={active_data.mean():.1f}%', 
                           ha='center', fontsize=9)
                    ax.text(2, control_data.mean() + 1, f'μ={control_data.mean():.1f}%', 
                           ha='center', fontsize=9)
    
    plt.suptitle('Percentage of Time Roof Cavity Air is Warmer AND Drier Than Living Spaces', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'q10_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_scatter_analysis(all_results: pd.DataFrame):
    """
    Create scatter plots showing relationships with solar irradiance and temperature.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Beneficial % vs Average Solar Irradiance
    if 'avg_solar' in all_results.columns and 'living_daytime_pct' in all_results.columns:
        ax = axes[0, 0]
        
        for group, color in [('active', COLORS.get('active', '#4ECDC4')), 
                             ('control', COLORS.get('control', '#FF6B6B'))]:
            group_data = all_results[all_results['house_group'] == group]
            ax.scatter(group_data['avg_solar'], group_data['living_daytime_pct'], 
                      color=color, alpha=0.6, label=group.capitalize(), s=50)
        
        # Fit trend line
        valid_data = all_results[['avg_solar', 'living_daytime_pct']].dropna()
        if len(valid_data) > 2:
            z = np.polyfit(valid_data['avg_solar'], valid_data['living_daytime_pct'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(valid_data['avg_solar'].min(), valid_data['avg_solar'].max(), 100)
            ax.plot(x_trend, p(x_trend), 'k--', alpha=0.5, label='Trend')
        
        ax.set_xlabel('Average Solar Irradiance (W/m²)')
        ax.set_ylabel('% Time Beneficial (Daytime)')
        ax.set_title('Solar Irradiance vs Beneficial Conditions')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Temperature Difference vs Beneficial %
    if 'living_temp_diff_mean' in all_results.columns:
        ax = axes[0, 1]
        
        for group, color in [('active', COLORS.get('active', '#4ECDC4')), 
                             ('control', COLORS.get('control', '#FF6B6B'))]:
            group_data = all_results[all_results['house_group'] == group]
            ax.scatter(group_data['living_temp_diff_mean'], group_data['living_overall_pct'], 
                      color=color, alpha=0.6, label=group.capitalize(), s=50)
        
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Mean Temperature Difference (Roof - Living) (°C)')
        ax.set_ylabel('% Time Beneficial (Overall)')
        ax.set_title('Temperature Difference vs Beneficial Conditions')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Warmer % vs Drier %
    if 'living_warmer_pct' in all_results.columns and 'living_drier_pct' in all_results.columns:
        ax = axes[1, 0]
        
        for group, color in [('active', COLORS.get('active', '#4ECDC4')), 
                             ('control', COLORS.get('control', '#FF6B6B'))]:
            group_data = all_results[all_results['house_group'] == group]
            ax.scatter(group_data['living_warmer_pct'], group_data['living_drier_pct'], 
                      color=color, alpha=0.6, label=group.capitalize(), s=50)
        
        # Add diagonal line
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Equal')
        ax.set_xlabel('% Time Roof Warmer')
        ax.set_ylabel('% Time Roof Drier')
        ax.set_title('Temperature vs Humidity Conditions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
    
    # Plot 4: Year and Week comparison
    ax = axes[1, 1]
    
    # Group by year and week
    grouped = all_results.groupby(['year', 'week', 'house_group'])['living_overall_pct'].mean().reset_index()
    
    x_pos = np.arange(len(grouped))
    bars = ax.bar(x_pos, grouped['living_overall_pct'], 
                  color=[COLORS.get(g, '#888888') for g in grouped['house_group']], 
                  alpha=0.7)
    
    # Customize x-axis
    labels = [f"{row['year']}\nW{row['week']}\n{row['house_group'][:3].upper()}" 
              for _, row in grouped.iterrows()]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Mean % Time Beneficial')
    ax.set_title('Temporal Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Factors Affecting Beneficial Roof Cavity Conditions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'q10_scatter_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_statistical_summary(all_results: pd.DataFrame):
    """
    Print comprehensive statistical summary.
    """
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY - QUESTION 10")
    print("="*80)
    
    # Overall statistics
    print("\n" + "-"*60)
    print("OVERALL STATISTICS (% Time Roof Cavity Better)")
    print("-"*60)
    
    for location in ['living', 'bedroom']:
        for period in ['overall', 'daytime', 'evening']:
            col_name = f'{location}_{period}_pct'
            
            if col_name in all_results.columns:
                active_data = all_results[all_results['house_group'] == 'active'][col_name].dropna()
                control_data = all_results[all_results['house_group'] == 'control'][col_name].dropna()
                
                if len(active_data) > 0 and len(control_data) > 0:
                    print(f"\n{location.capitalize()} Room - {period.capitalize()}:")
                    print(f"  Active houses:  {active_data.mean():.1f}% ± {active_data.std():.1f}% "
                          f"(median={active_data.median():.1f}%, n={len(active_data)})")
                    print(f"  Control houses: {control_data.mean():.1f}% ± {control_data.std():.1f}% "
                          f"(median={control_data.median():.1f}%, n={len(control_data)})")
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(active_data, control_data)
                    print(f"  Difference: {active_data.mean() - control_data.mean():.1f}% "
                          f"(p={p_value:.4f})")
    
    # Year-by-year comparison
    print("\n" + "-"*60)
    print("YEAR-BY-YEAR COMPARISON")
    print("-"*60)
    
    for year in [2008, 2009]:
        year_data = all_results[all_results['year'] == year]
        
        if not year_data.empty:
            print(f"\nYear {year}:")
            
            for location in ['living', 'bedroom']:
                col_name = f'{location}_daytime_pct'
                
                if col_name in year_data.columns:
                    active_mean = year_data[year_data['house_group'] == 'active'][col_name].mean()
                    control_mean = year_data[year_data['house_group'] == 'control'][col_name].mean()
                    
                    print(f"  {location.capitalize()} (Daytime): "
                          f"Active={active_mean:.1f}%, Control={control_mean:.1f}%")
    
    # Solar irradiance relationship
    if 'avg_solar' in all_results.columns and 'living_daytime_pct' in all_results.columns:
        valid_data = all_results[['avg_solar', 'living_daytime_pct']].dropna()
        
        if len(valid_data) > 3:
            corr, p_value = stats.pearsonr(valid_data['avg_solar'], valid_data['living_daytime_pct'])
            
            print("\n" + "-"*60)
            print("SOLAR IRRADIANCE CORRELATION")
            print("-"*60)
            print(f"Correlation between average solar irradiance and beneficial conditions:")
            print(f"  r = {corr:.3f}, p = {p_value:.4f}")
            
            if abs(corr) > 0.5:
                print(f"  → {'Strong' if abs(corr) > 0.7 else 'Moderate'} "
                      f"{'positive' if corr > 0 else 'negative'} correlation")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run the complete Question 10 analysis."""
    
    print("=" * 80)
    print("QUESTION 10: ROOF CAVITY AIR QUALITY ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing how often roof cavity air was warmer AND drier than living spaces")
    print("=" * 80)
    
    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_results = []
    
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
            
            # Process active houses only (ventilation relevant for them)
            for house_num in ACTIVE_HOUSES:
                house_results = analyze_roof_cavity_conditions(week, year, house_num)
                
                if house_results:
                    all_results.append(house_results)
                    
                    # Print summary for this house
                    if 'living_overall_pct' in house_results:
                        print(f"  House {house_num:2d}: Living {house_results['living_overall_pct']:.1f}%, "
                              f"Bedroom {house_results.get('bedroom_overall_pct', 0):.1f}% beneficial")
            
            # Also analyze control houses for comparison
            for house_num in CONTROL_HOUSES:
                house_results = analyze_roof_cavity_conditions(week, year, house_num)
                if house_results:
                    all_results.append(house_results)
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv(TABLES_DIR / 'question_10_results.csv', index=False)
    print(f"\n✓ Results saved to {TABLES_DIR / 'question_10_results.csv'}")
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    create_summary_boxplots(results_df)
    print("✓ Box plots created")
    
    create_scatter_analysis(results_df)
    print("✓ Scatter analysis created")
    
    # Create detailed time series for selected houses
    sample_houses = [1, 5, 10, 15, 20]  # Mix of active and control
    for house_num in sample_houses:
        if house_num in ACTIVE_HOUSES[:3]:  # Limit to avoid too many plots
            create_time_series_plot(2, 8, house_num)  # Week 2, 2008 (with ventilation)
            print(f"✓ Time series for House {house_num} created")
    
    # Print statistical summary
    print_statistical_summary(results_df)
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # 1. Overall beneficial percentage
    living_overall = results_df['living_overall_pct'].mean()
    bedroom_overall = results_df['bedroom_overall_pct'].mean()
    
    print(f"\n1. OVERALL BENEFICIAL CONDITIONS:")
    print(f"   Living Room:  {living_overall:.1f}% of the time")
    print(f"   Bedroom:      {bedroom_overall:.1f}% of the time")
    
    # 2. Daytime vs Evening
    living_day = results_df['living_daytime_pct'].mean()
    living_evening = results_df['living_evening_pct'].mean()
    
    print(f"\n2. TIME PERIOD DIFFERENCES (Living Room):")
    print(f"   Daytime:      {living_day:.1f}% beneficial")
    print(f"   Evening/Night: {living_evening:.1f}% beneficial")
    print(f"   → Roof cavity is {living_day - living_evening:.1f}% more beneficial during daytime")
    
    # 3. Active vs Control comparison
    active_living = results_df[results_df['house_group'] == 'active']['living_daytime_pct'].mean()
    control_living = results_df[results_df['house_group'] == 'control']['living_daytime_pct'].mean()
    
    print(f"\n3. HOUSE GROUP COMPARISON (Daytime Living Room):")
    print(f"   Active houses:  {active_living:.1f}% beneficial")
    print(f"   Control houses: {control_living:.1f}% beneficial")
    
    # 4. Seasonal variation
    winter_2008 = results_df[results_df['year'] == 2008]['living_daytime_pct'].mean()
    winter_2009 = results_df[results_df['year'] == 2009]['living_daytime_pct'].mean()
    
    print(f"\n4. SEASONAL VARIATION (Daytime Living Room):")
    print(f"   2008 (late winter/spring): {winter_2008:.1f}% beneficial")
    print(f"   2009 (full winter):        {winter_2009:.1f}% beneficial")
    print(f"   → Spring conditions are {winter_2008 - winter_2009:.1f}% more favorable")
    
    # 5. Practical implications
    print(f"\n5. PRACTICAL IMPLICATIONS:")
    
    # Calculate average hours per day
    if 'living_daytime_hours' in results_df.columns:
        avg_daytime_hours = results_df['living_daytime_hours'].mean()
        beneficial_hours = avg_daytime_hours * (living_day / 100)
        
        print(f"   Average daytime period: {avg_daytime_hours:.1f} hours")
        print(f"   Beneficial hours per day: {beneficial_hours:.1f} hours")
        print(f"   → Ventilation system can draw beneficial air for ~{beneficial_hours:.0f} hours daily")
    
    # 6. Limiting factors
    print(f"\n6. LIMITING FACTORS:")
    
    warmer_pct = results_df['living_warmer_pct'].mean()
    drier_pct = results_df['living_drier_pct'].mean()
    
    print(f"   Roof warmer than living: {warmer_pct:.1f}% of time")
    print(f"   Roof drier than living:  {drier_pct:.1f}% of time")
    print(f"   Both conditions met:     {living_overall:.1f}% of time")
    
    if warmer_pct < drier_pct:
        print(f"   → Temperature is the primary limiting factor")
    else:
        print(f"   → Humidity is the primary limiting factor")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*80)
    
    return results_df


if __name__ == "__main__":
    results = main()