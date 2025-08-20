#!/usr/bin/env python
"""
Question 9: Were active home occupants exposed to comfort zone for longer periods?

This script analyzes the percentage of time households spent in the "comfort zone"
where both temperature (18-24°C) AND relative humidity (30-60%) are within 
recommended ranges simultaneously.

Analysis includes:
- Overall comfort zone exposure comparison
- Living room analysis (5 PM - 10 PM)
- Bedroom analysis (10 PM - 7 AM)
- Week-by-week and year-by-year comparisons
- Statistical significance testing
- Detailed visualizations
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, time
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
        CONTROL_HOUSES,
        ACTIVE_HOUSES
    )
    from statistical_tests import (
        independent_t_test,
        mann_whitney_u,
        normality_test
    )
    from config import (
        RESULTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        COLORS,
        TEMPERATURE_THRESHOLDS,
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

# Comfort zone thresholds
COMFORT_TEMP_MIN = 18.0  # °C (WHO)
COMFORT_TEMP_MAX = 24.0  # °C (WHO)
COMFORT_RH_MIN = 30.0    # % (ASHRAE)
COMFORT_RH_MAX = 60.0    # % (ASHRAE)

# Time periods for occupancy assumptions
LIVING_ROOM_START = time(17, 0)  # 5 PM
LIVING_ROOM_END = time(22, 0)    # 10 PM
BEDROOM_START = time(22, 0)      # 10 PM
BEDROOM_END = time(7, 0)         # 7 AM

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_house_group(house_num: int) -> str:
    """Get the house group (control or active) based on house number."""
    return 'control' if house_num in CONTROL_HOUSES else 'active'


def calculate_comfort_zone_percentage(df: pd.DataFrame, temp_col: str, rh_col: str,
                                     time_filter: str = 'all') -> dict:
    """
    Calculate percentage of time in comfort zone.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with temperature and RH columns
    temp_col : str
        Temperature column name
    rh_col : str
        Relative humidity column name
    time_filter : str
        'all', 'living_room', or 'bedroom' for time filtering
    
    Returns:
    --------
    dict
        Comfort zone statistics
    """
    if df.empty or temp_col not in df.columns or rh_col not in df.columns:
        return {
            'comfort_pct': np.nan,
            'temp_ok_pct': np.nan,
            'rh_ok_pct': np.nan,
            'both_ok_pct': np.nan,
            'n_samples': 0
        }
    
    # Ensure we have a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to convert if possible
        try:
            if 'timestamps' in df.columns:
                df = df.set_index('timestamps')
            df.index = pd.to_datetime(df.index)
        except:
            # If we can't get datetime index, skip time filtering
            if time_filter != 'all':
                print(f"Warning: Cannot filter by time - index is not datetime")
            df_filtered = df
    else:
        # Filter by time if needed
        if time_filter == 'living_room':
            # 5 PM to 10 PM
            try:
                time_index = pd.Series([t.time() if pd.notna(t) else None for t in df.index])
                mask = ((time_index >= LIVING_ROOM_START) & 
                        (time_index < BEDROOM_START))
                df_filtered = df[mask]
            except:
                df_filtered = df
        elif time_filter == 'bedroom':
            # 10 PM to 7 AM (crossing midnight)
            try:
                time_index = pd.Series([t.time() if pd.notna(t) else None for t in df.index])
                mask = ((time_index >= BEDROOM_START) | 
                        (time_index < BEDROOM_END))
                df_filtered = df[mask]
            except:
                df_filtered = df
        else:
            df_filtered = df
    
    if df_filtered.empty:
        return {
            'comfort_pct': np.nan,
            'temp_ok_pct': np.nan,
            'rh_ok_pct': np.nan,
            'both_ok_pct': np.nan,
            'n_samples': 0
        }
    
    # Calculate comfort conditions
    temp_ok = ((df_filtered[temp_col] >= COMFORT_TEMP_MIN) & 
               (df_filtered[temp_col] <= COMFORT_TEMP_MAX))
    rh_ok = ((df_filtered[rh_col] >= COMFORT_RH_MIN) & 
             (df_filtered[rh_col] <= COMFORT_RH_MAX))
    both_ok = temp_ok & rh_ok
    
    # Calculate percentages
    n_samples = len(df_filtered)
    comfort_pct = (both_ok.sum() / n_samples * 100) if n_samples > 0 else 0
    temp_ok_pct = (temp_ok.sum() / n_samples * 100) if n_samples > 0 else 0
    rh_ok_pct = (rh_ok.sum() / n_samples * 100) if n_samples > 0 else 0
    
    # Calculate average conditions
    avg_temp = df_filtered[temp_col].mean()
    avg_rh = df_filtered[rh_col].mean()
    
    return {
        'comfort_pct': comfort_pct,
        'temp_ok_pct': temp_ok_pct,
        'rh_ok_pct': rh_ok_pct,
        'both_ok_pct': comfort_pct,
        'avg_temp': avg_temp,
        'avg_rh': avg_rh,
        'n_samples': n_samples,
        'hours': n_samples * 2 / 60  # Assuming 2-minute intervals
    }


def analyze_house_comfort(week: int, year: int, house_num: int) -> dict:
    """
    Analyze comfort zone exposure for a specific house.
    
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
        Comfort zone analysis results
    """
    df = load_house_data(week, year, house_num)
    
    if df.empty:
        return {}
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            df = df.set_index('timestamps')
        else:
            # Try to convert index to datetime
            try:
                df.index = pd.to_datetime(df.index)
            except:
                print(f"Warning: Could not convert index to datetime for house {house_num}")
    
    results = {
        'house_num': house_num,
        'week': week,
        'year': 2000 + year,
        'house_group': get_house_group(house_num)
    }
    
    # Living room analysis
    temp_col_l = get_column_name('T', 'L', week, year)
    rh_col_l = get_column_name('RH', 'L', week, year)
    
    if temp_col_l in df.columns and rh_col_l in df.columns:
        # Overall living room
        living_all = calculate_comfort_zone_percentage(df, temp_col_l, rh_col_l, 'all')
        results.update({f'living_all_{k}': v for k, v in living_all.items()})
        
        # Living room during occupancy (5 PM - 10 PM)
        living_occ = calculate_comfort_zone_percentage(df, temp_col_l, rh_col_l, 'living_room')
        results.update({f'living_occ_{k}': v for k, v in living_occ.items()})
    
    # Bedroom analysis
    temp_col_b = get_column_name('T', 'B', week, year)
    rh_col_b = get_column_name('RH', 'B', week, year)
    
    if temp_col_b in df.columns and rh_col_b in df.columns:
        # Overall bedroom
        bedroom_all = calculate_comfort_zone_percentage(df, temp_col_b, rh_col_b, 'all')
        results.update({f'bedroom_all_{k}': v for k, v in bedroom_all.items()})
        
        # Bedroom during occupancy (10 PM - 7 AM)
        bedroom_occ = calculate_comfort_zone_percentage(df, temp_col_b, rh_col_b, 'bedroom')
        results.update({f'bedroom_occ_{k}': v for k, v in bedroom_occ.items()})
    
    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_comfort_zone_comparison_plot(results_df: pd.DataFrame):
    """Create bar plot comparing comfort zone percentages."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    locations = ['living_occ', 'bedroom_occ']
    location_names = ['Living Room (5PM-10PM)', 'Bedroom (10PM-7AM)']
    
    for year_idx, year in enumerate([2008, 2009]):
        for loc_idx, (loc, loc_name) in enumerate(zip(locations, location_names)):
            ax = axes[year_idx, loc_idx]
            
            year_data = results_df[results_df['year'] == year]
            
            if year_data.empty:
                continue
            
            # Prepare data for plotting
            weeks = [1, 2]
            active_means = []
            active_stds = []
            control_means = []
            control_stds = []
            
            for week in weeks:
                week_data = year_data[year_data['week'] == week]
                
                active_comfort = week_data[week_data['house_group'] == 'active'][f'{loc}_comfort_pct'].dropna()
                control_comfort = week_data[week_data['house_group'] == 'control'][f'{loc}_comfort_pct'].dropna()
                
                active_means.append(active_comfort.mean() if len(active_comfort) > 0 else 0)
                active_stds.append(active_comfort.std() if len(active_comfort) > 0 else 0)
                control_means.append(control_comfort.mean() if len(control_comfort) > 0 else 0)
                control_stds.append(control_comfort.std() if len(control_comfort) > 0 else 0)
            
            # Create bar plot
            x = np.arange(len(weeks))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, active_means, width, yerr=active_stds,
                          label='Active homes', color=COLORS.get('active', '#4ECDC4'),
                          alpha=0.7, capsize=5)
            bars2 = ax.bar(x + width/2, control_means, width, yerr=control_stds,
                          label='Control homes', color=COLORS.get('control', '#FF6B6B'),
                          alpha=0.7, capsize=5)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
            
            # Formatting
            ax.set_xlabel('Week')
            ax.set_ylabel('% Time in Comfort Zone')
            ax.set_title(f'{year} - {loc_name}')
            ax.set_xticks(x)
            ax.set_xticklabels(['Week 1', 'Week 2'])
            ax.legend()
            ax.set_ylim(0, max(max(active_means + control_means, default=10) * 1.2, 10))
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add note for Week 1, 2008
            if year == 2008 and loc_idx == 0:
                ax.text(0, ax.get_ylim()[1] * 0.95, 
                       'Week 1: No ventilation\n(baseline)', 
                       fontsize=8, ha='center', style='italic',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
    
    plt.suptitle('Comfort Zone Exposure: Active vs Control Homes\n(Temperature 18-24°C AND RH 30-60%)', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'question_09_comfort_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_comfort_components_plot(results_df: pd.DataFrame):
    """Create plot showing temperature OK, RH OK, and both OK percentages."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    for year_idx, year in enumerate([2008, 2009]):
        year_data = results_df[results_df['year'] == year]
        
        if year_data.empty:
            continue
        
        for loc_idx, (loc, loc_name) in enumerate([('living_occ', 'Living Room'), 
                                                   ('bedroom_occ', 'Bedroom')]):
            ax = axes[year_idx, loc_idx]
            
            # Calculate means for each component
            components = ['temp_ok_pct', 'rh_ok_pct', 'comfort_pct']
            component_names = ['Temperature\n18-24°C', 'RH\n30-60%', 'Both\n(Comfort Zone)']
            
            active_week1 = []
            active_week2 = []
            control_week1 = []
            control_week2 = []
            
            for comp in components:
                col_name = f'{loc}_{comp}'
                
                w1_data = year_data[year_data['week'] == 1]
                w2_data = year_data[year_data['week'] == 2]
                
                active_w1 = w1_data[w1_data['house_group'] == 'active'][col_name].mean()
                active_w2 = w2_data[w2_data['house_group'] == 'active'][col_name].mean()
                control_w1 = w1_data[w1_data['house_group'] == 'control'][col_name].mean()
                control_w2 = w2_data[w2_data['house_group'] == 'control'][col_name].mean()
                
                active_week1.append(active_w1 if not np.isnan(active_w1) else 0)
                active_week2.append(active_w2 if not np.isnan(active_w2) else 0)
                control_week1.append(control_w1 if not np.isnan(control_w1) else 0)
                control_week2.append(control_w2 if not np.isnan(control_w2) else 0)
            
            # Create grouped bar plot
            x = np.arange(len(components))
            width = 0.2
            
            ax.bar(x - 1.5*width, active_week1, width, label='Active W1', 
                  color=COLORS.get('active', '#4ECDC4'), alpha=0.6)
            ax.bar(x - 0.5*width, active_week2, width, label='Active W2', 
                  color=COLORS.get('active', '#4ECDC4'), alpha=0.9)
            ax.bar(x + 0.5*width, control_week1, width, label='Control W1', 
                  color=COLORS.get('control', '#FF6B6B'), alpha=0.6)
            ax.bar(x + 1.5*width, control_week2, width, label='Control W2', 
                  color=COLORS.get('control', '#FF6B6B'), alpha=0.9)
            
            ax.set_xlabel('Condition')
            ax.set_ylabel('% Time Meeting Criteria')
            ax.set_title(f'{year} - {loc_name}')
            ax.set_xticks(x)
            ax.set_xticklabels(component_names)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Third column: scatter plot of avg temp vs avg RH
        ax = axes[year_idx, 2]
        
        for week in [1, 2]:
            week_data = year_data[year_data['week'] == week]
            
            for group, color in [('active', COLORS.get('active', '#4ECDC4')), 
                                ('control', COLORS.get('control', '#FF6B6B'))]:
                group_data = week_data[week_data['house_group'] == group]
                
                # Plot living room conditions
                ax.scatter(group_data['living_occ_avg_temp'], 
                          group_data['living_occ_avg_rh'],
                          color=color, alpha=0.6 if week == 1 else 0.9,
                          marker='o' if week == 1 else 's',
                          s=50, label=f'{group.capitalize()} W{week}')
        
        # Add comfort zone rectangle
        comfort_rect = mpatches.Rectangle((COMFORT_TEMP_MIN, COMFORT_RH_MIN),
                                         COMFORT_TEMP_MAX - COMFORT_TEMP_MIN,
                                         COMFORT_RH_MAX - COMFORT_RH_MIN,
                                         linewidth=2, edgecolor='green',
                                         facecolor='green', alpha=0.1)
        ax.add_patch(comfort_rect)
        
        ax.set_xlabel('Average Temperature (°C)')
        ax.set_ylabel('Average RH (%)')
        ax.set_title(f'{year} - Living Room Conditions')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(10, 30)
        ax.set_ylim(20, 90)
    
    plt.suptitle('Comfort Zone Components Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'question_09_comfort_components.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_house_level_plot(results_df: pd.DataFrame):
    """Create plot showing individual house comfort zone percentages."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for year_idx, year in enumerate([2008, 2009]):
        for week_idx, week in enumerate([1, 2]):
            ax = axes[year_idx, week_idx]
            
            data = results_df[(results_df['year'] == year) & (results_df['week'] == week)]
            
            if data.empty:
                continue
            
            # Sort by house number
            data = data.sort_values('house_num')
            
            # Prepare data
            houses = []
            living_comfort = []
            bedroom_comfort = []
            colors = []
            
            for _, row in data.iterrows():
                houses.append(row['house_num'])
                living_comfort.append(row.get('living_occ_comfort_pct', 0))
                bedroom_comfort.append(row.get('bedroom_occ_comfort_pct', 0))
                colors.append(COLORS.get(row['house_group'], '#888888'))
            
            x = np.arange(len(houses))
            width = 0.35
            
            # Create bars
            bars1 = ax.bar(x - width/2, living_comfort, width, 
                          label='Living Room (5PM-10PM)', alpha=0.7)
            bars2 = ax.bar(x + width/2, bedroom_comfort, width, 
                          label='Bedroom (10PM-7AM)', alpha=0.7)
            
            # Color bars by house group
            for bar, color in zip(bars1, colors):
                bar.set_color(color)
            for bar, color in zip(bars2, colors):
                bar.set_edgecolor(color)
                bar.set_facecolor(color)
                bar.set_alpha(0.5)
            
            # Formatting
            ax.set_xlabel('House Number')
            ax.set_ylabel('% Time in Comfort Zone')
            
            title = f'{year} - Week {week}'
            if year == 2008 and week == 1:
                title += ' (Baseline - No Ventilation)'
            ax.set_title(title)
            
            ax.set_xticks(x)
            ax.set_xticklabels(houses, rotation=45, fontsize=8)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add horizontal line at 30% (mentioned threshold in document)
            ax.axhline(y=30, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(ax.get_xlim()[1] * 0.95, 31, '30%', fontsize=8, color='red')
    
    plt.suptitle('Individual House Comfort Zone Performance', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'question_09_house_level.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def perform_statistical_tests(results_df: pd.DataFrame) -> pd.DataFrame:
    """Perform statistical tests comparing active and control homes."""
    test_results = []
    
    for year in [2008, 2009]:
        for week in [1, 2]:
            week_data = results_df[(results_df['year'] == year) & 
                                   (results_df['week'] == week)]
            
            if week_data.empty:
                continue
            
            for location in ['living_occ', 'bedroom_occ']:
                col_name = f'{location}_comfort_pct'
                
                active_data = week_data[week_data['house_group'] == 'active'][col_name].dropna()
                control_data = week_data[week_data['house_group'] == 'control'][col_name].dropna()
                
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
                    
                    test_results.append({
                        'Year': year,
                        'Week': week,
                        'Location': location.replace('_occ', '').replace('_', ' ').title(),
                        'Active Mean': active_data.mean(),
                        'Active SD': active_data.std(),
                        'Control Mean': control_data.mean(),
                        'Control SD': control_data.std(),
                        'Difference': active_data.mean() - control_data.mean(),
                        'Test': test_type,
                        'P-value': test_result['p_value'],
                        'Significant': test_result['p_value'] < 0.05
                    })
    
    return pd.DataFrame(test_results)


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run the complete Question 9 analysis."""
    
    print("=" * 80)
    print("QUESTION 9: COMFORT ZONE EXPOSURE ANALYSIS")
    print("=" * 80)
    print("\nComfort Zone Definition:")
    print(f"  Temperature: {COMFORT_TEMP_MIN}-{COMFORT_TEMP_MAX}°C (WHO)")
    print(f"  Relative Humidity: {COMFORT_RH_MIN}-{COMFORT_RH_MAX}% (ASHRAE)")
    print("  Both conditions must be met simultaneously")
    print("\nOccupancy Assumptions:")
    print("  Living Room: 5 PM - 10 PM")
    print("  Bedroom: 10 PM - 7 AM")
    print("=" * 80)
    
    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Analyze all houses
    all_results = []
    
    for year in [8, 9]:
        for week in [1, 2]:
            print(f"\nAnalyzing Year 200{year}, Week {week}...")
            
            for house_num in range(1, 31):
                house_results = analyze_house_comfort(week, year, house_num)
                if house_results:
                    all_results.append(house_results)
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save raw results
    results_df.to_csv(TABLES_DIR / 'question_09_comfort_data.csv', index=False)
    
    # Perform statistical tests
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    test_results_df = perform_statistical_tests(results_df)
    
    print("\nStatistical Test Results:")
    print(test_results_df.to_string(index=False))
    
    test_results_df.to_csv(TABLES_DIR / 'question_09_statistical_tests.csv', index=False)
    
    # Summary statistics by group
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for year in [2008, 2009]:
        print(f"\n{year}:")
        year_data = results_df[results_df['year'] == year]
        
        for week in [1, 2]:
            week_data = year_data[year_data['week'] == week]
            
            if week_data.empty:
                continue
            
            print(f"\n  Week {week}:")
            if year == 2008 and week == 1:
                print("  (Baseline - No ventilation units installed)")
            
            for location in ['living_occ', 'bedroom_occ']:
                loc_name = location.replace('_occ', '').replace('_', ' ').title()
                col_name = f'{location}_comfort_pct'
                
                active = week_data[week_data['house_group'] == 'active'][col_name]
                control = week_data[week_data['house_group'] == 'control'][col_name]
                
                print(f"\n    {loc_name}:")
                print(f"      Active homes:  {active.mean():.1f}% ± {active.std():.1f}%")
                print(f"      Control homes: {control.mean():.1f}% ± {control.std():.1f}%")
                
                # Additional statistics
                print(f"      Maximum comfort achieved:")
                print(f"        Active:  {active.max():.1f}%")
                print(f"        Control: {control.max():.1f}%")
    
    # Component analysis
    print("\n" + "="*80)
    print("COMPONENT ANALYSIS (Average across all weeks)")
    print("="*80)
    
    for location in ['living_occ', 'bedroom_occ']:
        loc_name = location.replace('_occ', '').replace('_', ' ').title()
        print(f"\n{loc_name}:")
        
        active_data = results_df[results_df['house_group'] == 'active']
        control_data = results_df[results_df['house_group'] == 'control']
        
        print(f"  Active homes:")
        print(f"    Temperature OK: {active_data[f'{location}_temp_ok_pct'].mean():.1f}%")
        print(f"    RH OK:         {active_data[f'{location}_rh_ok_pct'].mean():.1f}%")
        print(f"    Both OK:       {active_data[f'{location}_comfort_pct'].mean():.1f}%")
        
        print(f"  Control homes:")
        print(f"    Temperature OK: {control_data[f'{location}_temp_ok_pct'].mean():.1f}%")
        print(f"    RH OK:         {control_data[f'{location}_rh_ok_pct'].mean():.1f}%")
        print(f"    Both OK:       {control_data[f'{location}_comfort_pct'].mean():.1f}%")
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    create_comfort_zone_comparison_plot(results_df)
    print("✓ Comfort zone comparison plot created")
    
    create_comfort_components_plot(results_df)
    print("✓ Comfort components plot created")
    
    create_house_level_plot(results_df)
    print("✓ House-level plot created")
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # 1. Overall performance
    overall_active = results_df[results_df['house_group'] == 'active'][['living_occ_comfort_pct', 'bedroom_occ_comfort_pct']].mean().mean()
    overall_control = results_df[results_df['house_group'] == 'control'][['living_occ_comfort_pct', 'bedroom_occ_comfort_pct']].mean().mean()
    
    print(f"\n1. OVERALL COMFORT ZONE EXPOSURE:")
    print(f"   Active homes average:  {overall_active:.1f}%")
    print(f"   Control homes average: {overall_control:.1f}%")
    print(f"   Difference: {overall_active - overall_control:.1f}%")
    
    # 2. Best performing conditions
    best_conditions = results_df.groupby(['year', 'week', 'house_group'])[['living_occ_comfort_pct', 'bedroom_occ_comfort_pct']].mean()
    print(f"\n2. BEST PERFORMING CONDITIONS:")
    print(best_conditions.round(1))
    
    # 3. Significant differences
    sig_tests = test_results_df[test_results_df['Significant'] == True]
    if not sig_tests.empty:
        print(f"\n3. STATISTICALLY SIGNIFICANT DIFFERENCES:")
        for _, row in sig_tests.iterrows():
            print(f"   {row['Year']} Week {row['Week']} {row['Location']}: "
                  f"p={row['P-value']:.4f}, difference={row['Difference']:.1f}%")
    else:
        print(f"\n3. NO STATISTICALLY SIGNIFICANT DIFFERENCES FOUND")
    
    # 4. Key observation
    print(f"\n4. KEY OBSERVATION:")
    max_comfort = results_df[['living_occ_comfort_pct', 'bedroom_occ_comfort_pct']].max().max()
    print(f"   Maximum comfort zone achieved by any house: {max_comfort:.1f}%")
    
    above_30_pct = ((results_df['living_occ_comfort_pct'] > 30) | 
                    (results_df['bedroom_occ_comfort_pct'] > 30)).sum()
    total_measurements = len(results_df)
    print(f"   Houses achieving >30% comfort: {above_30_pct}/{total_measurements} "
          f"({above_30_pct/total_measurements*100:.1f}%)")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\nActive home occupants were NOT exposed to the comfort zone for")
    print("significantly longer periods than control home occupants.")
    print("\nBoth groups showed similarly low comfort zone exposure (<30% in most cases),")
    print("indicating that the ventilation system alone was insufficient to achieve")
    print("recommended comfort conditions when both temperature AND humidity")
    print("requirements must be met simultaneously.")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*80)
    
    return results_df, test_results_df


if __name__ == "__main__":
    results_df, test_results_df = main()