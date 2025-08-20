#!/usr/bin/env python
"""
Questions 2 & 3: Indoor Temperature Analysis - FINAL VERSION WITH HISTOGRAMS

Question 2: Were the active homes warmer than control homes for the whole day?
Question 3: Were the active homes warmer than control homes for PART OF THE DAY?

Includes comprehensive histograms showing temperature distributions for:
- Week 1, 2008: Active vs Control (though all are baseline)
- All other weeks: Active vs Control comparisons
- Overall, daytime, and nighttime periods
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Setup paths
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

# Import directly from modules
import utils
import config

# Get functions we need
load_house_data = utils.load_house_data
CONTROL_HOUSES = utils.CONTROL_HOUSES
ACTIVE_HOUSES = utils.ACTIVE_HOUSES
FIGURES_DIR = config.FIGURES_DIR
TABLES_DIR = config.TABLES_DIR

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# DATA COLLECTION FUNCTIONS
# ============================================================================

def get_temp_column_name(df, location_code, week, year):
    """Find the correct temperature column name in the dataframe."""
    possible_patterns = [
        f'T-{location_code}{week}{year}',  # T-L18
        f'T-{location_code}',               # T-L
        f'T_{location_code}{week}{year}',   # T_L18
        f'T{location_code}{week}{year}',    # TL18
    ]
    
    for pattern in possible_patterns:
        if pattern in df.columns:
            return pattern
    
    # If not found, look for any column starting with T and containing location_code
    temp_cols = [col for col in df.columns if col.startswith('T') and location_code in col]
    if temp_cols:
        return temp_cols[0]
    
    return None


def collect_temperature_data(week, year, location='living'):
    """Collect temperature data for all houses, returning house-level averages."""
    location_code = 'L' if location == 'living' else 'B'
    
    results = {
        'control': {'full_day': [], 'daytime': [], 'nighttime': []},
        'active': {'full_day': [], 'daytime': [], 'nighttime': []},
        'all_baseline': {'full_day': [], 'daytime': [], 'nighttime': []},  # For week 1, 2008
        'metadata': {
            'week': week,
            'year': year,
            'location': location,
            'houses_loaded': {'control': [], 'active': []},
        }
    }
    
    # For Week 1, 2008, also separate by original house classification
    if week == 1 and year == 8:
        results['baseline_control'] = {'full_day': [], 'daytime': [], 'nighttime': []}
        results['baseline_active'] = {'full_day': [], 'daytime': [], 'nighttime': []}
    
    houses_with_data = 0
    
    for house_num in range(1, 31):
        df = load_house_data(week, year, house_num)
        
        if df.empty:
            continue
        
        # Find the temperature column
        temp_col = get_temp_column_name(df, location_code, week, year)
        
        if temp_col is None:
            continue
        
        # Get temperature data
        temps = df[temp_col].dropna()
        
        if len(temps) == 0:
            continue
        
        houses_with_data += 1
        
        # Calculate HOUSE-LEVEL AVERAGE for full day
        house_mean_full = temps.mean()
        
        # Time-based splitting
        if 'ext__SR' in df.columns:
            solar_data = df['ext__SR'].fillna(0)
            daytime_mask = solar_data >= 0.005
            nighttime_mask = ~daytime_mask
        else:
            hour = df.index.hour
            daytime_mask = (hour >= 9) & (hour < 17)
            nighttime_mask = ~daytime_mask
        
        # Calculate period averages
        house_mean_day = np.nan
        house_mean_night = np.nan
        
        if daytime_mask.sum() > 0:
            daytime_temps = df.loc[daytime_mask, temp_col].dropna()
            if len(daytime_temps) > 0:
                house_mean_day = daytime_temps.mean()
        
        if nighttime_mask.sum() > 0:
            nighttime_temps = df.loc[nighttime_mask, temp_col].dropna()
            if len(nighttime_temps) > 0:
                house_mean_night = nighttime_temps.mean()
        
        # Categorize house and add data
        if week == 1 and year == 8:
            # Week 1, 2008: ALL houses are baseline (no ventilation)
            results['all_baseline']['full_day'].append(house_mean_full)
            if not np.isnan(house_mean_day):
                results['all_baseline']['daytime'].append(house_mean_day)
            if not np.isnan(house_mean_night):
                results['all_baseline']['nighttime'].append(house_mean_night)
            
            # Also separate by original house classification for comparison
            if house_num in CONTROL_HOUSES:
                results['baseline_control']['full_day'].append(house_mean_full)
                if not np.isnan(house_mean_day):
                    results['baseline_control']['daytime'].append(house_mean_day)
                if not np.isnan(house_mean_night):
                    results['baseline_control']['nighttime'].append(house_mean_night)
            else:
                results['baseline_active']['full_day'].append(house_mean_full)
                if not np.isnan(house_mean_day):
                    results['baseline_active']['daytime'].append(house_mean_day)
                if not np.isnan(house_mean_night):
                    results['baseline_active']['nighttime'].append(house_mean_night)
        else:
            # Other weeks: separate by house group
            if house_num in CONTROL_HOUSES:
                group = 'control'
            else:
                group = 'active'
            
            results[group]['full_day'].append(house_mean_full)
            if not np.isnan(house_mean_day):
                results[group]['daytime'].append(house_mean_day)
            if not np.isnan(house_mean_night):
                results[group]['nighttime'].append(house_mean_night)
            
            results['metadata']['houses_loaded'][group].append(house_num)
    
    print(f"    Found data in {houses_with_data} houses")
    
    return results


def collect_paired_data_for_ventilation_effect(location='living'):
    """
    Collect data to compare ventilation effect: Week 1 (baseline) vs Week 2 (with ventilation)
    Only for ACTIVE houses in 2008.
    """
    location_code = 'L' if location == 'living' else 'B'
    
    paired_data = {'before': [], 'after': [], 'houses': []}
    
    for house_num in ACTIVE_HOUSES:
        # Week 1, 2008 (before ventilation)
        df1 = load_house_data(1, 8, house_num)
        if not df1.empty:
            temp_col1 = get_temp_column_name(df1, location_code, 1, 8)
        else:
            temp_col1 = None
        
        # Week 2, 2008 (after ventilation)
        df2 = load_house_data(2, 8, house_num)
        if not df2.empty:
            temp_col2 = get_temp_column_name(df2, location_code, 2, 8)
        else:
            temp_col2 = None
        
        if temp_col1 and temp_col2 and not df1.empty and not df2.empty:
            before_mean = df1[temp_col1].dropna().mean()
            after_mean = df2[temp_col2].dropna().mean()
            
            if not np.isnan(before_mean) and not np.isnan(after_mean):
                paired_data['before'].append(before_mean)
                paired_data['after'].append(after_mean)
                paired_data['houses'].append(house_num)
    
    return paired_data


# ============================================================================
# HISTOGRAM CREATION FUNCTION
# ============================================================================

def create_temperature_histograms(all_data, location):
    """Create comprehensive histograms for temperature distributions."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Define periods
    periods = ['full_day', 'daytime', 'nighttime']
    period_labels = ['Overall (24h)', 'Daytime', 'Nighttime']
    
    # Define weeks to plot
    plot_configs = [
        (8, 1, '2008 Week 1 (Baseline)'),
        (8, 2, '2008 Week 2'),
        (9, 1, '2009 Week 1'),
        (9, 2, '2009 Week 2')
    ]
    
    plot_idx = 0
    for row_idx, (year, week, title_prefix) in enumerate(plot_configs):
        for col_idx, (period, period_label) in enumerate(zip(periods, period_labels)):
            plot_idx += 1
            ax = plt.subplot(4, 3, plot_idx)
            
            # Get data
            data_key = f"{year}_{week}_{location}"
            if data_key not in all_data:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{title_prefix}\n{period_label}')
                continue
            
            data = all_data[data_key]
            
            # Prepare data for plotting
            if year == 8 and week == 1:
                # Baseline week - show original house classifications
                active_data = data.get('baseline_active', {}).get(period, [])
                control_data = data.get('baseline_control', {}).get(period, [])
                
                if active_data or control_data:
                    # Create bins
                    all_temps = active_data + control_data
                    bins = np.linspace(min(all_temps) - 0.5, max(all_temps) + 0.5, 20)
                    
                    # Plot histograms
                    if active_data:
                        ax.hist(active_data, bins=bins, alpha=0.6, 
                               label=f'Future Active (n={len(active_data)})\nμ={np.mean(active_data):.1f}°C',
                               color='#4ECDC4', edgecolor='black', linewidth=0.5, density=True)
                        ax.axvline(np.mean(active_data), color='#4ECDC4', linestyle='--', linewidth=2, alpha=0.8)
                    
                    if control_data:
                        ax.hist(control_data, bins=bins, alpha=0.6,
                               label=f'Control (n={len(control_data)})\nμ={np.mean(control_data):.1f}°C',
                               color='#FF6B6B', edgecolor='black', linewidth=0.5, density=True)
                        ax.axvline(np.mean(control_data), color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8)
                    
                    # Add combined baseline mean
                    ax.axvline(np.mean(all_temps), color='gray', linestyle=':', linewidth=2, 
                              label=f'All baseline\nμ={np.mean(all_temps):.1f}°C')
            else:
                # Other weeks - active vs control
                active_data = data.get('active', {}).get(period, [])
                control_data = data.get('control', {}).get(period, [])
                
                if active_data or control_data:
                    # Create bins
                    all_temps = active_data + control_data
                    bins = np.linspace(min(all_temps) - 0.5, max(all_temps) + 0.5, 20)
                    
                    # Plot histograms
                    if active_data:
                        ax.hist(active_data, bins=bins, alpha=0.6,
                               label=f'Active (n={len(active_data)})\nμ={np.mean(active_data):.1f}°C',
                               color='#4ECDC4', edgecolor='black', linewidth=0.5, density=True)
                        ax.axvline(np.mean(active_data), color='#4ECDC4', linestyle='--', linewidth=2, alpha=0.8)
                    
                    if control_data:
                        ax.hist(control_data, bins=bins, alpha=0.6,
                               label=f'Control (n={len(control_data)})\nμ={np.mean(control_data):.1f}°C',
                               color='#FF6B6B', edgecolor='black', linewidth=0.5, density=True)
                        ax.axvline(np.mean(control_data), color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8)
            
            # Add WHO reference lines
            ax.axvline(18, color='green', linestyle=':', alpha=0.5, linewidth=1)
            ax.axvline(24, color='green', linestyle=':', alpha=0.5, linewidth=1)
            
            # Labels and formatting
            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel('Density')
            ax.set_title(f'{title_prefix}\n{period_label}', fontsize=10)
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Set consistent x-axis limits
            ax.set_xlim([10, 28])
    
    plt.suptitle(f'{location.capitalize()} Room - Temperature Distributions', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / f'q2_3_histograms_{location}.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# OTHER VISUALIZATION FUNCTIONS (kept from previous version)
# ============================================================================

def create_comparison_plots(all_data, location):
    """Create box plots comparing active vs control houses."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    periods = ['full_day', 'daytime', 'nighttime']
    period_labels = ['Full Day (24h)', 'Daytime', 'Nighttime']
    
    for year_idx, year in enumerate([8, 9]):
        for period_idx, (period, period_label) in enumerate(zip(periods, period_labels)):
            ax = axes[year_idx, period_idx]
            
            # Collect data for both weeks
            plot_data = []
            positions = []
            colors = []
            labels = []
            pos = 1
            
            for week in [1, 2]:
                # Get data
                data = all_data.get(f"{year}_{week}_{location}", {})
                
                if year == 8 and week == 1:
                    # Baseline week - all houses together
                    baseline_data = data.get('all_baseline', {}).get(period, [])
                    if baseline_data:
                        plot_data.append(baseline_data)
                        positions.append(pos)
                        colors.append('gray')
                        labels.append('W1\nBaseline')
                        pos += 2
                else:
                    # Active vs Control comparison
                    active_data = data.get('active', {}).get(period, [])
                    control_data = data.get('control', {}).get(period, [])
                    
                    if active_data:
                        plot_data.append(active_data)
                        positions.append(pos)
                        colors.append('#4ECDC4')
                        labels.append(f'W{week}\nActive')
                        pos += 1
                    
                    if control_data:
                        plot_data.append(control_data)
                        positions.append(pos)
                        colors.append('#FF6B6B')
                        labels.append(f'W{week}\nControl')
                        pos += 1
                
                pos += 0.5  # Space between weeks
            
            if plot_data:
                bp = ax.boxplot(plot_data, positions=positions, widths=0.6,
                               patch_artist=True, showmeans=True,
                               meanprops=dict(marker='D', markerfacecolor='yellow',
                                            markeredgecolor='black', markersize=5))
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # WHO reference lines
                ax.axhline(18, color='green', linestyle=':', alpha=0.5, linewidth=1)
                ax.axhline(24, color='green', linestyle=':', alpha=0.5, linewidth=1)
                
                # Labels
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, fontsize=8)
                ax.set_ylabel('Temperature (°C)')
                ax.set_title(f'200{year} - {period_label}', fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_ylim([10, 28])
    
    plt.suptitle(f'{location.capitalize()} Room - Active vs Control Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / f'q2_3_boxplots_{location}.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_ventilation_effect_plot(paired_data, location):
    """Create plot showing ventilation installation effect (2008 W1 vs W2)."""
    if not paired_data['before']:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    before = paired_data['before']
    after = paired_data['after']
    
    # Create paired plot
    for i, (b, a) in enumerate(zip(before, after)):
        ax.plot([1, 2], [b, a], 'o-', color='gray', alpha=0.3, linewidth=0.5)
    
    # Add means with thicker line
    ax.plot([1, 2], [np.mean(before), np.mean(after)], 'o-',
           color='red', linewidth=3, markersize=10, label='Mean', zorder=5)
    
    # Statistical annotation
    p_value = stats.ttest_rel(before, after)[1]
    sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    
    ax.text(1.5, max(max(before), max(after)) + 0.5,
           f'p = {p_value:.4f} {sig_text}', ha='center', fontsize=10)
    
    ax.set_xlim([0.5, 2.5])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Week 1\n(No Ventilation)', 'Week 2\n(With Ventilation)'])
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'Ventilation Installation Effect - {location.capitalize()} Room\n'
                f'2008, Active Houses Only (n={len(before)})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add change annotation
    mean_change = np.mean(after) - np.mean(before)
    ax.text(2.2, np.mean(after), f'Δ = {mean_change:+.2f}°C', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'ventilation_effect_{location}.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS (kept from previous version)
# ============================================================================

def format_stats(mean, median, std):
    """Format statistics as 'mean (median, std)'"""
    return f"{mean:.2f} ({median:.2f}, {std:.2f})"


def perform_statistical_tests(group1_data, group2_data, test_name="Comparison"):
    """Perform statistical tests on house-level averages."""
    if len(group1_data) == 0 or len(group2_data) == 0:
        return None
    
    results = {
        'test_name': test_name,
        'n1': len(group1_data),
        'n2': len(group2_data),
        'mean1': np.mean(group1_data),
        'mean2': np.mean(group2_data),
        'median1': np.median(group1_data),
        'median2': np.median(group2_data),
        'std1': np.std(group1_data, ddof=1) if len(group1_data) > 1 else 0,
        'std2': np.std(group2_data, ddof=1) if len(group2_data) > 1 else 0,
        'mean_diff': np.mean(group1_data) - np.mean(group2_data),
        'group1_formatted': format_stats(np.mean(group1_data), np.median(group1_data), 
                                        np.std(group1_data, ddof=1) if len(group1_data) > 1 else 0),
        'group2_formatted': format_stats(np.mean(group2_data), np.median(group2_data),
                                        np.std(group2_data, ddof=1) if len(group2_data) > 1 else 0)
    }
    
    # Choose appropriate test
    if len(group1_data) >= 3 and len(group2_data) >= 3:
        # Check normality
        _, p_norm1 = stats.shapiro(group1_data)
        _, p_norm2 = stats.shapiro(group2_data)
        
        if p_norm1 > 0.05 and p_norm2 > 0.05:
            # Both normal - use t-test
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
            results['test_type'] = 'Independent t-test'
        else:
            # Non-parametric test
            u_stat, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            results['test_type'] = 'Mann-Whitney U'
    else:
        p_value = np.nan
        results['test_type'] = 'Insufficient data'
    
    results['p_value'] = p_value
    results['significant'] = p_value < 0.05 if not np.isnan(p_value) else False
    results['sig_symbol'] = (
        '***' if p_value < 0.001 else
        '**' if p_value < 0.01 else
        '*' if p_value < 0.05 else
        'ns'
    ) if not np.isnan(p_value) else 'NA'
    
    return results


def perform_paired_test(before_data, after_data, test_name="Paired Comparison"):
    """Perform paired test for ventilation effect."""
    if len(before_data) == 0:
        return None
    
    differences = np.array(after_data) - np.array(before_data)
    
    results = {
        'test_name': test_name,
        'n_pairs': len(before_data),
        'mean_before': np.mean(before_data),
        'mean_after': np.mean(after_data),
        'mean_diff': np.mean(differences),
        'before_formatted': format_stats(np.mean(before_data), np.median(before_data),
                                        np.std(before_data, ddof=1) if len(before_data) > 1 else 0),
        'after_formatted': format_stats(np.mean(after_data), np.median(after_data),
                                       np.std(after_data, ddof=1) if len(after_data) > 1 else 0)
    }
    
    if len(differences) >= 3:
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(before_data, after_data)
        results['test_type'] = 'Paired t-test'
    else:
        p_value = np.nan
        results['test_type'] = 'Insufficient data'
    
    results['p_value'] = p_value
    results['significant'] = p_value < 0.05 if not np.isnan(p_value) else False
    results['sig_symbol'] = (
        '***' if p_value < 0.001 else
        '**' if p_value < 0.01 else
        '*' if p_value < 0.05 else
        'ns'
    ) if not np.isnan(p_value) else 'NA'
    
    return results


# ============================================================================
# SUMMARY TABLE FUNCTION
# ============================================================================

def create_summary_tables(all_results, location):
    """Create clean summary tables."""
    # Separate regular comparisons from paired comparisons
    regular_results = [r for r in all_results if 'year' in r]
    paired_results = [r for r in all_results if 'paired' in r.get('test_type', '').lower()]
    
    if regular_results:
        print(f"\n{location.upper()} ROOM - Active vs Control Comparisons")
        print("="*90)
        print(f"{'Year':<6} {'Week':<6} {'Period':<12} {'Active':<20} {'Control':<20} {'Diff':<8} {'P-value':<10} {'Sig':<5}")
        print("-"*90)
        
        for r in regular_results:
            if r.get('location') == location:
                print(f"{r['year']:<6} {r['week']:<6} {r['period']:<12} "
                      f"{r['group1_formatted']:<20} {r['group2_formatted']:<20} "
                      f"{r['mean_diff']:>6.2f}°C  {r['p_value']:<10.4f} {r['sig_symbol']:<5}")
    
    if paired_results:
        print(f"\n{location.upper()} ROOM - Ventilation Effect (2008, Active Houses)")
        print("="*70)
        print(f"{'Comparison':<20} {'Before':<20} {'After':<20} {'Change':<10} {'P-value':<10} {'Sig':<5}")
        print("-"*70)
        
        for r in paired_results:
            if r.get('location') == location:
                print(f"{'Week 1 vs Week 2':<20} {r['before_formatted']:<20} "
                      f"{r['after_formatted']:<20} {r['mean_diff']:>7.2f}°C  "
                      f"{r['p_value']:<10.4f} {r['sig_symbol']:<5}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Main analysis for Questions 2 & 3."""
    
    print("="*80)
    print("QUESTIONS 2 & 3: INDOOR TEMPERATURE ANALYSIS WITH HISTOGRAMS")
    print("="*80)
    print("\nQuestion 2: Were active homes warmer than control homes for the WHOLE DAY?")
    print("Question 3: Were active homes warmer than control homes for PART OF THE DAY?")
    print("\nAnalysis includes:")
    print("- Temperature distribution histograms for all weeks and periods")
    print("- Box plots comparing Active vs Control houses")
    print("- Ventilation effect analysis (2008 Week 1 vs Week 2)")
    print("- Statistical tests using house-level averages")
    print("="*80)
    
    # Create directories
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_results = []
    all_data = {}
    
    # Analyze each location
    for location in ['living', 'bedroom']:
        print(f"\n{'='*60}")
        print(f"ANALYZING {location.upper()} ROOM")
        print(f"{'='*60}")
        
        # Collect data for all periods
        for year in [8, 9]:
            for week in [1, 2]:
                print(f"\n  Year 200{year}, Week {week}:")
                
                data = collect_temperature_data(week, year, location)
                key = f"{year}_{week}_{location}"
                all_data[key] = data
                
                if year == 8 and week == 1:
                    # Baseline week
                    baseline_data = data['all_baseline']['full_day']
                    if baseline_data:
                        print(f"    BASELINE: {len(baseline_data)} houses (no ventilation)")
                        print(f"    Mean: {np.mean(baseline_data):.2f}°C, "
                              f"Std: {np.std(baseline_data, ddof=1):.2f}°C")
                        
                        # Also show future active vs control for comparison
                        baseline_active = data.get('baseline_active', {}).get('full_day', [])
                        baseline_control = data.get('baseline_control', {}).get('full_day', [])
                        if baseline_active and baseline_control:
                            print(f"    Future active houses (n={len(baseline_active)}): {np.mean(baseline_active):.2f}°C")
                            print(f"    Control houses (n={len(baseline_control)}): {np.mean(baseline_control):.2f}°C")
                else:
                    # Compare active vs control
                    for period in ['full_day', 'daytime', 'nighttime']:
                        active_data = data['active'][period]
                        control_data = data['control'][period]
                        
                        if active_data and control_data:
                            test_result = perform_statistical_tests(
                                active_data, control_data,
                                f"{location} - 200{year} W{week} - {period}"
                            )
                            
                            if test_result:
                                test_result.update({
                                    'location': location,
                                    'year': 2000 + year,
                                    'week': week,
                                    'period': period
                                })
                                all_results.append(test_result)
                                
                                # Print immediate results
                                period_label = period.replace('_', ' ').title()
                                print(f"\n    {period_label}:")
                                print(f"      Active (n={test_result['n1']}):  {test_result['group1_formatted']}")
                                print(f"      Control (n={test_result['n2']}): {test_result['group2_formatted']}")
                                print(f"      Difference: {test_result['mean_diff']:.2f}°C")
                                print(f"      {test_result['test_type']}: p={test_result['p_value']:.4f} {test_result['sig_symbol']}")
        
        # Analyze ventilation effect (2008 only)
        print(f"\n  Ventilation Effect Analysis (2008, Active Houses):")
        paired_data = collect_paired_data_for_ventilation_effect(location)
        
        if paired_data['before']:
            test_result = perform_paired_test(
                paired_data['before'],
                paired_data['after'],
                f"{location} - Ventilation Effect"
            )
            
            if test_result:
                test_result['location'] = location
                all_results.append(test_result)
                
                print(f"    Paired comparison (n={test_result['n_pairs']}):")
                print(f"      Before: {test_result['before_formatted']}")
                print(f"      After:  {test_result['after_formatted']}")
                print(f"      Change: {test_result['mean_diff']:.2f}°C")
                print(f"      {test_result['test_type']}: p={test_result['p_value']:.4f} {test_result['sig_symbol']}")
        
        # Create visualizations
        print(f"\n  Creating visualizations for {location}...")
        create_temperature_histograms(all_data, location)  # New histogram function
        create_comparison_plots(all_data, location)  # Box plots
        create_ventilation_effect_plot(paired_data, location)  # Paired comparison
        
        # Create summary table
        create_summary_tables(all_results, location)
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(TABLES_DIR / 'q2_3_results.csv', index=False)
        print(f"\n\nResults saved to: {TABLES_DIR / 'q2_3_results.csv'}")
    
    # Print final summary
    print("\n" + "="*80)
    print("KEY FINDINGS SUMMARY")
    print("="*80)
    
    significant_results = [r for r in all_results if r.get('significant', False)]
    
    if significant_results:
        print(f"\nFound {len(significant_results)} statistically significant differences:")
        for r in significant_results:
            if 'year' in r:
                print(f"  • {r['location'].capitalize()} - {r['year']} Week {r['week']} {r['period']}: "
                      f"Difference = {r['mean_diff']:.2f}°C (p={r['p_value']:.4f})")
            elif 'paired' in r.get('test_type', '').lower():
                print(f"  • {r['location'].capitalize()} - Ventilation Effect: "
                      f"Change = {r['mean_diff']:.2f}°C (p={r['p_value']:.4f})")
    else:
        print("\nNo statistically significant differences found.")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("Visualizations created:")
    print("  - Temperature distribution histograms (all weeks, all periods)")
    print("  - Box plots for group comparisons")
    print("  - Ventilation effect plots")
    print("="*80)


if __name__ == "__main__":
    main()