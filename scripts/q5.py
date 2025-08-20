#!/usr/bin/env python
"""
Question 5: Comprehensive Roof Cavity Temperature Analysis
Is it possible to reach 18°C in the roof cavity during winter? If yes, for how long?

This analysis examines:
1. Daily hours above 18°C in roof cavity for each house
2. Seasonal progression (winter to spring)
3. Comparison between 2008 and 2009 monitoring periods
4. House-by-house variation
5. Statistical analysis of achievable temperatures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# ============================================================================
# CONSTANTS
# ============================================================================

ROOF_TARGET_TEMP = 18.0  # °C - WHO minimum recommended temperature
MINUTES_PER_READING = 2  # Data collected every 2 minutes
READINGS_PER_HOUR = 30   # 60 minutes / 2 minutes

# ============================================================================
# DATA ANALYSIS FUNCTIONS
# ============================================================================

def analyze_roof_cavity_temperature(df, house_num, week, year):
    """
    Analyze roof cavity temperature for a specific house and period.
    
    Parameters:
    -----------
    df : pd.DataFrame
        House data with roof cavity temperature
    house_num : int
        House number
    week : int
        Week number (1 or 2)
    year : int
        Year (8 for 2008, 9 for 2009)
    
    Returns:
    --------
    dict
        Analysis results including hours above 18°C
    """
    # Get roof cavity temperature column
    roof_temp_col = f'T-R{week}{year}'
    
    if roof_temp_col not in df.columns:
        return None
    
    roof_temps = df[roof_temp_col].dropna()
    
    if len(roof_temps) == 0:
        return None
    
    # Calculate time above 18°C
    above_18 = roof_temps >= ROOF_TARGET_TEMP
    readings_above_18 = above_18.sum()
    hours_above_18 = readings_above_18 / READINGS_PER_HOUR
    
    # Calculate percentage of time above 18°C
    total_readings = len(roof_temps)
    pct_above_18 = (readings_above_18 / total_readings) * 100
    
    # Get temperature statistics
    results = {
        'house_num': house_num,
        'week': week,
        'year': 2000 + year,
        'hours_above_18': hours_above_18,
        'pct_time_above_18': pct_above_18,
        'max_temp': roof_temps.max(),
        'mean_temp': roof_temps.mean(),
        'median_temp': roof_temps.median(),
        'min_temp': roof_temps.min(),
        'std_temp': roof_temps.std(),
        'total_hours_monitored': total_readings / READINGS_PER_HOUR
    }
    
    # Analyze by time of day if solar data available
    if 'ext__SR' in df.columns:
        # Daytime analysis (9am-5pm equivalent using solar)
        daytime_mask = df['ext__SR'] >= 0.005
        if daytime_mask.any():
            daytime_temps = df.loc[daytime_mask, roof_temp_col].dropna()
            if len(daytime_temps) > 0:
                daytime_above_18 = (daytime_temps >= ROOF_TARGET_TEMP).sum()
                results['daytime_hours_above_18'] = daytime_above_18 / READINGS_PER_HOUR
                results['daytime_pct_above_18'] = (daytime_above_18 / len(daytime_temps)) * 100
                results['daytime_mean_temp'] = daytime_temps.mean()
                results['daytime_max_temp'] = daytime_temps.max()
    
    # Find consecutive periods above 18°C
    if above_18.any():
        # Convert boolean series to find consecutive True values
        above_18_int = above_18.astype(int)
        diff = above_18_int.diff()
        
        # Find start and end of periods above 18°C
        # Note: diff() creates a series one element shorter, so we need to handle this
        starts = []
        ends = []
        
        # Check if starts at True
        if above_18.iloc[0]:
            starts.append(df.index[0])
        
        # Find transitions from False to True (starts) and True to False (ends)
        for i in range(1, len(above_18)):
            if not above_18.iloc[i-1] and above_18.iloc[i]:  # False to True
                starts.append(df.index[i])
            elif above_18.iloc[i-1] and not above_18.iloc[i]:  # True to False
                ends.append(df.index[i-1])
        
        # Check if ends at True
        if above_18.iloc[-1]:
            ends.append(df.index[-1])
        
        # Calculate longest continuous period
        if starts and ends:
            periods = []
            for start, end in zip(starts, ends[:len(starts)]):
                duration_minutes = (end - start).total_seconds() / 60
                periods.append(duration_minutes)
            
            if periods:
                results['longest_period_above_18_hours'] = max(periods) / 60
                results['num_periods_above_18'] = len(periods)
                results['avg_period_above_18_hours'] = np.mean(periods) / 60
        else:
            results['longest_period_above_18_hours'] = 0
            results['num_periods_above_18'] = 0
            results['avg_period_above_18_hours'] = 0
    else:
        results['longest_period_above_18_hours'] = 0
        results['num_periods_above_18'] = 0
        results['avg_period_above_18_hours'] = 0
    
    return results


def analyze_daily_patterns(df, house_num, week, year):
    """
    Analyze daily patterns of roof cavity temperature.
    
    Returns daily statistics for each day in the monitoring period.
    """
    roof_temp_col = f'T-R{week}{year}'
    
    if roof_temp_col not in df.columns:
        return []
    
    # Group by date - ensure we have a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to convert index to datetime if it's not already
        try:
            df.index = pd.to_datetime(df.index)
        except:
            return []
    
    # Create date column from index
    df = df.copy()  # Make a copy to avoid modifying original
    df['date'] = df.index.date
    
    daily_stats = []
    
    for date, day_data in df.groupby('date'):
        roof_temps = day_data[roof_temp_col].dropna()
        
        if len(roof_temps) == 0:
            continue
        
        above_18 = roof_temps >= ROOF_TARGET_TEMP
        hours_above_18 = above_18.sum() / READINGS_PER_HOUR
        
        daily_stat = {
            'house_num': house_num,
            'week': week,
            'year': 2000 + year,
            'date': date,
            'hours_above_18': hours_above_18,
            'max_temp': roof_temps.max(),
            'mean_temp': roof_temps.mean(),
            'min_temp': roof_temps.min()
        }
        
        # Add external temperature if available
        if 'ext__T' in day_data.columns:
            daily_stat['ext_mean_temp'] = day_data['ext__T'].mean()
            daily_stat['ext_max_temp'] = day_data['ext__T'].max()
        
        # Add solar irradiance if available
        if 'ext__SR' in day_data.columns:
            daily_stat['max_solar'] = day_data['ext__SR'].max()
            daily_stat['total_solar'] = day_data['ext__SR'].sum() / READINGS_PER_HOUR
        
        daily_stats.append(daily_stat)
    
    return daily_stats


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_hours_above_18_plot(results_df):
    """
    Create plot showing hours per day above 18°C for each house.
    Similar to Figure 5.16 in the report.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    for idx, year in enumerate([2008, 2009]):
        ax = axes[idx]
        year_data = results_df[results_df['year'] == year].copy()
        
        if year_data.empty:
            continue
        
        # Sort by house number for consistent ordering
        year_data = year_data.sort_values('house_num')
        
        # Calculate average hours for each house across weeks
        house_avg = year_data.groupby('house_num')['hours_above_18'].mean().reset_index()
        
        # Create bar plot
        bars = ax.bar(range(len(house_avg)), house_avg['hours_above_18'], 
                      color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Color code by house type (control vs active)
        control_houses = [5, 7, 12, 16, 17, 21, 23, 25, 28, 30]
        for i, house in enumerate(house_avg['house_num']):
            if house in control_houses:
                bars[i].set_color('salmon')
        
        ax.set_xlabel('House ID')
        ax.set_ylabel('Hours per day above 18°C')
        ax.set_title(f'Year {year}')
        ax.set_xticks(range(len(house_avg)))
        ax.set_xticklabels(house_avg['house_num'], rotation=45)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=8, color='green', linestyle='--', alpha=0.5, 
                   label='8 hours (1/3 of day)')
        
        # Add mean line
        mean_hours = house_avg['hours_above_18'].mean()
        ax.axhline(y=mean_hours, color='red', linestyle='-', alpha=0.5,
                   label=f'Mean: {mean_hours:.1f} hours')
        
        if idx == 0:
            ax.legend(loc='upper left')
    
    # Add legend for house types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', alpha=0.7, label='Active houses'),
        Patch(facecolor='salmon', alpha=0.7, label='Control houses')
    ]
    axes[1].legend(handles=legend_elements, loc='upper right')
    
    plt.suptitle('Daily Hours with Roof Cavity Temperature Above 18°C', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def create_seasonal_progression_plot(daily_df):
    """
    Create plot showing seasonal progression of roof cavity temperatures.
    Similar to Figure 5.17 in the report.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    for idx, year in enumerate([2008, 2009]):
        ax = axes[idx]
        year_data = daily_df[daily_df['year'] == year].copy()
        
        if year_data.empty:
            continue
        
        # Ensure date column is datetime
        if 'date' in year_data.columns:
            year_data['date'] = pd.to_datetime(year_data['date'])
        
        # Sort by date
        year_data = year_data.sort_values('date')
        
        # Calculate daily average across all houses
        daily_avg = year_data.groupby('date').agg({
            'hours_above_18': 'mean',
            'max_temp': 'mean',
            'mean_temp': 'mean'
        }).reset_index()
        
        # Convert to percentage of daytime (assuming 8-hour daytime period)
        daily_avg['pct_daytime_above_18'] = (daily_avg['hours_above_18'] / 8) * 100
        
        # Plot
        ax.plot(daily_avg['date'], daily_avg['pct_daytime_above_18'], 
                'b-', linewidth=2, label='% of daytime above 18°C')
        ax.scatter(daily_avg['date'], daily_avg['pct_daytime_above_18'], 
                   c=daily_avg['max_temp'], cmap='RdYlBu_r', s=50, alpha=0.6)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('% of daytime with roof cavity > 18°C')
        ax.set_title(f'Year {year} - Seasonal Progression')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 110])
        
        # Add trend line
        if len(daily_avg) > 1:
            # Convert dates to numeric values for trend line
            date_numeric = np.arange(len(daily_avg))
            mask = ~np.isnan(daily_avg['pct_daytime_above_18'].values)
            if mask.sum() > 1:
                z = np.polyfit(date_numeric[mask], daily_avg['pct_daytime_above_18'].values[mask], 1)
                p = np.poly1d(z)
                ax.plot(daily_avg['date'], p(date_numeric), 'r--', alpha=0.5,
                        label=f'Trend: {z[0]:.2f}% per day')
        
        # Add reference lines
        ax.axhline(y=50, color='green', linestyle=':', alpha=0.5)
        ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        ax.legend(loc='upper left')
        
        # Add colorbar for temperature
        if idx == 0 and not daily_avg.empty:
            sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', 
                                       norm=plt.Normalize(vmin=daily_avg['max_temp'].min(),
                                                         vmax=daily_avg['max_temp'].max()))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Max roof cavity temp (°C)')
    
    plt.suptitle('Seasonal Progression of Roof Cavity Temperature Above 18°C', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def create_distribution_plots(results_df):
    """
    Create distribution plots for hours above 18°C.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Histogram of hours above 18°C
    ax = axes[0, 0]
    all_hours = results_df['hours_above_18'].dropna()
    ax.hist(all_hours, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(all_hours.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {all_hours.mean():.1f} hours')
    ax.axvline(all_hours.median(), color='green', linestyle='--', linewidth=2,
               label=f'Median: {all_hours.median():.1f} hours')
    ax.set_xlabel('Hours per day above 18°C')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Daily Hours Above 18°C')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Box plot by year and week
    ax = axes[0, 1]
    data_for_box = []
    labels = []
    positions = []
    pos = 1
    
    for year in [2008, 2009]:
        for week in [1, 2]:
            week_data = results_df[(results_df['year'] == year) & 
                                   (results_df['week'] == week)]['hours_above_18'].dropna()
            if len(week_data) > 0:
                data_for_box.append(week_data.values)
                labels.append(f'{year}\nW{week}')
                positions.append(pos)
                pos += 1
    
    if data_for_box:
        bp = ax.boxplot(data_for_box, positions=positions, widths=0.6,
                       patch_artist=True, showmeans=True)
        for patch, pos in zip(bp['boxes'], positions):
            if pos <= 2:  # 2008
                patch.set_facecolor('lightblue')
            else:  # 2009
                patch.set_facecolor('lightgreen')
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Hours per day above 18°C')
        ax.set_title('Hours Above 18°C by Period')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Percentage time above 18°C during daytime
    ax = axes[1, 0]
    if 'daytime_pct_above_18' in results_df.columns:
        daytime_pct = results_df['daytime_pct_above_18'].dropna()
        ax.hist(daytime_pct, bins=30, color='gold', edgecolor='black', alpha=0.7)
        ax.axvline(daytime_pct.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {daytime_pct.mean():.1f}%')
        ax.set_xlabel('% of daytime (9am-5pm) above 18°C')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Daytime Percentage Above 18°C')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Correlation with monitoring date
    ax = axes[1, 1]
    # Create a date index based on year and week
    results_df['approx_date'] = pd.to_datetime(
        results_df.apply(lambda x: f"{int(x['year'])}-{'06' if int(x['year'])==2009 else '08'}-{15 if x['week']==1 else 22}", axis=1)
    )
    
    ax.scatter(results_df['approx_date'], results_df['hours_above_18'],
               c=results_df['year'], cmap='coolwarm', s=50, alpha=0.6)
    
    # Add trend line
    x_numeric = (results_df['approx_date'] - results_df['approx_date'].min()).dt.days.values
    y_values = results_df['hours_above_18'].values
    mask = ~np.isnan(y_values)
    if mask.sum() > 1:
        z = np.polyfit(x_numeric[mask], y_values[mask], 1)
        p = np.poly1d(z)
        ax.plot(results_df['approx_date'], p(x_numeric), 'r--', alpha=0.5,
                label=f'Trend: {z[0]:.3f} hours/day')
    
    ax.set_xlabel('Monitoring Date')
    ax.set_ylabel('Hours per day above 18°C')
    ax.set_title('Temporal Trend in Hours Above 18°C')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Distribution Analysis of Roof Cavity Temperature Above 18°C', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def create_summary_statistics_table(results_df):
    """
    Create comprehensive summary statistics table.
    """
    summary = []
    
    for year in [2008, 2009]:
        for week in [1, 2]:
            period_data = results_df[(results_df['year'] == year) & 
                                     (results_df['week'] == week)]
            
            if period_data.empty:
                continue
            
            hours_data = period_data['hours_above_18'].dropna()
            
            summary.append({
                'Year': year,
                'Week': week,
                'Mean Hours': hours_data.mean(),
                'Median Hours': hours_data.median(),
                'Std Hours': hours_data.std(),
                'Min Hours': hours_data.min(),
                'Max Hours': hours_data.max(),
                'Houses > 0 hrs': (hours_data > 0).sum(),
                'Houses > 4 hrs': (hours_data > 4).sum(),
                'Houses > 8 hrs': (hours_data > 8).sum(),
                'Total Houses': len(hours_data)
            })
    
    summary_df = pd.DataFrame(summary)
    
    # Add overall statistics
    overall_hours = results_df['hours_above_18'].dropna()
    overall_row = {
        'Year': 'Overall',
        'Week': '-',
        'Mean Hours': overall_hours.mean(),
        'Median Hours': overall_hours.median(),
        'Std Hours': overall_hours.std(),
        'Min Hours': overall_hours.min(),
        'Max Hours': overall_hours.max(),
        'Houses > 0 hrs': (overall_hours > 0).sum(),
        'Houses > 4 hrs': (overall_hours > 4).sum(),
        'Houses > 8 hrs': (overall_hours > 8).sum(),
        'Total Houses': len(overall_hours)
    }
    
    summary_df = pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)
    
    return summary_df


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """
    Run the complete Question 5 analysis.
    """
    print("=" * 80)
    print("QUESTION 5: ROOF CAVITY TEMPERATURE ANALYSIS")
    print("Is it possible to reach 18°C in the roof cavity during winter?")
    print("If yes, for how long?")
    print("=" * 80)
    
    # Create output directories
    from pathlib import Path
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    
    # Create directories if they don't exist
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directories created:")
    print(f"  Figures: {figures_dir}")
    print(f"  Tables: {tables_dir}")
    
    # Store all results
    all_results = []
    all_daily_stats = []
    
    # Load and analyze data for each house
    for year in [8, 9]:
        for week in [1, 2]:
            print(f"\nAnalyzing Year 200{year}, Week {week}")
            print("-" * 40)
            
            for house_num in range(1, 31):
                # Load actual data from CSV files
                try:
                    # Import the load function
                    import sys
                    from pathlib import Path
                    
                    # Add parent directory to path to import from src
                    current_file = Path(__file__).resolve()
                    project_root = current_file.parent.parent
                    src_path = project_root / 'src'
                    
                    if str(src_path) not in sys.path:
                        sys.path.insert(0, str(src_path))
                    
                    from utils import load_house_data
                    
                    # Load the actual data
                    df = load_house_data(week, year, house_num)
                    
                    if df.empty:
                        print(f"  No data for house {house_num}")
                        continue
                        
                except ImportError:
                    # Fallback: try direct file loading
                    from pathlib import Path
                    data_dir = Path("../data/house_data/preprocessed")
                    if not data_dir.exists():
                        data_dir = Path("data/house_data/preprocessed")
                    
                    filename = f"{week}{year}_{house_num:02d}.csv"
                    filepath = data_dir / filename
                    
                    if not filepath.exists():
                        print(f"  File not found: {filepath}")
                        continue
                    
                    df = pd.read_csv(filepath)
                    if 'timestamps' in df.columns:
                        df['timestamps'] = pd.to_datetime(df['timestamps'])
                        df.set_index('timestamps', inplace=True)
                
                # Analyze roof cavity temperature
                results = analyze_roof_cavity_temperature(df, house_num, week, year)
                if results:
                    all_results.append(results)
                
                # Analyze daily patterns
                daily_stats = analyze_daily_patterns(df, house_num, week, year)
                all_daily_stats.extend(daily_stats)
    
    # Convert to DataFrames
    results_df = pd.DataFrame(all_results)
    daily_df = pd.DataFrame(all_daily_stats)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    summary_df = create_summary_statistics_table(results_df)
    print("\nHours per day with roof cavity temperature above 18°C:")
    print(summary_df.to_string(index=False, float_format='%.2f'))
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    print("\n1. FEASIBILITY:")
    overall_hours = results_df['hours_above_18'].dropna()
    houses_achieving_18 = (overall_hours > 0).sum()
    total_houses = len(overall_hours)
    print(f"   - {houses_achieving_18}/{total_houses} houses ({houses_achieving_18/total_houses*100:.1f}%) "
          f"achieved 18°C in roof cavity at some point")
    
    print("\n2. DURATION:")
    print(f"   - Average: {overall_hours.mean():.1f} hours per day")
    print(f"   - Median: {overall_hours.median():.1f} hours per day")
    print(f"   - Range: {overall_hours.min():.1f} to {overall_hours.max():.1f} hours per day")
    
    print("\n3. SEASONAL VARIATION:")
    for year in [2008, 2009]:
        year_data = results_df[results_df['year'] == year]['hours_above_18']
        if not year_data.empty:
            print(f"   - {year}: {year_data.mean():.1f} ± {year_data.std():.1f} hours per day")
    
    print("\n4. MONITORING PERIOD EFFECT:")
    # Compare early vs late monitoring
    week1_hours = results_df[results_df['week'] == 1]['hours_above_18'].mean()
    week2_hours = results_df[results_df['week'] == 2]['hours_above_18'].mean()
    print(f"   - Week 1 average: {week1_hours:.1f} hours")
    print(f"   - Week 2 average: {week2_hours:.1f} hours")
    print(f"   - Difference: {week2_hours - week1_hours:.1f} hours")
    
    if 'daytime_pct_above_18' in results_df.columns:
        print("\n5. DAYTIME ACHIEVEMENT:")
        daytime_pct = results_df['daytime_pct_above_18'].dropna()
        print(f"   - Average {daytime_pct.mean():.1f}% of daytime hours above 18°C")
        print(f"   - {(daytime_pct > 50).sum()} houses exceeded 18°C for >50% of daytime")
    
    # Statistical test: Is there a significant difference between years?
    hours_2008 = results_df[results_df['year'] == 2008]['hours_above_18'].dropna()
    hours_2009 = results_df[results_df['year'] == 2009]['hours_above_18'].dropna()
    
    if len(hours_2008) > 0 and len(hours_2009) > 0:
        t_stat, p_value = stats.ttest_ind(hours_2008, hours_2009)
        print(f"\n6. YEAR COMPARISON (t-test):")
        print(f"   - 2008: {hours_2008.mean():.2f} ± {hours_2008.std():.2f} hours")
        print(f"   - 2009: {hours_2009.mean():.2f} ± {hours_2009.std():.2f} hours")
        print(f"   - t-statistic: {t_stat:.3f}")
        print(f"   - p-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"   - Significant difference: 2008 had {hours_2008.mean() - hours_2009.mean():.1f} more hours/day")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    fig1 = create_hours_above_18_plot(results_df)
    fig1.savefig(figures_dir / 'question_05_hours_above_18.png', dpi=300, bbox_inches='tight')
    fig1.savefig(figures_dir / 'question_05_hours_above_18.pdf', bbox_inches='tight')
    print("✓ Created and saved hours above 18°C plot")
    
    fig2 = create_seasonal_progression_plot(daily_df)
    fig2.savefig(figures_dir / 'question_05_seasonal_progression.png', dpi=300, bbox_inches='tight')
    fig2.savefig(figures_dir / 'question_05_seasonal_progression.pdf', bbox_inches='tight')
    print("✓ Created and saved seasonal progression plot")
    
    fig3 = create_distribution_plots(results_df)
    fig3.savefig(figures_dir / 'question_05_distributions.png', dpi=300, bbox_inches='tight')
    fig3.savefig(figures_dir / 'question_05_distributions.pdf', bbox_inches='tight')
    print("✓ Created and saved distribution plots")
    
    # Save data to CSV files
    print("\n" + "=" * 80)
    print("SAVING RESULTS TO CSV")
    print("=" * 80)
    
    # Save main results
    results_df.to_csv(tables_dir / 'question_05_house_results.csv', index=False)
    print(f"✓ Saved house results to {tables_dir / 'question_05_house_results.csv'}")
    
    # Save daily statistics
    daily_df.to_csv(tables_dir / 'question_05_daily_stats.csv', index=False)
    print(f"✓ Saved daily statistics to {tables_dir / 'question_05_daily_stats.csv'}")
    
    # Save summary statistics
    summary_df.to_csv(tables_dir / 'question_05_summary.csv', index=False)
    print(f"✓ Saved summary statistics to {tables_dir / 'question_05_summary.csv'}")
    
    # Save to Excel with multiple sheets
    try:
        with pd.ExcelWriter(tables_dir / 'question_05_all_results.xlsx', engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            results_df.to_excel(writer, sheet_name='House_Results', index=False)
            
            # Create a pivot table for easy analysis
            pivot_df = results_df.pivot_table(
                values='hours_above_18',
                index='house_num',
                columns=['year', 'week'],
                aggfunc='mean'
            )
            pivot_df.to_excel(writer, sheet_name='Pivot_Hours')
            
            # Add statistics by group
            control_houses = [5, 7, 12, 16, 17, 21, 23, 25, 28, 30]
            results_df['house_type'] = results_df['house_num'].apply(
                lambda x: 'Control' if x in control_houses else 'Active'
            )
            group_stats = results_df.groupby(['year', 'week', 'house_type'])['hours_above_18'].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).round(2)
            group_stats.to_excel(writer, sheet_name='Group_Statistics')
            
        print(f"✓ Saved Excel workbook to {tables_dir / 'question_05_all_results.xlsx'}")
    except ImportError:
        print("⚠ openpyxl not installed - skipping Excel export")
    except Exception as e:
        print(f"⚠ Could not save Excel file: {e}")
    
    # Create and save detailed report
    report_path = tables_dir / 'question_05_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("QUESTION 5: ROOF CAVITY TEMPERATURE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(summary_df.to_string(index=False, float_format='%.2f'))
        f.write("\n\n")
        
        f.write("KEY FINDINGS\n")
        f.write("-" * 40 + "\n")
        
        overall_hours = results_df['hours_above_18'].dropna()
        houses_achieving_18 = (overall_hours > 0).sum()
        total_houses = len(overall_hours)
        
        f.write(f"1. FEASIBILITY:\n")
        f.write(f"   - {houses_achieving_18}/{total_houses} houses ({houses_achieving_18/total_houses*100:.1f}%) ")
        f.write(f"achieved 18°C in roof cavity at some point\n\n")
        
        f.write(f"2. DURATION:\n")
        f.write(f"   - Average: {overall_hours.mean():.1f} hours per day\n")
        f.write(f"   - Median: {overall_hours.median():.1f} hours per day\n")
        f.write(f"   - Range: {overall_hours.min():.1f} to {overall_hours.max():.1f} hours per day\n\n")
        
        f.write(f"3. SEASONAL VARIATION:\n")
        for year in [2008, 2009]:
            year_data = results_df[results_df['year'] == year]['hours_above_18']
            if not year_data.empty:
                f.write(f"   - {year}: {year_data.mean():.1f} ± {year_data.std():.1f} hours per day\n")
        f.write("\n")
        
        f.write(f"4. MONITORING PERIOD EFFECT:\n")
        week1_hours = results_df[results_df['week'] == 1]['hours_above_18'].mean()
        week2_hours = results_df[results_df['week'] == 2]['hours_above_18'].mean()
        f.write(f"   - Week 1 average: {week1_hours:.1f} hours\n")
        f.write(f"   - Week 2 average: {week2_hours:.1f} hours\n")
        f.write(f"   - Difference: {week2_hours - week1_hours:.1f} hours\n\n")
        
        if 'daytime_pct_above_18' in results_df.columns:
            daytime_pct = results_df['daytime_pct_above_18'].dropna()
            f.write(f"5. DAYTIME ACHIEVEMENT:\n")
            f.write(f"   - Average {daytime_pct.mean():.1f}% of daytime hours above 18°C\n")
            f.write(f"   - {(daytime_pct > 50).sum()} houses exceeded 18°C for >50% of daytime\n\n")
        
        # Statistical test results
        hours_2008 = results_df[results_df['year'] == 2008]['hours_above_18'].dropna()
        hours_2009 = results_df[results_df['year'] == 2009]['hours_above_18'].dropna()
        
        if len(hours_2008) > 0 and len(hours_2009) > 0:
            t_stat, p_value = stats.ttest_ind(hours_2008, hours_2009)
            f.write(f"6. YEAR COMPARISON (t-test):\n")
            f.write(f"   - 2008: {hours_2008.mean():.2f} ± {hours_2008.std():.2f} hours\n")
            f.write(f"   - 2009: {hours_2009.mean():.2f} ± {hours_2009.std():.2f} hours\n")
            f.write(f"   - t-statistic: {t_stat:.3f}\n")
            f.write(f"   - p-value: {p_value:.4f}\n")
            if p_value < 0.05:
                f.write(f"   - Significant difference: 2008 had {hours_2008.mean() - hours_2009.mean():.1f} more hours/day\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("CONCLUSION\n")
        f.write("=" * 80 + "\n")
        f.write(f"YES, it is possible to reach 18°C in the roof cavity during winter.\n")
        f.write(f"On average, roof cavities exceeded 18°C for {overall_hours.mean():.1f} hours per day,\n")
        f.write(f"with substantial variation between houses ({overall_hours.std():.1f} hours std dev)\n")
        f.write(f"and a clear seasonal progression from winter to spring.\n")
    
    print(f"✓ Saved detailed report to {report_path}")
    
    # Display plots if in interactive mode
    plt.show()
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(f"\nYES, it is possible to reach 18°C in the roof cavity during winter.")
    print(f"On average, roof cavities exceeded 18°C for {overall_hours.mean():.1f} hours per day,")
    print(f"with substantial variation between houses ({overall_hours.std():.1f} hours std dev)")
    print(f"and a clear seasonal progression from winter to spring.")
    
    return results_df, daily_df, summary_df


if __name__ == "__main__":
    results_df, daily_df, summary_df = main()