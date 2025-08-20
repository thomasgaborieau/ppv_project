#!/usr/bin/env python
"""
Question 6 ENHANCED: Comprehensive Analysis of Outdoor Conditions for 18°C Roof Cavity

What average outside conditions (temperature and solar irradiance) are needed 
to reach 18°C in the roof cavity during winter time?

Analysis includes:
- Threshold analysis using multiple methods
- Statistical modeling of temperature/solar relationships
- Temporal patterns and seasonal variations
- House-specific variations
- Predictive modeling
- Comprehensive visualizations
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
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.cm as cm
from matplotlib.colors import Normalize
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
        correlation_test,
        normality_test
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

# Target temperature for roof cavity
TARGET_TEMP = 18.0  # °C

# ============================================================================
# DATA COLLECTION FUNCTIONS
# ============================================================================

def collect_roof_cavity_data(week: int, year: int, house_num: int, 
                           solar_threshold: float = 0.005) -> pd.DataFrame:
    """
    Collect roof cavity temperature and outdoor conditions for a house.
    
    IMPORTANT: Only includes daytime data when solar irradiance > threshold
    to ensure we're analyzing active solar heating conditions.
    Solar threshold of 0.005 W/m² effectively means any daylight.
    """
    df = load_house_data(week, year, house_num)
    
    if df.empty:
        return pd.DataFrame()
    
    # Get roof cavity temperature column
    roof_col = get_column_name('T', 'R', week, year)
    
    if roof_col not in df.columns:
        return pd.DataFrame()
    
    # Check for outdoor data
    required_cols = ['ext__T', 'ext__SR']
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()
    
    # CRITICAL: Filter for daytime only (when solar heating is possible)
    # Use a meaningful solar threshold, not the tiny day/night threshold
    daytime_mask = df['ext__SR'] >= solar_threshold
    df_daytime = df[daytime_mask].copy()
    
    if df_daytime.empty:
        return pd.DataFrame()
    
    # Create analysis dataframe from daytime data only
    analysis_df = pd.DataFrame({
        'roof_temp': df_daytime[roof_col],
        'outdoor_temp': df_daytime['ext__T'],
        'solar_irradiance': df_daytime['ext__SR'],
        'house_num': house_num,
        'week': week,
        'year': 2000 + year,
        'timestamp': df_daytime.index
    })
    
    # Add derived features
    analysis_df['hour'] = analysis_df['timestamp'].dt.hour
    analysis_df['day_of_year'] = analysis_df['timestamp'].dt.dayofyear
    analysis_df['above_target'] = analysis_df['roof_temp'] >= TARGET_TEMP
    analysis_df['is_daytime'] = True  # Flag that this is daytime data
    
    # Remove rows with missing data
    analysis_df = analysis_df.dropna()
    
    # Add metadata about filtering
    if not analysis_df.empty:
        analysis_df.attrs['solar_threshold'] = solar_threshold
        analysis_df.attrs['daytime_hours'] = len(analysis_df) * 2 / 60  # Assuming 2-min intervals
    
    return analysis_df


def analyze_threshold_conditions(data: pd.DataFrame) -> dict:
    """
    Analyze conditions when roof cavity reaches target temperature.
    """
    # Split data by target achievement
    above_target = data[data['above_target']]
    below_target = data[~data['above_target']]
    
    if len(above_target) == 0:
        return {'error': 'No data points above target temperature'}
    
    results = {
        'n_above': len(above_target),
        'n_below': len(below_target),
        'pct_above': len(above_target) / len(data) * 100
    }
    
    # Statistics for outdoor temperature when target is reached
    results['outdoor_temp_when_above'] = {
        'mean': above_target['outdoor_temp'].mean(),
        'median': above_target['outdoor_temp'].median(),
        'std': above_target['outdoor_temp'].std(),
        'min': above_target['outdoor_temp'].min(),
        'max': above_target['outdoor_temp'].max(),
        'q25': above_target['outdoor_temp'].quantile(0.25),
        'q75': above_target['outdoor_temp'].quantile(0.75),
        'mode': above_target['outdoor_temp'].mode().iloc[0] if len(above_target['outdoor_temp'].mode()) > 0 else np.nan
    }
    
    # Statistics for solar irradiance when target is reached
    results['solar_when_above'] = {
        'mean': above_target['solar_irradiance'].mean(),
        'median': above_target['solar_irradiance'].median(),
        'std': above_target['solar_irradiance'].std(),
        'min': above_target['solar_irradiance'].min(),
        'max': above_target['solar_irradiance'].max(),
        'q25': above_target['solar_irradiance'].quantile(0.25),
        'q75': above_target['solar_irradiance'].quantile(0.75),
        'mode': above_target['solar_irradiance'].mode().iloc[0] if len(above_target['solar_irradiance'].mode()) > 0 else np.nan
    }
    
    # Find minimum thresholds (5th percentile)
    results['threshold_5pct'] = {
        'outdoor_temp': above_target['outdoor_temp'].quantile(0.05),
        'solar_irradiance': above_target['solar_irradiance'].quantile(0.05)
    }
    
    # Find reliable thresholds (25th percentile)
    results['threshold_25pct'] = {
        'outdoor_temp': above_target['outdoor_temp'].quantile(0.25),
        'solar_irradiance': above_target['solar_irradiance'].quantile(0.25)
    }
    
    # Correlation analysis
    if len(above_target) > 10:
        temp_corr = correlation_test(
            above_target['outdoor_temp'].values,
            above_target['roof_temp'].values
        )
        solar_corr = correlation_test(
            above_target['solar_irradiance'].values,
            above_target['roof_temp'].values
        )
        results['correlations'] = {
            'outdoor_temp': temp_corr['correlation'],
            'solar': solar_corr['correlation'],
            'temp_p': temp_corr['p_value'],
            'solar_p': solar_corr['p_value']
        }
    
    return results


def find_minimum_conditions(data: pd.DataFrame, percentile: float = 5) -> dict:
    """
    Find minimum conditions that achieve target temperature.
    
    IMPORTANT: Now working with daytime-only data, so results will be meaningful.
    """
    above_target = data[data['above_target']]
    
    if len(above_target) == 0:
        return {}
    
    # Method 1: Simple percentile
    simple_threshold = {
        'outdoor_temp': above_target['outdoor_temp'].quantile(percentile/100),
        'solar': above_target['solar_irradiance'].quantile(percentile/100)
    }
    
    # Method 2: Find boundary points (where transitions occur)
    transitions = []
    sorted_data = data.sort_values(['outdoor_temp', 'solar_irradiance'])
    
    for i in range(1, len(sorted_data)):
        if sorted_data.iloc[i-1]['above_target'] != sorted_data.iloc[i]['above_target']:
            transitions.append({
                'outdoor_temp': sorted_data.iloc[i]['outdoor_temp'],
                'solar': sorted_data.iloc[i]['solar_irradiance'],
                'to_above': sorted_data.iloc[i]['above_target']
            })
    
    if transitions:
        to_above = [t for t in transitions if t['to_above']]
        if to_above:
            boundary_threshold = {
                'outdoor_temp': np.percentile([t['outdoor_temp'] for t in to_above], percentile),
                'solar': np.percentile([t['solar'] for t in to_above], percentile)
            }
        else:
            boundary_threshold = simple_threshold
    else:
        boundary_threshold = simple_threshold
    
    # Method 3: Combined condition (both must be satisfied)
    # Find cases where low temp is compensated by high solar and vice versa
    low_temp_high_solar = above_target[
        (above_target['outdoor_temp'] < above_target['outdoor_temp'].quantile(0.25)) &
        (above_target['solar_irradiance'] > above_target['solar_irradiance'].quantile(0.75))
    ]
    
    high_temp_low_solar = above_target[
        (above_target['outdoor_temp'] > above_target['outdoor_temp'].quantile(0.75)) &
        (above_target['solar_irradiance'] < above_target['solar_irradiance'].quantile(0.25))
    ]
    
    compensatory_conditions = {
        'low_temp_compensated': {
            'min_temp': low_temp_high_solar['outdoor_temp'].min() if len(low_temp_high_solar) > 0 else np.nan,
            'required_solar': low_temp_high_solar['solar_irradiance'].median() if len(low_temp_high_solar) > 0 else np.nan,
            'n_cases': len(low_temp_high_solar)
        },
        'low_solar_compensated': {
            'min_solar': high_temp_low_solar['solar_irradiance'].min() if len(high_temp_low_solar) > 0 else np.nan,
            'required_temp': high_temp_low_solar['outdoor_temp'].median() if len(high_temp_low_solar) > 0 else np.nan,
            'n_cases': len(high_temp_low_solar)
        }
    }
    
    # Method 4: Find optimal combination (where both are moderate)
    optimal_threshold = {
        'outdoor_temp': above_target['outdoor_temp'].quantile(0.50),  # Median
        'solar': above_target['solar_irradiance'].quantile(0.50)  # Median
    }
    
    return {
        'simple_threshold': simple_threshold,
        'boundary_threshold': boundary_threshold,
        'compensatory': compensatory_conditions,
        'optimal_threshold': optimal_threshold,
        'n_transitions': len(transitions),
        'n_above_target': len(above_target),
        'pct_above_target': len(above_target) / len(data) * 100
    }


def build_predictive_model(data: pd.DataFrame) -> dict:
    """
    Build a logistic regression model to predict when target is reached.
    """
    if len(data) < 100:
        return {'error': 'Insufficient data for modeling'}
    
    # Prepare features
    X = data[['outdoor_temp', 'solar_irradiance']].values
    y = data['above_target'].values
    
    # Add interaction term
    X_with_interaction = np.column_stack([
        X,
        X[:, 0] * X[:, 1]  # Temperature × Solar interaction
    ])
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_with_interaction)
    
    # Fit logistic regression
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    # Get predictions and probabilities
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = (y_pred == y).mean()
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Find threshold for 95% true positive rate
    idx_95 = np.where(tpr >= 0.95)[0]
    if len(idx_95) > 0:
        threshold_95 = thresholds[idx_95[0]]
        
        # Convert back to original scale
        # Find conditions at this probability threshold
        high_prob_idx = y_prob >= threshold_95
        if high_prob_idx.sum() > 0:
            conditions_95 = {
                'outdoor_temp': data.loc[high_prob_idx, 'outdoor_temp'].min(),
                'solar': data.loc[high_prob_idx, 'solar_irradiance'].min()
            }
        else:
            conditions_95 = None
    else:
        conditions_95 = None
    
    # Feature importance (coefficients)
    feature_names = ['Outdoor Temp', 'Solar Irradiance', 'Temp×Solar']
    feature_importance = dict(zip(feature_names, model.coef_[0]))
    
    return {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'feature_importance': feature_importance,
        'conditions_95': conditions_95,
        'fpr': fpr,
        'tpr': tpr
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_threshold_visualization(all_data: pd.DataFrame, results_summary: dict):
    """
    Create comprehensive visualization of threshold conditions.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Scatter plot with target achievement
    ax1 = fig.add_subplot(gs[0, :2])
    above = all_data[all_data['above_target']]
    below = all_data[~all_data['above_target']]
    
    scatter1 = ax1.scatter(below['outdoor_temp'], below['solar_irradiance'], 
                          c='blue', alpha=0.3, s=1, label=f'Below {TARGET_TEMP}°C')
    scatter2 = ax1.scatter(above['outdoor_temp'], above['solar_irradiance'], 
                          c='red', alpha=0.3, s=1, label=f'Above {TARGET_TEMP}°C')
    
    # Add threshold lines
    if 'overall_thresholds' in results_summary:
        thresholds = results_summary['overall_thresholds']
        ax1.axvline(x=thresholds['outdoor_temp']['median'], color='darkred', 
                   linestyle='--', label=f"Median Temp: {thresholds['outdoor_temp']['median']:.1f}°C")
        ax1.axhline(y=thresholds['solar']['median'], color='darkorange', 
                   linestyle='--', label=f"Median Solar: {thresholds['solar']['median']:.0f} W/m²")
    
    ax1.set_xlabel('Outdoor Temperature (°C)')
    ax1.set_ylabel('Solar Irradiance (W/m²)')
    ax1.set_title('Conditions for Achieving 18°C in Roof Cavity')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Histograms of conditions when target reached
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(above['outdoor_temp'], bins=30, alpha=0.7, color='red', density=True)
    ax2.axvline(x=above['outdoor_temp'].median(), color='darkred', linestyle='--', linewidth=2)
    ax2.axvline(x=above['outdoor_temp'].mean(), color='orange', linestyle=':', linewidth=2)
    ax2.set_xlabel('Outdoor Temperature (°C)')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Temperature Distribution\nWhen Roof ≥ {TARGET_TEMP}°C')
    ax2.grid(True, alpha=0.3)
    
    # 3. Solar irradiance histogram
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.hist(above['solar_irradiance'], bins=30, alpha=0.7, color='orange', density=True)
    ax3.axvline(x=above['solar_irradiance'].median(), color='darkorange', linestyle='--', linewidth=2)
    ax3.axvline(x=above['solar_irradiance'].mean(), color='red', linestyle=':', linewidth=2)
    ax3.set_xlabel('Solar Irradiance (W/m²)')
    ax3.set_ylabel('Density')
    ax3.set_title(f'Solar Distribution\nWhen Roof ≥ {TARGET_TEMP}°C')
    ax3.grid(True, alpha=0.3)
    
    # 4. Heatmap of achievement probability
    ax4 = fig.add_subplot(gs[1, :2])
    
    # Create grid for heatmap
    temp_range = np.linspace(all_data['outdoor_temp'].min(), 
                            all_data['outdoor_temp'].max(), 50)
    solar_range = np.linspace(all_data['solar_irradiance'].min(), 
                              all_data['solar_irradiance'].max(), 50)
    
    temp_grid, solar_grid = np.meshgrid(temp_range, solar_range)
    
    # Calculate probability for each grid point
    prob_grid = np.zeros_like(temp_grid)
    for i in range(len(temp_range)):
        for j in range(len(solar_range)):
            # Find nearby points
            mask = ((all_data['outdoor_temp'] - temp_range[i]).abs() < 1) & \
                   ((all_data['solar_irradiance'] - solar_range[j]).abs() < 50)
            if mask.sum() > 0:
                prob_grid[j, i] = all_data.loc[mask, 'above_target'].mean()
    
    im = ax4.contourf(temp_grid, solar_grid, prob_grid, levels=20, cmap='RdYlBu_r')
    plt.colorbar(im, ax=ax4, label='Probability of Reaching 18°C')
    
    # Add contour lines
    contours = ax4.contour(temp_grid, solar_grid, prob_grid, 
                          levels=[0.25, 0.5, 0.75, 0.95], colors='black', linewidths=1)
    ax4.clabel(contours, inline=True, fontsize=8)
    
    ax4.set_xlabel('Outdoor Temperature (°C)')
    ax4.set_ylabel('Solar Irradiance (W/m²)')
    ax4.set_title('Probability of Achieving Target Temperature')
    
    # 5. Time series of achievement by day
    ax5 = fig.add_subplot(gs[2, :])
    
    # Group by day and calculate achievement rate
    all_data['date'] = pd.to_datetime(all_data['timestamp']).dt.date
    daily_achievement = all_data.groupby('date').agg({
        'above_target': 'mean',
        'outdoor_temp': 'mean',
        'solar_irradiance': 'mean'
    }).reset_index()
    
    ax5_twin = ax5.twinx()
    
    line1 = ax5.plot(daily_achievement['date'], daily_achievement['above_target'] * 100, 
                    'b-', label='% Time Above 18°C', linewidth=2)
    line2 = ax5_twin.plot(daily_achievement['date'], daily_achievement['outdoor_temp'], 
                         'r--', label='Avg Outdoor Temp', alpha=0.7)
    line3 = ax5_twin.plot(daily_achievement['date'], daily_achievement['solar_irradiance']/10, 
                         'orange', linestyle=':', label='Avg Solar/10', alpha=0.7)
    
    ax5.set_xlabel('Date')
    ax5.set_ylabel('% Time Above 18°C', color='b')
    ax5_twin.set_ylabel('Temperature (°C) / Solar÷10 (W/m²)', color='r')
    ax5.set_title('Daily Achievement of Target Temperature')
    ax5.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='upper left')
    
    plt.suptitle('Comprehensive Analysis of Conditions for 18°C Roof Cavity Temperature', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'question_06_threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_house_comparison_plot(house_results: pd.DataFrame):
    """
    Create comparison plot showing house-specific variations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sort houses by achievement rate
    house_results = house_results.sort_values('pct_above', ascending=False)
    
    # 1. Achievement rate by house
    ax1 = axes[0, 0]
    colors = ['red' if h in CONTROL_HOUSES else 'blue' for h in house_results['house_num']]
    bars = ax1.bar(range(len(house_results)), house_results['pct_above'], color=colors, alpha=0.7)
    ax1.set_xlabel('House (sorted by achievement)')
    ax1.set_ylabel('% Time Above 18°C')
    ax1.set_title('Roof Cavity Temperature Achievement by House')
    ax1.set_xticks(range(0, len(house_results), 2))
    ax1.set_xticklabels(house_results['house_num'].iloc[::2], rotation=45)
    ax1.axhline(y=house_results['pct_above'].mean(), color='green', linestyle='--', 
               label=f'Mean: {house_results["pct_above"].mean():.1f}%')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add legend for house types
    red_patch = mpatches.Patch(color='red', alpha=0.7, label='Control Houses')
    blue_patch = mpatches.Patch(color='blue', alpha=0.7, label='Active Houses')
    ax1.legend(handles=[red_patch, blue_patch], loc='upper right')
    
    # 2. Required outdoor temperature by house
    ax2 = axes[0, 1]
    ax2.scatter(house_results['house_num'], house_results['temp_threshold_median'], 
               c=colors, alpha=0.7, s=50)
    ax2.errorbar(house_results['house_num'], house_results['temp_threshold_median'],
                yerr=[house_results['temp_threshold_median'] - house_results['temp_threshold_q25'],
                      house_results['temp_threshold_q75'] - house_results['temp_threshold_median']],
                fmt='none', ecolor='gray', alpha=0.3)
    ax2.set_xlabel('House Number')
    ax2.set_ylabel('Outdoor Temperature (°C)')
    ax2.set_title('Required Outdoor Temperature (Median with IQR)')
    ax2.axhline(y=house_results['temp_threshold_median'].mean(), color='green', 
               linestyle='--', alpha=0.7)
    ax2.grid(True, alpha=0.3)
    
    # 3. Required solar irradiance by house
    ax3 = axes[1, 0]
    ax3.scatter(house_results['house_num'], house_results['solar_threshold_median'], 
               c=colors, alpha=0.7, s=50)
    ax3.errorbar(house_results['house_num'], house_results['solar_threshold_median'],
                yerr=[house_results['solar_threshold_median'] - house_results['solar_threshold_q25'],
                      house_results['solar_threshold_q75'] - house_results['solar_threshold_median']],
                fmt='none', ecolor='gray', alpha=0.3)
    ax3.set_xlabel('House Number')
    ax3.set_ylabel('Solar Irradiance (W/m²)')
    ax3.set_title('Required Solar Irradiance (Median with IQR)')
    ax3.axhline(y=house_results['solar_threshold_median'].mean(), color='green', 
               linestyle='--', alpha=0.7)
    ax3.grid(True, alpha=0.3)
    
    # 4. Relationship between thresholds
    ax4 = axes[1, 1]
    scatter = ax4.scatter(house_results['temp_threshold_median'], 
                         house_results['solar_threshold_median'],
                         c=house_results['pct_above'], cmap='RdYlGn', 
                         s=100, alpha=0.7)
    plt.colorbar(scatter, ax=ax4, label='% Achievement')
    
    # Add trend line
    z = np.polyfit(house_results['temp_threshold_median'].dropna(), 
                   house_results['solar_threshold_median'].dropna(), 1)
    p = np.poly1d(z)
    x_trend = np.linspace(house_results['temp_threshold_median'].min(), 
                         house_results['temp_threshold_median'].max(), 100)
    ax4.plot(x_trend, p(x_trend), 'b--', alpha=0.5, label='Trend')
    
    ax4.set_xlabel('Required Outdoor Temperature (°C)')
    ax4.set_ylabel('Required Solar Irradiance (W/m²)')
    ax4.set_title('Trade-off Between Temperature and Solar Requirements')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('House-Specific Variations in Achieving 18°C Roof Cavity Temperature', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    
    plt.savefig(FIGURES_DIR / 'question_06_house_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_model_performance_plot(model_results: dict):
    """
    Create visualization of predictive model performance.
    """
    if 'error' in model_results:
        print("Insufficient data for model visualization")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. ROC Curve
    ax1 = axes[0]
    ax1.plot(model_results['fpr'], model_results['tpr'], 
            color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {model_results["roc_auc"]:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve for Target Achievement Prediction')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # 2. Feature Importance
    ax2 = axes[1]
    features = list(model_results['feature_importance'].keys())
    importance = list(model_results['feature_importance'].values())
    colors = ['green' if i > 0 else 'red' for i in importance]
    
    bars = ax2.barh(features, importance, color=colors, alpha=0.7)
    ax2.set_xlabel('Coefficient Value')
    ax2.set_title('Feature Importance in Logistic Regression')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Model accuracy summary
    ax3 = axes[2]
    ax3.axis('off')
    
    summary_text = f"""Model Performance Summary
    
Accuracy: {model_results['accuracy']:.1%}
ROC AUC: {model_results['roc_auc']:.3f}

95% Confidence Conditions:"""
    
    if model_results['conditions_95']:
        summary_text += f"""
Outdoor Temp: {model_results['conditions_95']['outdoor_temp']:.1f}°C
Solar Irradiance: {model_results['conditions_95']['solar']:.0f} W/m²"""
    else:
        summary_text += "\nNot available"
    
    summary_text += f"""

Feature Coefficients:
"""
    for feat, coef in model_results['feature_importance'].items():
        summary_text += f"\n{feat}: {coef:.3f}"
    
    ax3.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            fontfamily='monospace')
    
    plt.suptitle('Predictive Model Performance for 18°C Achievement', fontsize=14, y=1.02)
    plt.tight_layout()
    
    plt.savefig(FIGURES_DIR / 'question_06_model_performance.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# SUMMARY AND REPORTING FUNCTIONS
# ============================================================================

def print_comprehensive_summary(results_summary: dict, house_results: pd.DataFrame, 
                               model_results: dict):
    """
    Print comprehensive summary of findings.
    """
    print("\n" + "="*80)
    print("QUESTION 6: OUTDOOR CONDITIONS FOR 18°C IN ROOF CAVITY")
    print("="*80)
    
    # Overall statistics
    if 'overall_thresholds' in results_summary:
        thresholds = results_summary['overall_thresholds']
        
        print("\n" + "-"*60)
        print("THRESHOLD CONDITIONS (When Roof Cavity ≥ 18°C)")
        print("-"*60)
        
        print("\n1. OUTDOOR TEMPERATURE REQUIREMENTS:")
        print(f"   Mean:   {thresholds['outdoor_temp']['mean']:.1f}°C")
        print(f"   Median: {thresholds['outdoor_temp']['median']:.1f}°C")
        print(f"   Mode:   {thresholds['outdoor_temp']['mode']:.1f}°C")
        print(f"   Std Dev: {thresholds['outdoor_temp']['std']:.1f}°C")
        print(f"   Range:  {thresholds['outdoor_temp']['min']:.1f}°C to {thresholds['outdoor_temp']['max']:.1f}°C")
        print(f"   IQR:    {thresholds['outdoor_temp']['q25']:.1f}°C to {thresholds['outdoor_temp']['q75']:.1f}°C")
        
        print("\n2. SOLAR IRRADIANCE REQUIREMENTS:")
        print(f"   Mean:   {thresholds['solar']['mean']:.0f} W/m²")
        print(f"   Median: {thresholds['solar']['median']:.0f} W/m²")
        print(f"   Mode:   {thresholds['solar']['mode']:.0f} W/m²")
        print(f"   Std Dev: {thresholds['solar']['std']:.0f} W/m²")
        print(f"   Range:  {thresholds['solar']['min']:.0f} to {thresholds['solar']['max']:.0f} W/m²")
        print(f"   IQR:    {thresholds['solar']['q25']:.0f} to {thresholds['solar']['q75']:.0f} W/m²")
    
    # Minimum conditions
    if 'minimum_conditions' in results_summary:
        min_cond = results_summary['minimum_conditions']
        
        print("\n" + "-"*60)
        print("MINIMUM CONDITIONS ANALYSIS")
        print("-"*60)
        
        print("\n3. MINIMUM THRESHOLDS (5th Percentile):")
        if 'simple_threshold' in min_cond:
            print(f"   Outdoor Temperature: {min_cond['simple_threshold']['outdoor_temp']:.1f}°C")
            print(f"   Solar Irradiance:    {min_cond['simple_threshold']['solar']:.0f} W/m²")
        
        print("\n4. COMPENSATORY CONDITIONS:")
        if 'compensatory' in min_cond:
            comp = min_cond['compensatory']
            if not np.isnan(comp['low_temp_compensated']['min_temp']):
                print(f"   Low temperature ({comp['low_temp_compensated']['min_temp']:.1f}°C) "
                      f"compensated by high solar ({comp['low_temp_compensated']['required_solar']:.0f} W/m²)")
            if not np.isnan(comp['low_solar_compensated']['min_solar']):
                print(f"   Low solar ({comp['low_solar_compensated']['min_solar']:.0f} W/m²) "
                      f"compensated by high temperature ({comp['low_solar_compensated']['required_temp']:.1f}°C)")
    
    # Achievement statistics
    print("\n" + "-"*60)
    print("ACHIEVEMENT STATISTICS")
    print("-"*60)
    
    print(f"\n5. OVERALL ACHIEVEMENT:")
    print(f"   Total data points: {results_summary.get('total_points', 0):,}")
    print(f"   Points above 18°C: {results_summary.get('points_above', 0):,}")
    print(f"   Overall achievement: {results_summary.get('overall_pct_above', 0):.1f}%")
    
    # House variations
    if not house_results.empty:
        print(f"\n6. HOUSE VARIATIONS:")
        print(f"   Houses analyzed: {len(house_results)}")
        print(f"   Achievement range: {house_results['pct_above'].min():.1f}% to {house_results['pct_above'].max():.1f}%")
        print(f"   Mean achievement: {house_results['pct_above'].mean():.1f}%")
        print(f"   Std deviation: {house_results['pct_above'].std():.1f}%")
        
        # Compare control vs active
        control_achievement = house_results[house_results['house_num'].isin(CONTROL_HOUSES)]['pct_above'].mean()
        active_achievement = house_results[~house_results['house_num'].isin(CONTROL_HOUSES)]['pct_above'].mean()
        
        if not np.isnan(control_achievement) and not np.isnan(active_achievement):
            print(f"\n   Control houses: {control_achievement:.1f}%")
            print(f"   Active houses:  {active_achievement:.1f}%")
            print(f"   Difference:     {active_achievement - control_achievement:.1f}%")
    
    # Temporal patterns
    if 'temporal_patterns' in results_summary:
        patterns = results_summary['temporal_patterns']
        
        print("\n" + "-"*60)
        print("TEMPORAL PATTERNS")
        print("-"*60)
        
        print("\n7. SEASONAL VARIATION:")
        for year in [2008, 2009]:
            if f'year_{year}' in patterns:
                year_data = patterns[f'year_{year}']
                print(f"\n   Year {year}:")
                print(f"     Achievement: {year_data['pct_above']:.1f}%")
                print(f"     Avg outdoor temp when achieved: {year_data['avg_temp']:.1f}°C")
                print(f"     Avg solar when achieved: {year_data['avg_solar']:.0f} W/m²")
    
    # Model results
    if model_results and 'error' not in model_results:
        print("\n" + "-"*60)
        print("PREDICTIVE MODEL RESULTS")
        print("-"*60)
        
        print(f"\n8. MODEL PERFORMANCE:")
        print(f"   Accuracy: {model_results['accuracy']:.1%}")
        print(f"   ROC AUC: {model_results['roc_auc']:.3f}")
        
        if model_results['conditions_95']:
            print(f"\n   95% Confidence Conditions:")
            print(f"     Outdoor Temperature: {model_results['conditions_95']['outdoor_temp']:.1f}°C")
            print(f"     Solar Irradiance: {model_results['conditions_95']['solar']:.0f} W/m²")
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    if 'overall_thresholds' in results_summary:
        thresholds = results_summary['overall_thresholds']
        
        print(f"\n1. TYPICAL CONDITIONS:")
        print(f"   To achieve 18°C in the roof cavity, typical conditions are:")
        print(f"   • Outdoor temperature: {thresholds['outdoor_temp']['median']:.1f}°C")
        print(f"   • Solar irradiance: {thresholds['solar']['median']:.0f} W/m²")
        
        print(f"\n2. MINIMUM VIABLE CONDITIONS:")
        if 'minimum_conditions' in results_summary:
            min_cond = results_summary['minimum_conditions']
            if 'simple_threshold' in min_cond:
                print(f"   Absolute minimum (5th percentile):")
                print(f"   • Outdoor temperature: {min_cond['simple_threshold']['outdoor_temp']:.1f}°C")
                print(f"   • Solar irradiance: {min_cond['simple_threshold']['solar']:.0f} W/m²")
        
        print(f"\n3. PRACTICAL IMPLICATIONS:")
        print(f"   • Achievement varies significantly between houses ({house_results['pct_above'].min():.0f}-{house_results['pct_above'].max():.0f}%)")
        print(f"   • Both temperature AND solar irradiance are important")
        print(f"   • Low temperature can be compensated by high solar and vice versa")
        
        # Time availability
        avg_achievement = results_summary.get('overall_pct_above', 0)
        if avg_achievement > 0:
            hours_per_day = avg_achievement / 100 * 24
            print(f"\n4. TIME AVAILABILITY:")
            print(f"   • Target temperature achieved {avg_achievement:.1f}% of the time")
            print(f"   • Approximately {hours_per_day:.1f} hours per day on average")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """
    Run the complete Question 6 enhanced analysis.
    """
    print("\n" + "="*80)
    print("QUESTION 6 ENHANCED: OUTDOOR CONDITIONS FOR 18°C ROOF CAVITY")
    print("=" * 80)
    print("\nAnalyzing what outdoor conditions are needed to reach 18°C in the roof cavity")
    print("IMPORTANT: Analysis focuses on DAYTIME CONDITIONS ONLY (solar > 0.005 W/m²)")
    print("This excludes nighttime to focus on solar-driven heating, not residual heat.")
    print("=" * 80)
    
    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect all data - DAYTIME ONLY
    all_data_list = []
    house_results_list = []
    
    # Define solar threshold for "daytime" - any measurable solar radiation
    SOLAR_THRESHOLD = 0.005  # W/m² - essentially any daylight
    
    print(f"\nCollecting DAYTIME data from all houses (solar > {SOLAR_THRESHOLD} W/m²)...")
    print("This excludes nighttime periods to focus on solar-driven heating conditions.")
    
    for year in [8, 9]:
        for week in [1, 2]:
            print(f"\n  Processing Year 200{year}, Week {week}...")
            
            for house_num in range(1, 31):
                # Collect data - FILTERED FOR DAYTIME ONLY
                house_data = collect_roof_cavity_data(week, year, house_num, SOLAR_THRESHOLD)
                
                if not house_data.empty:
                    all_data_list.append(house_data)
                    
                    # Analyze this house
                    house_analysis = analyze_threshold_conditions(house_data)
                    
                    if 'error' not in house_analysis:
                        house_results_list.append({
                            'house_num': house_num,
                            'week': week,
                            'year': 2000 + year,
                            'n_points': len(house_data),
                            'pct_above': house_analysis['pct_above'],
                            'temp_threshold_mean': house_analysis['outdoor_temp_when_above']['mean'],
                            'temp_threshold_median': house_analysis['outdoor_temp_when_above']['median'],
                            'temp_threshold_q25': house_analysis['outdoor_temp_when_above']['q25'],
                            'temp_threshold_q75': house_analysis['outdoor_temp_when_above']['q75'],
                            'solar_threshold_mean': house_analysis['solar_when_above']['mean'],
                            'solar_threshold_median': house_analysis['solar_when_above']['median'],
                            'solar_threshold_q25': house_analysis['solar_when_above']['q25'],
                            'solar_threshold_q75': house_analysis['solar_when_above']['q75']
                        })
    
    # Combine all data
    if not all_data_list:
        print("\nNo data available for analysis!")
        print("Please check that:")
        print("  1. Data files exist in the expected location")
        print("  2. Data files contain required columns: T-RWY, ext__T, ext__SR")
        print("  3. Solar irradiance data is available")
        return None, None, None, None
    
    all_data = pd.concat(all_data_list, ignore_index=True)
    house_results = pd.DataFrame(house_results_list)
    
    print(f"\nTotal DAYTIME data points collected: {len(all_data):,}")
    print(f"Average solar irradiance in dataset: {all_data['solar_irradiance'].mean():.1f} W/m²")
    print(f"Houses with valid daytime data: {house_results['house_num'].nunique()}")
    
    # Overall analysis
    print("\n" + "="*60)
    print("ANALYZING THRESHOLD CONDITIONS")
    print("="*60)
    
    overall_analysis = analyze_threshold_conditions(all_data)
    minimum_conditions = find_minimum_conditions(all_data)
    
    # Temporal patterns
    temporal_patterns = {}
    for year in [2008, 2009]:
        year_data = all_data[all_data['year'] == year]
        if not year_data.empty:
            year_above = year_data[year_data['above_target']]
            temporal_patterns[f'year_{year}'] = {
                'pct_above': (year_data['above_target'].mean() * 100),
                'avg_temp': year_above['outdoor_temp'].mean() if len(year_above) > 0 else np.nan,
                'avg_solar': year_above['solar_irradiance'].mean() if len(year_above) > 0 else np.nan
            }
    
    # Build predictive model
    print("\nBuilding predictive model...")
    model_results = build_predictive_model(all_data)
    
    # Prepare summary
    results_summary = {
        'total_points': len(all_data),
        'points_above': all_data['above_target'].sum(),
        'overall_pct_above': all_data['above_target'].mean() * 100,
        'overall_thresholds': {
            'outdoor_temp': overall_analysis['outdoor_temp_when_above'],
            'solar': overall_analysis['solar_when_above']
        },
        'minimum_conditions': minimum_conditions,
        'temporal_patterns': temporal_patterns
    }
    
    # Group house results by house number (aggregate across weeks/years)
    house_summary = house_results.groupby('house_num').agg({
        'pct_above': 'mean',
        'temp_threshold_median': 'mean',
        'temp_threshold_q25': 'mean',
        'temp_threshold_q75': 'mean',
        'solar_threshold_median': 'mean',
        'solar_threshold_q25': 'mean',
        'solar_threshold_q75': 'mean'
    }).reset_index()
    
    # Save results
    print("\nSaving results...")
    
    # Save detailed data
    all_data.to_csv(TABLES_DIR / 'question_06_all_data.csv', index=False)
    house_results.to_csv(TABLES_DIR / 'question_06_house_results.csv', index=False)
    house_summary.to_csv(TABLES_DIR / 'question_06_house_summary.csv', index=False)
    
    # Save summary statistics
    summary_df = pd.DataFrame([results_summary])
    summary_df.to_csv(TABLES_DIR / 'question_06_summary.csv', index=False)
    
    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    create_threshold_visualization(all_data, results_summary)
    print("✓ Threshold analysis visualization created")
    
    create_house_comparison_plot(house_summary)
    print("✓ House comparison plot created")
    
    create_model_performance_plot(model_results)
    print("✓ Model performance plot created")
    
    # Print comprehensive summary
    print_comprehensive_summary(results_summary, house_summary, model_results)
    
    # Final summary statistics
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nKey Results:")
    print(f"  Median outdoor temperature required: {results_summary['overall_thresholds']['outdoor_temp']['median']:.1f}°C")
    print(f"  Median solar irradiance required: {results_summary['overall_thresholds']['solar']['median']:.0f} W/m²")
    print(f"  Overall achievement rate: {results_summary['overall_pct_above']:.1f}%")
    print(f"\nResults saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*80)
    
    return all_data, house_summary, results_summary, model_results


if __name__ == "__main__":
    all_data, house_summary, results_summary, model_results = main()