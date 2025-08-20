"""
Statistical testing functions for HRV study analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, List, Optional
import warnings


def independent_t_test(group1: np.ndarray, group2: np.ndarray,
                      alternative: str = 'two-sided',
                      equal_var: bool = False) -> Dict:
    """
    Perform independent samples t-test with comprehensive output.
    
    Parameters:
    -----------
    group1, group2 : array-like
        Data for two groups
    alternative : str
        'two-sided', 'less', or 'greater'
    equal_var : bool
        If True, assume equal variances (Student's t-test)
        If False, use Welch's t-test
    
    Returns:
    --------
    dict
        Test results including statistics, p-value, and effect size
    """
    # Clean data
    group1 = np.array(group1)[~np.isnan(group1)]
    group2 = np.array(group2)[~np.isnan(group2)]
    
    if len(group1) == 0 or len(group2) == 0:
        return {
            't_statistic': np.nan,
            'p_value': np.nan,
            'df': np.nan,
            'mean_diff': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'cohen_d': np.nan,
            'power': np.nan
        }
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(group1, group2, 
                                      alternative=alternative,
                                      equal_var=equal_var)
    
    # Calculate degrees of freedom
    if equal_var:
        df = len(group1) + len(group2) - 2
    else:
        # Welch-Satterthwaite equation
        s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
    
    # Mean difference and confidence interval
    mean_diff = np.mean(group1) - np.mean(group2)
    se_diff = np.sqrt(np.var(group1, ddof=1)/len(group1) + 
                     np.var(group2, ddof=1)/len(group2))
    ci = stats.t.interval(0.95, df, loc=mean_diff, scale=se_diff)
    
    # Cohen's d effect size
    pooled_std = np.sqrt(((len(group1)-1)*np.var(group1, ddof=1) + 
                          (len(group2)-1)*np.var(group2, ddof=1)) / 
                         (len(group1) + len(group2) - 2))
    cohen_d = mean_diff / pooled_std if pooled_std > 0 else np.nan
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'df': df,
        'mean_diff': mean_diff,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'cohen_d': cohen_d,
        'group1_mean': np.mean(group1),
        'group2_mean': np.mean(group2),
        'group1_std': np.std(group1, ddof=1),
        'group2_std': np.std(group2, ddof=1),
        'group1_n': len(group1),
        'group2_n': len(group2)
    }


def paired_t_test(before: np.ndarray, after: np.ndarray,
                 alternative: str = 'two-sided') -> Dict:
    """
    Perform paired samples t-test.
    
    Parameters:
    -----------
    before, after : array-like
        Paired data
    alternative : str
        'two-sided', 'less', or 'greater'
    
    Returns:
    --------
    dict
        Test results
    """
    # Ensure same length and remove pairs with any NaN
    before = np.array(before)
    after = np.array(after)
    
    if len(before) != len(after):
        raise ValueError("Paired samples must have same length")
    
    # Remove pairs with NaN
    valid_pairs = ~(np.isnan(before) | np.isnan(after))
    before = before[valid_pairs]
    after = after[valid_pairs]
    
    if len(before) == 0:
        return {
            't_statistic': np.nan,
            'p_value': np.nan,
            'df': np.nan,
            'mean_diff': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'cohen_d': np.nan
        }
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(before, after, alternative=alternative)
    
    # Calculate differences
    differences = after - before
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    se_diff = std_diff / np.sqrt(len(differences))
    
    # Confidence interval
    df = len(differences) - 1
    ci = stats.t.interval(0.95, df, loc=mean_diff, scale=se_diff)
    
    # Cohen's d for paired samples
    cohen_d = mean_diff / std_diff if std_diff > 0 else np.nan
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'df': df,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'cohen_d': cohen_d,
        'n_pairs': len(differences)
    }


def anova_one_way(groups: List[np.ndarray], group_names: Optional[List[str]] = None) -> Dict:
    """
    Perform one-way ANOVA with post-hoc tests.
    
    Parameters:
    -----------
    groups : list of array-like
        Data for each group
    group_names : list of str, optional
        Names for each group
    
    Returns:
    --------
    dict
        ANOVA results and post-hoc comparisons
    """
    # Clean data
    cleaned_groups = []
    for g in groups:
        clean_g = np.array(g)[~np.isnan(g)]
        if len(clean_g) > 0:
            cleaned_groups.append(clean_g)
    
    if len(cleaned_groups) < 2:
        return {
            'f_statistic': np.nan,
            'p_value': np.nan,
            'df_between': np.nan,
            'df_within': np.nan,
            'eta_squared': np.nan,
            'post_hoc': {}
        }
    
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*cleaned_groups)
    
    # Calculate effect size (eta-squared)
    all_data = np.concatenate(cleaned_groups)
    grand_mean = np.mean(all_data)
    
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in cleaned_groups)
    ss_total = np.sum((all_data - grand_mean)**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else np.nan
    
    # Degrees of freedom
    k = len(cleaned_groups)  # number of groups
    N = sum(len(g) for g in cleaned_groups)  # total observations
    df_between = k - 1
    df_within = N - k
    
    # Post-hoc tests (Tukey HSD) if ANOVA is significant
    post_hoc_results = {}
    if p_value < 0.05 and len(cleaned_groups) > 2:
        from scipy.stats import tukey_hsd
        result = tukey_hsd(*cleaned_groups)
        
        if group_names is None:
            group_names = [f"Group_{i+1}" for i in range(len(cleaned_groups))]
        
        for i in range(len(cleaned_groups)):
            for j in range(i+1, len(cleaned_groups)):
                key = f"{group_names[i]}_vs_{group_names[j]}"
                idx = i * len(cleaned_groups) + j
                post_hoc_results[key] = {
                    'mean_diff': np.mean(cleaned_groups[i]) - np.mean(cleaned_groups[j]),
                    'p_value': result.pvalue[i, j] if i < len(result.pvalue) and j < len(result.pvalue[0]) else np.nan
                }
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'df_between': df_between,
        'df_within': df_within,
        'eta_squared': eta_squared,
        'group_means': [np.mean(g) for g in cleaned_groups],
        'group_stds': [np.std(g, ddof=1) for g in cleaned_groups],
        'group_ns': [len(g) for g in cleaned_groups],
        'post_hoc': post_hoc_results
    }


def mann_whitney_u(group1: np.ndarray, group2: np.ndarray,
                  alternative: str = 'two-sided') -> Dict:
    """
    Perform Mann-Whitney U test (non-parametric alternative to t-test).
    
    Parameters:
    -----------
    group1, group2 : array-like
        Data for two groups
    alternative : str
        'two-sided', 'less', or 'greater'
    
    Returns:
    --------
    dict
        Test results
    """
    # Clean data
    group1 = np.array(group1)[~np.isnan(group1)]
    group2 = np.array(group2)[~np.isnan(group2)]
    
    if len(group1) == 0 or len(group2) == 0:
        return {
            'u_statistic': np.nan,
            'p_value': np.nan,
            'median_diff': np.nan,
            'effect_size': np.nan
        }
    
    # Perform Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)
    
    # Calculate medians and difference
    median1 = np.median(group1)
    median2 = np.median(group2)
    median_diff = median1 - median2
    
    # Effect size (rank-biserial correlation)
    n1, n2 = len(group1), len(group2)
    effect_size = 1 - (2*u_stat) / (n1 * n2)
    
    return {
        'u_statistic': u_stat,
        'p_value': p_value,
        'median_diff': median_diff,
        'effect_size': effect_size,
        'group1_median': median1,
        'group2_median': median2,
        'group1_n': n1,
        'group2_n': n2
    }


def normality_test(data: np.ndarray, test: str = 'shapiro') -> Dict:
    """
    Test for normality of data.
    
    Parameters:
    -----------
    data : array-like
        Data to test
    test : str
        'shapiro' for Shapiro-Wilk test or 'ks' for Kolmogorov-Smirnov test
    
    Returns:
    --------
    dict
        Test results
    """
    # Clean data
    data = np.array(data)[~np.isnan(data)]
    
    if len(data) < 3:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'is_normal': False,
            'test_name': test
        }
    
    if test == 'shapiro':
        if len(data) > 5000:
            warnings.warn("Shapiro-Wilk test may be inaccurate for n > 5000")
        statistic, p_value = stats.shapiro(data)
        test_name = "Shapiro-Wilk"
    elif test == 'ks':
        # Kolmogorov-Smirnov test against normal distribution
        statistic, p_value = stats.kstest(data, 'norm', 
                                         args=(np.mean(data), np.std(data, ddof=1)))
        test_name = "Kolmogorov-Smirnov"
    else:
        raise ValueError(f"Unknown test: {test}")
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'is_normal': p_value > 0.05,
        'test_name': test_name,
        'n': len(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }


def levene_test(groups: List[np.ndarray], center: str = 'median') -> Dict:
    """
    Levene's test for equality of variances.
    
    Parameters:
    -----------
    groups : list of array-like
        Data for each group
    center : str
        'mean', 'median', or 'trimmed'
    
    Returns:
    --------
    dict
        Test results
    """
    # Clean data
    cleaned_groups = []
    for g in groups:
        clean_g = np.array(g)[~np.isnan(g)]
        if len(clean_g) > 0:
            cleaned_groups.append(clean_g)
    
    if len(cleaned_groups) < 2:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'equal_variance': False
        }
    
    statistic, p_value = stats.levene(*cleaned_groups, center=center)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'equal_variance': p_value > 0.05,
        'group_vars': [np.var(g, ddof=1) for g in cleaned_groups]
    }


def correlation_test(x: np.ndarray, y: np.ndarray, 
                    method: str = 'pearson') -> Dict:
    """
    Calculate correlation with significance test.
    
    Parameters:
    -----------
    x, y : array-like
        Variables to correlate
    method : str
        'pearson', 'spearman', or 'kendall'
    
    Returns:
    --------
    dict
        Correlation results
    """
    # Remove pairs with NaN
    x = np.array(x)
    y = np.array(y)
    valid_pairs = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_pairs]
    y = y[valid_pairs]
    
    if len(x) < 3:
        return {
            'correlation': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'method': method,
            'n': len(x)
        }
    
    if method == 'pearson':
        corr, p_value = stats.pearsonr(x, y)
        # Fisher z-transformation for confidence interval
        z = np.arctanh(corr)
        se = 1 / np.sqrt(len(x) - 3)
        z_ci = z + np.array([-1.96, 1.96]) * se
        ci = np.tanh(z_ci)
    elif method == 'spearman':
        corr, p_value = stats.spearmanr(x, y)
        ci = [np.nan, np.nan]  # CI calculation is complex for Spearman
    elif method == 'kendall':
        corr, p_value = stats.kendalltau(x, y)
        ci = [np.nan, np.nan]  # CI calculation is complex for Kendall
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'correlation': corr,
        'p_value': p_value,
        'ci_lower': ci[0] if isinstance(ci, (list, np.ndarray)) else np.nan,
        'ci_upper': ci[1] if isinstance(ci, (list, np.ndarray)) else np.nan,
        'method': method,
        'n': len(x),
        'r_squared': corr**2 if method == 'pearson' else np.nan
    }