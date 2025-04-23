# All-in-one Python script for hypothesis testing

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. One-Sample Z-Test ---
def one_sample_z_test(sample, pop_mean, pop_std):
    sample_mean = np.mean(sample)
    z = (sample_mean - pop_mean) / (pop_std / np.sqrt(len(sample)))
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value

# --- 2. Simulate data for Z-test ---
def simulate_data_for_z_test():
    np.random.seed(0)
    sample = np.random.normal(loc=52, scale=10, size=30)
    z, p = one_sample_z_test(sample, 50, 10)
    return sample, z, p

# --- 3. Visualize Two-Tailed Z-Test ---
def visualize_z_test(z_stat, alpha=0.05):
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x)
    plt.plot(x, y)
    plt.fill_between(x, y, where=(x <= stats.norm.ppf(alpha/2)), color='red', alpha=0.3)
    plt.fill_between(x, y, where=(x >= stats.norm.ppf(1 - alpha/2)), color='red', alpha=0.3)
    plt.axvline(z_stat, color='blue', linestyle='--')
    plt.title('Two-Tailed Z-Test')
    plt.show()

# --- 4. Type I and Type II Errors Visualization ---
def visualize_type_errors(mu0, mu1, sigma, n, alpha):
    x = np.linspace(mu0 - 4*sigma, mu1 + 4*sigma, 1000)
    se = sigma / np.sqrt(n)
    z_crit = stats.norm.ppf(1 - alpha)
    crit_val = mu0 + z_crit * se
    
    y0 = stats.norm.pdf(x, mu0, se)
    y1 = stats.norm.pdf(x, mu1, se)
    
    plt.plot(x, y0, label='H0')
    plt.plot(x, y1, label='H1')
    plt.axvline(crit_val, color='red', linestyle='--', label='Critical Value')
    plt.fill_between(x, y0, where=(x >= crit_val), alpha=0.3, color='red', label='Type I Error')
    plt.fill_between(x, y1, where=(x < crit_val), alpha=0.3, color='blue', label='Type II Error')
    plt.legend()
    plt.title('Type I and Type II Errors')
    plt.show()

# --- 5. Independent T-Test ---
def independent_t_test(data1, data2):
    t_stat, p_value = stats.ttest_ind(data1, data2)
    return t_stat, p_value

# --- 6. Paired Sample T-Test ---
def paired_t_test(before, after):
    t_stat, p_value = stats.ttest_rel(before, after)
    return t_stat, p_value

# --- 7. Compare Z-Test and T-Test ---
def compare_z_t_test():
    sample = np.random.normal(50, 10, 30)
    z_stat, z_p = one_sample_z_test(sample, 50, 10)
    t_stat, t_p = stats.ttest_1samp(sample, 50)
    return z_stat, z_p, t_stat, t_p

# --- 8. Confidence Interval ---
def confidence_interval(sample, confidence=0.95):
    mean = np.mean(sample)
    sem = stats.sem(sample)
    margin = sem * stats.t.ppf((1 + confidence) / 2, df=len(sample)-1)
    return mean - margin, mean + margin

# --- 9. Margin of Error ---
def margin_of_error(sample, confidence=0.95):
    sem = stats.sem(sample)
    margin = sem * stats.t.ppf((1 + confidence) / 2, df=len(sample)-1)
    return margin

# --- 10. Bayesian Inference ---
def bayes_theorem(prior_a, prior_b, likelihood_a, likelihood_b):
    norm = prior_a * likelihood_a + prior_b * likelihood_b
    posterior_a = (prior_a * likelihood_a) / norm
    posterior_b = (prior_b * likelihood_b) / norm
    return posterior_a, posterior_b

# --- 11. Chi-Square Independence Test ---
def chi_square_test_independence(table):
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return chi2, p, expected

# --- 12. Chi-Square Expected Frequencies ---
def calculate_expected_frequencies(observed):
    _, _, _, expected = stats.chi2_contingency(observed)
    return expected

# --- 13. Chi-Square Goodness of Fit ---
def chi_square_goodness_of_fit(observed, expected):
    chi2, p = stats.chisquare(f_obs=observed, f_exp=expected)
    return chi2, p

# --- 14. Visualize Chi-Square Distribution ---
def visualize_chi_square(df):
    x = np.linspace(0, 30, 1000)
    y = stats.chi2.pdf(x, df)
    plt.plot(x, y)
    plt.title('Chi-Square Distribution (df={})'.format(df))
    plt.show()

# --- 15. F-Test ---
def f_test(sample1, sample2):
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)
    f_stat = var1 / var2
    dfn = len(sample1) - 1
    dfd = len(sample2) - 1
    p_value = 1 - stats.f.cdf(f_stat, dfn, dfd)
    return f_stat, p_value

# --- 16. One-Way ANOVA ---
def one_way_anova(*groups):
    f_stat, p_value = stats.f_oneway(*groups)
    return f_stat, p_value

# --- 17. ANOVA Assumptions ---
def check_anova_assumptions(groups):
    normality = [stats.shapiro(group).pvalue > 0.05 for group in groups]
    equal_var = stats.levene(*groups).pvalue > 0.05
    return all(normality), equal_var

# --- 18. Two-Way ANOVA ---
# For simplicity, assume data in DataFrame format with statsmodels
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

def two_way_anova(df, formula='value ~ C(factor1) + C(factor2) + C(factor1):C(factor2)'):
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table

# --- 19. Visualize F-distribution ---
def visualize_f_distribution(dfn, dfd):
    x = np.linspace(0.01, 5, 1000)
    y = stats.f.pdf(x, dfn, dfd)
    plt.plot(x, y)
    plt.title(f'F-distribution (dfn={dfn}, dfd={dfd})')
    plt.show()

# --- 20. Chi-Square Population Variance ---
def chi_square_variance_test(sample, pop_var):
    n = len(sample)
    sample_var = np.var(sample, ddof=1)
    chi2 = (n - 1) * sample_var / pop_var
    p_value = 1 - stats.chi2.cdf(chi2, df=n-1)
    return chi2, p_value
