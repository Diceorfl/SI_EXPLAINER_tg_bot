from scipy import stats
import pandas as pd


def shapiro_wilk_test(feature: pd.Series, alpha: float = 0.05) -> bool:
    stat, p_value = stats.shapiro(feature)
    return p_value > alpha


def normal_test(feature: pd.Series, alpha: float = 0.05) -> bool:
    stat, p_value = stats.normaltest(feature)
    return p_value > alpha


def anderson_test(feature: pd.Series) -> bool:
    result = stats.anderson(feature)
    for i in range(len(result.critical_values)):
        if result.statistic > result.critical_values[i]:
            return False
    return True


def is_normal(feature: pd.Series, alpha: float = 0.05) -> bool:
    sw_t = shapiro_wilk_test(feature, alpha)
    n_t = normal_test(feature, alpha)
    an_t = anderson_test(feature)
    return sw_t & n_t & an_t
