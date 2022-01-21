import pytest
import pandas as pd

from scipy import stats
from numpy.random import randn
from distribution_check import shapiro_wilk_test, normal_test, anderson_test, is_normal


def test_shapiro_wilk_test():
    alpha = 0.05
    df = pd.DataFrame({"Feature": 5 * randn(100) + 50})
    stat, p_value = stats.shapiro(df["Feature"])
    assert (p_value > alpha) == shapiro_wilk_test(df["Feature"])


def test_normal_test():
    alpha = 0.05
    df = pd.DataFrame({"Feature": 5 * randn(100) + 50})
    stat, p_value = stats.normaltest(df["Feature"])
    assert (p_value > alpha) == normal_test(df["Feature"])


def test_anderson_test():
    df = pd.DataFrame({"Feature": 5 * randn(100) + 50})
    result = stats.anderson(df["Feature"])
    res = True
    for i in range(len(result.critical_values)):
        if result.statistic > result.critical_values[i]:
            res = False
    assert res == anderson_test(df["Feature"])


def test_is_normal():
    alpha = 0.05
    df = pd.DataFrame({"Feature": 5 * randn(100) + 50})
    sh_stat, sh_p_value = stats.shapiro(df["Feature"])
    norm_stat, norm_p_value = stats.normaltest(df["Feature"])
    result = stats.anderson(df["Feature"])
    res = True
    for i in range(len(result.critical_values)):
        if result.statistic > result.critical_values[i]:
            res = False
    assert ((sh_p_value > alpha) & (norm_p_value > alpha) & res) & is_normal(df["Feature"])


if __name__ == '__main__':
    pytest.main()