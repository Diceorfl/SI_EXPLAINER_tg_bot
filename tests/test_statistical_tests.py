import pytest
import pandas as pd
import numpy as np

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats import contingency_tables
from statsmodels.stats.weightstats import ztest
from statsmodels.stats import proportion
from statistical_tests import ttest_ind, f_oneway, tukeys_test, ttest_rel, anovaRM, z_test, mannwhitneyu,\
                              kruskal_wallis_test, wilcoxon, friedmanchisquare, chi2, mcnemar, cochrans_q,\
                              proportions_ztest


def test_ttest_ind():
    rng = np.random.default_rng()

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    p_value = stats.ttest_ind(df[df["Clusters"] == 0]["Feature"], df[df["Clusters"] == 1]["Feature"])[1]
    test_p_value = list(ttest_ind(df).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    p_value = stats.ttest_ind(df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"])[1]
    test_p_value = list(ttest_ind(df, features=["Feature1"]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    p_value = [stats.ttest_ind(df[df["Clusters"] == 0]["Feature0"], df[df["Clusters"] == 1]["Feature0"])[1],
               stats.ttest_ind(df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"])[1]]
    test_p_value = list(ttest_ind(df).values())
    assert (test_p_value[0] == p_value[0]) & (test_p_value[1] == p_value[1])

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = stats.ttest_ind(df[df["Clusters"] == 2]["Feature"], df[df["Clusters"] == 3]["Feature"])[1]
    test_p_value = list(ttest_ind(df, clusters=[2, 3]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = stats.ttest_ind(df[df["Clusters"] == 2]["Feature1"], df[df["Clusters"] == 3]["Feature1"])[1]
    test_p_value = list(ttest_ind(df, clusters=[2, 3], features=["Feature1"]).values())[0]
    assert test_p_value == p_value


def test_f_oneway():
    rng = np.random.default_rng()

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = stats.f_oneway(df[df["Clusters"] == 0]["Feature"], df[df["Clusters"] == 1]["Feature"],
                             df[df["Clusters"] == 2]["Feature"], df[df["Clusters"] == 3]["Feature"])[1]
    test_p_value = list(f_oneway(df).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = stats.f_oneway(df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"],
                             df[df["Clusters"] == 2]["Feature1"], df[df["Clusters"] == 3]["Feature1"])[1]
    test_p_value = list(f_oneway(df, features=["Feature1"]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = [stats.f_oneway(df[df["Clusters"] == 0]["Feature0"], df[df["Clusters"] == 1]["Feature0"],
                              df[df["Clusters"] == 2]["Feature0"], df[df["Clusters"] == 3]["Feature0"])[1],
               stats.f_oneway(df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"],
                              df[df["Clusters"] == 2]["Feature1"], df[df["Clusters"] == 3]["Feature1"])[1]]
    test_p_value = list(f_oneway(df).values())
    assert (test_p_value[0] == p_value[0]) & (test_p_value[1] == p_value[1])

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = stats.f_oneway(df[df["Clusters"] == 1]["Feature"], df[df["Clusters"] == 2]["Feature"])[1]
    test_p_value = list(f_oneway(df, clusters=[1, 2]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = stats.f_oneway(df[df["Clusters"] == 1]["Feature1"], df[df["Clusters"] == 2]["Feature1"])[1]
    test_p_value = list(f_oneway(df, clusters=[1, 2], features=["Feature1"]).values())[0]
    assert test_p_value == p_value


def test_tukeys_test():
    rng = np.random.default_rng()

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    res_table = pairwise_tukeyhsd(df[df["Clusters"].isin([0, 1])]["Feature"],
                                  df[df["Clusters"].isin([0, 1])]["Clusters"])
    test_res_table = tukeys_test(df, clusters=[0, 1])
    assert res_table.__dict__["pvalues"][0] == list(test_res_table.values())[0]

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    res_table = pairwise_tukeyhsd(df[df["Clusters"].isin([0, 1, 2, 3])]["Feature1"],
                                  df[df["Clusters"].isin([0, 1, 2, 3])]["Clusters"])
    test_res_table = tukeys_test(df, features=["Feature1"])
    assert (res_table.__dict__["pvalues"][0] == list(test_res_table.values())[0]) &\
           (res_table.__dict__["pvalues"][1] == list(test_res_table.values())[1])


def test_ttest_rel():
    rng = np.random.default_rng()

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    p_value = stats.ttest_rel(df[df["Clusters"] == 0]["Feature"], df[df["Clusters"] == 1]["Feature"])[1]
    test_p_value = list(ttest_rel(df).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    p_value = stats.ttest_rel(df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"])[1]
    test_p_value = list(ttest_rel(df, features=["Feature1"]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    p_value = [stats.ttest_rel(df[df["Clusters"] == 0]["Feature0"], df[df["Clusters"] == 1]["Feature0"])[1],
               stats.ttest_rel(df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"])[1]]
    test_p_value = list(ttest_rel(df).values())
    assert (test_p_value[0] == p_value[0]) & (test_p_value[1] == p_value[1])

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = stats.ttest_rel(df[df["Clusters"] == 2]["Feature"], df[df["Clusters"] == 3]["Feature"])[1]
    test_p_value = list(ttest_rel(df, clusters=[2, 3]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = stats.ttest_rel(df[df["Clusters"] == 2]["Feature1"], df[df["Clusters"] == 3]["Feature1"])[1]
    test_p_value = list(ttest_rel(df, clusters=[2, 3], features=["Feature1"]).values())[0]
    assert test_p_value == p_value


def test_anovaRM():
    df = pd.DataFrame({'patient': np.repeat([1, 2, 3, 4, 5], 4),
                       'drug': np.tile([1, 2, 3, 4], 5),
                       'response': [30, 28, 16, 34,
                                    14, 18, 10, 22,
                                    24, 20, 18, 30,
                                    38, 34, 20, 44,
                                    26, 28, 14, 30]})
    res = AnovaRM(data=df, depvar='response', subject='patient', within=['drug']).fit()
    res = res.summary()
    res = res.tables[0]
    test_res = anovaRM(df, depvar='response', subject='patient', within=['drug'])
    assert res.equals(test_res)


def test_z_test():
    rng = np.random.default_rng()

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    p_value = ztest(df[df["Clusters"] == 0]["Feature"], df[df["Clusters"] == 1]["Feature"])[1]
    test_p_value = list(z_test(df).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    p_value = ztest(df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"])[1]
    test_p_value = list(z_test(df, features=["Feature1"]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    p_value = [ztest(df[df["Clusters"] == 0]["Feature0"], df[df["Clusters"] == 1]["Feature0"])[1],
               ztest(df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"])[1]]
    test_p_value = list(z_test(df).values())
    assert (test_p_value[0] == p_value[0]) & (test_p_value[1] == p_value[1])

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = ztest(df[df["Clusters"] == 2]["Feature"], df[df["Clusters"] == 3]["Feature"])[1]
    test_p_value = list(z_test(df, clusters=[2, 3]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = ztest(df[df["Clusters"] == 2]["Feature1"], df[df["Clusters"] == 3]["Feature1"])[1]
    test_p_value = list(z_test(df, clusters=[2, 3], features=["Feature1"]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = ztest(df[df["Clusters"] == 2]["Feature1"], value=10)[1]
    test_p_value = list(z_test(df, clusters=[2], features=["Feature1"], val=10).values())[0]
    assert test_p_value == p_value


def test_mannwhitneyu():
    rng = np.random.default_rng()

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    p_value = stats.mannwhitneyu(df[df["Clusters"] == 0]["Feature"], df[df["Clusters"] == 1]["Feature"],
                                 alternative="two-sided")[1]
    test_p_value = list(mannwhitneyu(df).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    p_value = stats.mannwhitneyu(df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"],
                                 alternative="two-sided")[1]
    test_p_value = list(mannwhitneyu(df, features=["Feature1"]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    p_value = [stats.mannwhitneyu(df[df["Clusters"] == 0]["Feature0"], df[df["Clusters"] == 1]["Feature0"],
                                  alternative="two-sided")[1],
               stats.mannwhitneyu(df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"],
                                  alternative="two-sided")[1]]
    test_p_value = list(mannwhitneyu(df).values())
    assert (test_p_value[0] == p_value[0]) & (test_p_value[1] == p_value[1])

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = stats.mannwhitneyu(df[df["Clusters"] == 2]["Feature"], df[df["Clusters"] == 3]["Feature"],
                                 alternative="two-sided")[1]
    test_p_value = list(mannwhitneyu(df, clusters=[2, 3]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = stats.mannwhitneyu(df[df["Clusters"] == 2]["Feature1"], df[df["Clusters"] == 3]["Feature1"],
                                 alternative="two-sided")[1]
    test_p_value = list(mannwhitneyu(df, clusters=[2, 3], features=["Feature1"]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = stats.mannwhitneyu(df[df["Clusters"] == 2]["Feature1"], df[df["Clusters"] == 3]["Feature1"],
                                 alternative="less")[1]
    test_p_value = list(mannwhitneyu(df, clusters=[2, 3], features=["Feature1"], alt="less").values())[0]
    assert test_p_value == p_value


def test_kruskal_wallis_test():
    rng = np.random.default_rng()

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    p_value = stats.kruskal(df[df["Clusters"] == 0]["Feature"], df[df["Clusters"] == 1]["Feature"])[1]
    test_p_value = list(kruskal_wallis_test(df).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    p_value = stats.kruskal(df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"])[1]
    test_p_value = list(kruskal_wallis_test(df, features=["Feature1"]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    p_value = [stats.kruskal(df[df["Clusters"] == 0]["Feature0"], df[df["Clusters"] == 1]["Feature0"])[1],
               stats.kruskal(df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"])[1]]
    test_p_value = list(kruskal_wallis_test(df).values())
    assert (test_p_value[0] == p_value[0]) & (test_p_value[1] == p_value[1])

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = stats.kruskal(df[df["Clusters"] == 2]["Feature"], df[df["Clusters"] == 3]["Feature"])[1]
    test_p_value = list(kruskal_wallis_test(df, clusters=[2, 3]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = stats.kruskal(df[df["Clusters"] == 2]["Feature1"], df[df["Clusters"] == 3]["Feature1"])[1]
    test_p_value = list(kruskal_wallis_test(df, clusters=[2, 3], features=["Feature1"]).values())[0]
    assert test_p_value == p_value


def test_wilcoxon():
    rng = np.random.default_rng()

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    p_value = stats.wilcoxon(df[df["Clusters"] == 0]["Feature"], df[df["Clusters"] == 1]["Feature"],
                             alternative="two-sided")[1]
    test_p_value = list(wilcoxon(df).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    p_value = stats.wilcoxon(df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"],
                             alternative="two-sided")[1]
    test_p_value = list(wilcoxon(df, features=["Feature1"]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    p_value = [stats.wilcoxon(df[df["Clusters"] == 0]["Feature0"], df[df["Clusters"] == 1]["Feature0"],
                              alternative="two-sided")[1],
               stats.wilcoxon(df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"],
                              alternative="two-sided")[1]]
    test_p_value = list(wilcoxon(df).values())
    assert (test_p_value[0] == p_value[0]) & (test_p_value[1] == p_value[1])

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = stats.wilcoxon(df[df["Clusters"] == 2]["Feature"], df[df["Clusters"] == 3]["Feature"],
                             alternative="two-sided")[1]
    test_p_value = list(wilcoxon(df, clusters=[2, 3]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = stats.wilcoxon(df[df["Clusters"] == 2]["Feature1"], df[df["Clusters"] == 3]["Feature1"],
                             alternative="two-sided")[1]
    test_p_value = list(wilcoxon(df, clusters=[2, 3], features=["Feature1"]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    val = 5
    y = [val]*250
    p_value = stats.wilcoxon(df[df["Clusters"] == 2]["Feature1"], y, alternative="greater")[1]
    test_p_value = list(wilcoxon(df, clusters=[2], features=["Feature1"], val=val, alt="greater").values())[0]
    assert test_p_value == p_value


def test_friedmanchisquare():
    rng = np.random.default_rng()

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = stats.friedmanchisquare(df[df["Clusters"] == 0]["Feature"], df[df["Clusters"] == 1]["Feature"],
                                      df[df["Clusters"] == 2]["Feature"], df[df["Clusters"] == 3]["Feature"])[1]
    test_p_value = list(friedmanchisquare(df).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = stats.friedmanchisquare(df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"],
                                      df[df["Clusters"] == 2]["Feature1"], df[df["Clusters"] == 3]["Feature1"])[1]
    test_p_value = list(friedmanchisquare(df, features=["Feature1"]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = [stats.friedmanchisquare(df[df["Clusters"] == 0]["Feature0"], df[df["Clusters"] == 1]["Feature0"],
                                       df[df["Clusters"] == 2]["Feature0"], df[df["Clusters"] == 3]["Feature0"])[1],
               stats.friedmanchisquare(df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"],
                                       df[df["Clusters"] == 2]["Feature1"], df[df["Clusters"] == 3]["Feature1"])[1]]
    test_p_value = list(friedmanchisquare(df).values())
    assert (test_p_value[0] == p_value[0]) & (test_p_value[1] == p_value[1])

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = stats.friedmanchisquare(df[df["Clusters"] == 1]["Feature"], df[df["Clusters"] == 2]["Feature"],
                                      df[df["Clusters"] == 3]["Feature"])[1]
    test_p_value = list(friedmanchisquare(df, clusters=[1, 2, 3]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_value = stats.friedmanchisquare(df[df["Clusters"] == 1]["Feature1"], df[df["Clusters"] == 2]["Feature1"],
                                      df[df["Clusters"] == 3]["Feature1"])[1]
    test_p_value = list(friedmanchisquare(df, clusters=[1, 2, 3], features=["Feature1"]).values())[0]
    assert test_p_value == p_value


def test_chi2():
    df = pd.DataFrame({"Feature": [1, 3, 5, 2, 4, 20],
                       "Clusters": np.concatenate((np.full((1, 3), 0), np.full((1, 3), 1)), axis=None)})
    tab = pd.crosstab(np.array(df[df["Clusters"] == 0]["Feature"]),
                      np.array(df[df["Clusters"] == 1]["Feature"]))
    stat, p_value, dof, expected = stats.chi2_contingency(tab)
    test_p_value = list(chi2(df).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": [1, 4, 15, 22, 21, 24],
                       "Feature1": [34, 31, 32, 44, 43, 41],
                       "Clusters": np.concatenate((np.full((1, 3), 0), np.full((1, 3), 1)), axis=None)})
    tab = pd.crosstab(np.array(df[df["Clusters"] == 0]["Feature1"]),
                      np.array(df[df["Clusters"] == 1]["Feature1"]))
    stat, p_value, dof, expected = stats.chi2_contingency(tab)
    test_p_value = list(chi2(df, features=["Feature1"]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature": [11, 1, 12, 23, 24, 25, 32, 33, 31, 42, 420, 401],
                       "Clusters": np.concatenate((np.full((1, 3), 0), np.full((1, 3), 1),
                                                   np.full((1, 3), 2), np.full((1, 3), 3)), axis=None)})
    tab = pd.crosstab(np.array(df[df["Clusters"] == 1]["Feature"]),
                      np.array(df[df["Clusters"] == 2]["Feature"]))
    stat, p_value, dof, expected = stats.chi2_contingency(tab)
    test_p_value = list(chi2(df, clusters=[1, 2]).values())[0]
    assert test_p_value == p_value


def test_mcnemar():
    df = pd.DataFrame({"Feature": [1, 3, 5, 1, 4, 20],
                       "Clusters": np.concatenate((np.full((1, 3), 0), np.full((1, 3), 1)), axis=None)})
    table = [df[df["Clusters"] == 0]["Feature"], df[df["Clusters"] == 1]["Feature"]]
    p_value = contingency_tables.mcnemar(table).pvalue
    test_p_value = list(mcnemar(df).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": [1, 3, 5, 1, 4, 20],
                       "Feature1": [2, 3, 3, 2, 4, 3],
                       "Clusters": np.concatenate((np.full((1, 3), 0), np.full((1, 3), 1)), axis=None)})
    table = [df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"]]
    p_value = contingency_tables.mcnemar(table).pvalue
    test_p_value = list(mcnemar(df, features=["Feature1"]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": [1, 3, 5, 1, 4, 20, 4, 5, 6, 6, 23, 1],
                       "Feature1": [2, 3, 3, 2, 4, 3, 4, 8, 6, 1, 23, 4],
                       "Clusters": np.concatenate((np.full((1, 3), 0), np.full((1, 3), 1),
                                                   np.full((1, 3), 2), np.full((1, 3), 3)), axis=None)})
    table = [df[df["Clusters"] == 1]["Feature1"], df[df["Clusters"] == 2]["Feature1"]]
    p_value = contingency_tables.mcnemar(table).pvalue
    test_p_value = list(mcnemar(df, features=["Feature1"], clusters=[1, 2]).values())[0]
    assert test_p_value == p_value


def test_cochrans_q():
    df = pd.DataFrame({"Feature0": [1, 3, 5, 1, 4, 20, 4, 5, 6, 6, 23, 1],
                       "Feature1": [2, 3, 3, 2, 4, 3, 4, 8, 6, 1, 23, 4],
                       "Clusters": np.concatenate((np.full((1, 3), 0), np.full((1, 3), 1),
                                                   np.full((1, 3), 2), np.full((1, 3), 3)), axis=None)})
    table = [df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"],
             df[df["Clusters"] == 2]["Feature1"], df[df["Clusters"] == 3]["Feature1"]]
    p_value = contingency_tables.cochrans_q(table).pvalue
    test_p_value = list(cochrans_q(df, features=["Feature1"]).values())[0]
    assert p_value == test_p_value

    df = pd.DataFrame({"Feature0": [1, 3, 5, 1, 4, 20, 4, 5, 6, 6, 23, 1],
                       "Feature1": [2, 3, 3, 2, 4, 3, 4, 8, 6, 1, 23, 4],
                       "Clusters": np.concatenate((np.full((1, 3), 0), np.full((1, 3), 1),
                                                   np.full((1, 3), 2), np.full((1, 3), 3)), axis=None)})
    table = [df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"],
             df[df["Clusters"] == 2]["Feature1"], df[df["Clusters"] == 3]["Feature1"]]
    p_value = contingency_tables.cochrans_q(table).pvalue
    test_p_value = list(cochrans_q(df, features=["Feature1"], clusters=[1, 2, 3]).values())[0]
    assert p_value == test_p_value

    df = pd.DataFrame({"Feature": [1, 3, 5, 1, 4, 20, 4, 5, 6, 6, 23, 1],
                       "Clusters": np.concatenate((np.full((1, 3), 0), np.full((1, 3), 1),
                                                   np.full((1, 3), 2), np.full((1, 3), 3)), axis=None)})
    table = [df[df["Clusters"] == 0]["Feature"], df[df["Clusters"] == 1]["Feature"],
             df[df["Clusters"] == 2]["Feature"], df[df["Clusters"] == 3]["Feature"]]
    p_value = contingency_tables.cochrans_q(table).pvalue
    test_p_value = list(cochrans_q(df).values())[0]
    assert p_value == test_p_value


def test_proportions_ztest():
    df = pd.DataFrame({"Feature": [1, 0, 1, 0, 0, 1],
                       "Clusters": np.concatenate((np.full((1, 3), 0), np.full((1, 3), 1)), axis=None)})
    count = [2, 1]
    nobs = [3, 3]
    p_value = proportion.proportions_ztest(count, nobs)[1]
    test_p_value = list(proportions_ztest(df).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": [1, 0, 1, 0, 0, 1],
                       "Feature1": [1, 1, 1, 0, 0, 1],
                       "Clusters": np.concatenate((np.full((1, 3), 0), np.full((1, 3), 1)), axis=None)})
    count = [3, 1]
    nobs = [3, 3]
    p_value = proportion.proportions_ztest(count, nobs)[1]
    test_p_value = list(proportions_ztest(df, features=["Feature1"]).values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature0": [1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                       "Feature1": [1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
                       "Clusters": np.concatenate((np.full((1, 3), 0), np.full((1, 3), 1),
                                                  np.full((1, 3), 2), np.full((1, 3), 3)), axis=None)})
    count = [1, 0]
    nobs = [3, 3]
    p_value = proportion.proportions_ztest(count, nobs)[1]
    test_p_value = list(proportions_ztest(df, features=["Feature1"], clusters=[1, 3]).values())[0]
    assert test_p_value == p_value


if __name__ == '__main__':
    pytest.main()
