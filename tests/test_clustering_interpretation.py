import pytest
import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats import contingency_tables
from clustering_interpretation import features_and_clusters, normal_distribution_features, \
                                      non_normal_distribution_features, categorical_features,\
                                      calculate_p_values, ClusteringInterpretation


def test_features_and_clusters():
    rng = np.random.default_rng()

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    features = df.columns[:-1]
    clusters = [0, 1, 2, 3]
    test_features, test_clusters = features_and_clusters(df)
    assert (list(test_features) == list(features)) & (test_clusters == clusters)


def test_normal_distribution_features():
    rng = np.random.default_rng()

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})

    p_values = normal_distribution_features(df=df, clusters=[0, 1], feature="Feature0", dependent_clusters=True)
    p_values = list(p_values.values())[0]
    test_p_values = stats.ttest_rel(df[df["Clusters"] == 0]["Feature0"], df[df["Clusters"] == 1]["Feature0"])[1]
    assert p_values == test_p_values

    df = pd.DataFrame({"Feature0": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Feature1": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    test_p_values = stats.f_oneway(df[df["Clusters"] == 0]["Feature0"], df[df["Clusters"] == 1]["Feature0"],
                                   df[df["Clusters"] == 2]["Feature0"], df[df["Clusters"] == 3]["Feature0"])[1]
    p_values = normal_distribution_features(df=df, clusters=[0, 1, 2, 3], feature="Feature0", dependent_clusters=False)
    p_values = list(p_values.values())[0]
    assert p_values == test_p_values

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    test_p_values = stats.ttest_ind(df[df["Clusters"] == 0]["Feature"], df[df["Clusters"] == 1]["Feature"])[1]
    p_values = normal_distribution_features(df=df, clusters=[0, 1], feature="Feature", dependent_clusters=False)
    p_values = list(p_values.values())[0]
    assert p_values == test_p_values


def test_non_normal_distribution_features():
    rng = np.random.default_rng()

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_values = non_normal_distribution_features(df=df, clusters=[0, 1, 2, 3], feature="Feature", dependent_clusters=True)
    p_values = list(p_values.values())[0]
    test_p_values = stats.friedmanchisquare(df[df["Clusters"] == 0]["Feature"], df[df["Clusters"] == 1]["Feature"],
                                            df[df["Clusters"] == 2]["Feature"], df[df["Clusters"] == 3]["Feature"])[1]
    assert test_p_values == p_values

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})
    p_values = non_normal_distribution_features(df=df, clusters=[0, 1], feature="Feature", dependent_clusters=True)
    p_values = list(p_values.values())[0]
    test_p_values = stats.wilcoxon(df[df["Clusters"] == 0]["Feature"], df[df["Clusters"] == 1]["Feature"],
                                   alternative="two-sided")[1]
    assert test_p_values == p_values

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_values = non_normal_distribution_features(df=df, clusters=[0, 2, 3], feature="Feature", dependent_clusters=False)
    p_values = list(p_values.values())[0]
    test_p_values = stats.kruskal(df[df["Clusters"] == 0]["Feature"], df[df["Clusters"] == 2]["Feature"])[1]
    assert test_p_values == p_values

    df = pd.DataFrame({"Feature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "Clusters": np.concatenate((np.full((1, 250), 0), np.full((1, 250), 1),
                                                   (np.full((1, 250), 2), np.full((1, 250), 3))), axis=None)})
    p_values = non_normal_distribution_features(df=df, clusters=[0, 2], feature="Feature", dependent_clusters=False)
    p_values = list(p_values.values())[0]
    test_p_values = stats.mannwhitneyu(df[df["Clusters"] == 0]["Feature"], df[df["Clusters"] == 2]["Feature"],
                                       alternative="two-sided")[1]
    assert test_p_values == p_values


def test_categorical_features():
    df = pd.DataFrame({"Feature0": [1, 3, 5, 1, 4, 20, 4, 5, 6, 6, 23, 1],
                       "Feature1": [2, 3, 3, 2, 4, 3, 4, 8, 6, 1, 23, 4],
                       "Clusters": np.concatenate((np.full((1, 3), 0), np.full((1, 3), 1),
                                                   np.full((1, 3), 2), np.full((1, 3), 3)), axis=None)})
    table = [df[df["Clusters"] == 0]["Feature1"], df[df["Clusters"] == 1]["Feature1"],
             df[df["Clusters"] == 2]["Feature1"], df[df["Clusters"] == 3]["Feature1"]]
    test_p_value = contingency_tables.cochrans_q(table).pvalue
    p_value = categorical_features(df=df, clusters=[0, 1, 2, 3], feature="Feature1", dependent_clusters=True)
    p_value = list(p_value.values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature": [1, 3, 5, 1, 4, 20],
                       "Clusters": np.concatenate((np.full((1, 3), 0), np.full((1, 3), 1)), axis=None)})
    table = [df[df["Clusters"] == 0]["Feature"], df[df["Clusters"] == 1]["Feature"]]
    test_p_value = contingency_tables.mcnemar(table).pvalue
    p_value = categorical_features(df=df, clusters=[0, 1], feature="Feature", dependent_clusters=True)
    p_value = list(p_value.values())[0]
    assert test_p_value == p_value

    df = pd.DataFrame({"Feature": [11, 1, 12, 23, 24, 25, 32, 33, 31, 42, 420, 401],
                       "Clusters": np.concatenate((np.full((1, 3), 0), np.full((1, 3), 1),
                                                   np.full((1, 3), 2), np.full((1, 3), 3)), axis=None)})
    tab = pd.crosstab(np.array(df[df["Clusters"] == 1]["Feature"]),
                      np.array(df[df["Clusters"] == 2]["Feature"]))
    stat, test_p_value, dof, expected = stats.chi2_contingency(tab)
    p_value = categorical_features(df=df, clusters=[1, 2], feature="Feature", dependent_clusters=False)
    p_value = list(p_value.values())[0]
    assert test_p_value == p_value


def test_calculate_p_values():
    rng = np.random.default_rng()

    df = pd.DataFrame({"normFeature": stats.norm.rvs(loc=5, scale=10, size=1000, random_state=rng),
                       "nonnormFeature": np.random.exponential(scale=1.0, size=1000),
                       "categoricalFeature": [3]*500 + [4]*500,
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})

    norm_p_value = stats.ttest_ind(df[df["Clusters"] == 0]["normFeature"], df[df["Clusters"] == 1]["normFeature"])[1]
    nonnorm_p_value = stats.mannwhitneyu(df[df["Clusters"] == 0]["nonnormFeature"],
                                         df[df["Clusters"] == 1]["nonnormFeature"], alternative="two-sided")[1]
    tab = pd.crosstab(np.array(df[df["Clusters"] == 0]["categoricalFeature"]),
                      np.array(df[df["Clusters"] == 1]["categoricalFeature"]))
    stat, categorical_p_value, dof, expected = stats.chi2_contingency(tab)

    p_values = calculate_p_values(df=df, continuous=["normFeature", "nonnormFeature"], categorical=["categoricalFeature"],
                                  dependent_clusters=False)
    p_values = p_values["normFeature"], p_values["nonnormFeature"], p_values["categoricalFeature"]
    p_values = list(p_values[0].values())[0], list(p_values[1].values())[0], list(p_values[2].values())[0]

    assert (norm_p_value == p_values[0]) & (nonnorm_p_value == p_values[1]) & (categorical_p_value == p_values[2])


def test_make_interpretation():
    rng = np.random.default_rng()

    df = pd.DataFrame({"normFeature": np.concatenate((stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng),
                                                      stats.norm.rvs(loc=5, scale=10, size=500,
                                                                     random_state=rng) + 100)),
                       "nonnormFeature": np.concatenate((np.random.exponential(scale=1.0, size=500),
                                                         np.random.exponential(scale=1.0, size=500) + 100)),
                       "categoricalFeature": [3] * 500 + [4] * 500,
                       "Clusters": np.concatenate((np.full((1, 500), 0), np.full((1, 500), 1)), axis=None)})

    interpretation_df = ClusteringInterpretation(df, clusters="Clusters")
    interpretation_df.set_continuous_and_categorical(continuous=["normFeature", "nonnormFeature"],
                                                     categorical=["categoricalFeature"])
    interpretation_df.set_dependent_clusters(dependent_clusters=False)
    interpretation = interpretation_df.get_significant_features()

    assert interpretation[0] == ["normFeature", "nonnormFeature"]


if __name__ == '__main__':
    pytest.main()
