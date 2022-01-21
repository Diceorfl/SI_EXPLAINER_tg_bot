import statistical_tests as st
import pandas as pd

from distribution_check import is_normal


def features_and_clusters(df: pd.DataFrame) -> tuple:
    features = df.columns[:-1].copy()
    clusters = df["Clusters"].copy()
    clusters = clusters.dropna().unique().tolist()
    return features, clusters


def normal_distribution_features(df: pd.DataFrame, feature: str, clusters: list, dependent_clusters: bool) -> dict:
    """
    Selects a suitable statistical test for a feature whose values obey the normal distribution law
    and calculates the p-value in pairs between each of the clusters

    """
    # whether the compared groups are dependent? (True/False)
    if dependent_clusters:
        return st.ttest_rel(df=df, clusters=clusters, features=[feature])

    if len(clusters) > 2:
        p_value = st.f_oneway(df=df, clusters=clusters, features=[feature])[feature]
        if p_value < 0.05:
            return st.tukeys_test(df=df, clusters=clusters, features=[feature])
        return {feature: p_value}

    variance_p_values = st.levene(df=df, clusters=clusters, features=[feature])
    check_variance = sum([x < 0.05 for x in list(variance_p_values.values())])
    if not check_variance:
        return st.ttest_ind(df=df, clusters=clusters, features=[feature])
    return st.mannwhitneyu(df=df, clusters=clusters, features=[feature])


def non_normal_distribution_features(df: pd.DataFrame, feature: str, clusters: list, dependent_clusters: bool) -> dict:
    """
    Selects an appropriate statistical test for a feature whose values follow a non-normal distribution,
    and calculates the p-value pairwise between each of the clusters.

    """
    # whether the compared groups are dependent? (True/False)
    if dependent_clusters:
        if len(clusters) > 2:
            p_value = st.friedmanchisquare(df=df, clusters=clusters, features=[feature])[feature]
            if p_value < 0.05:
                return st.wilcoxon(df=df, clusters=clusters, features=[feature])
            return {feature: p_value}
        return st.wilcoxon(df=df, clusters=clusters, features=[feature])

    if len(clusters) > 2:
        return st.kruskal_wallis_test(df=df, clusters=clusters, features=[feature])
    return st.mannwhitneyu(df=df, clusters=clusters, features=[feature])


def categorical_features(df: pd.DataFrame, feature: str, clusters: list, dependent_clusters: bool) -> dict:
    """
    Selects an appropriate statistical test for a categorical feature
    and calculates a p-value pairwise between each of the clusters.

    """
    # whether the compared groups are dependent? (True/False)
    if dependent_clusters:
        if len(clusters) > 2:
            p_value = st.cochrans_q(df=df, clusters=clusters, features=[feature])[feature]
            if p_value < 0.05:
                return st.mcnemar(df=df, clusters=clusters, features=[feature])
            return {feature: p_value}
        return st.mcnemar(df=df, clusters=clusters, features=[feature])

    return st.chi2(df=df, clusters=clusters, features=[feature])


def calculate_p_values(df: pd.DataFrame, continuous: list, categorical: list, dependent_clusters: bool) -> dict:
    """
    Specifies the type of feature (continuous/categorical),
    and also determines whether or not a continuous feature is normally distributed.
    Returns p-values for each of the features.

    """
    features, clusters = features_and_clusters(df)

    p_values = {}

    for feature in features:
        if feature in continuous:
            if is_normal(df[feature]):
                p_values[feature] = normal_distribution_features(df, feature, clusters, dependent_clusters)
            else:
                p_values[feature] = non_normal_distribution_features(df, feature, clusters, dependent_clusters)
        elif feature in categorical:
            p_values[feature] = categorical_features(df, feature, clusters, dependent_clusters)

    return p_values


def different_clusters(differences: list) -> dict:
    """
    Identifies features that make a statistically significant difference for each pair of clusters.

    """
    diff_clusters = {}
    for diff in differences:
        if tuple((diff[1], diff[2])) not in diff_clusters.keys():
            diff_clusters[tuple((diff[1], diff[2]))] = [diff[0]]
        else:
            diff_clusters[tuple((diff[1], diff[2]))].append(diff[0])

    return diff_clusters


def make_interpretation(df: pd.DataFrame, continuous: list, categorical: list, dependent_clusters: bool) -> (dict, dict):
    """
    Identifies clusters between which there are statistically significant differences,
    and the features that bring them into each of the clusters.

    """
    p_values = calculate_p_values(df, continuous, categorical, dependent_clusters)

    clusters = {el: [] for el in set(df["Clusters"])}
    differences = []
    for feature in p_values.keys():
        for key in p_values[feature].keys():
            # where key is (feature, cluster0, cluster1) or feature
            p_value = p_values[feature][key]
            if p_value < 0.05:
                differences.append(key)
                if feature not in clusters[key[1]]:
                    clusters[key[1]].append(feature)
                if feature not in clusters[key[2]]:
                    clusters[key[2]].append(feature)

    return clusters, different_clusters(differences)


class ClusteringInterpretation(object):

    def __init__(self, data: pd.DataFrame, clusters: str):
        if data.size <= 1:
            raise ValueError("`data` input should have multiple elements.")

        self._df = data.rename({clusters: "Clusters"}, axis=1)

        col_at_end = ["Clusters"]
        self._df = self._df[[c for c in self._df if c not in col_at_end] + [c for c in col_at_end if c in self._df]]

        self._continuous, self._categorical = [], []
        self._significant_features, self._differences = {}, {}
        self._dependent_clusters = False

    def set_continuous_and_categorical(self, continuous: list, categorical: list):
        self._continuous, self._categorical = continuous, categorical

    def set_dependent_clusters(self, dependent_clusters: bool):
        self._dependent_clusters = dependent_clusters

    def get_differences(self) -> dict:
        if not self._differences:
            self._significant_features, self._differences = make_interpretation(self._df, self._continuous,
                                                                                self._categorical, self._dependent_clusters)
        return self._differences

    def get_significant_features(self) -> dict:
        if not self._significant_features:
            self._significant_features, self._differences = make_interpretation(self._df, self._continuous,
                                                                                self._categorical, self._dependent_clusters)
        return self._significant_features
