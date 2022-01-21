from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats import contingency_tables
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats import proportion
from itertools import combinations
from scipy import stats

import pandas as pd
import numpy as np


def ttest_ind(df: pd.DataFrame, clusters: list = None, features: list = None) -> dict:
    """
        Calculate the T-test for the means of *two independent* samples of scores.
        This is a two-sided test for the null hypothesis that 2 independent samples
        have identical average (expected) values. This test assumes that the
        populations have identical variances by default.

        Parameters
        ----------
        df: DataFrame
        features: list
        clusters: list

        Returns
        -------
        p-values : dict
            key: tuple(feature, cluster_i, cluster_j)
            value: the two-tailed p-value

        Notes
        -----
        We can use this test, if we observe two independent samples from
        the same or different population, e.g. exam scores of boys and
        girls or of two ethnic groups. The test measures whether the
        average (expected) value differs significantly across samples. If
        we observe a large p-value, for example larger than 0.05 or 0.1,
        then we cannot reject the null hypothesis of identical average scores.
        If the p-value is smaller than the threshold, e.g. 1%, 5% or 10%,
        then we reject the null hypothesis of equal averages.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/T-test#Independent_two-sample_t-test
        .. [2] https://en.wikipedia.org/wiki/Welch%27s_t-test
    """

    if features is None:
        features = df.columns[:-1].copy()  # get the names of all features except clusters name

    if clusters is None:
        clusters = df["Clusters"].copy()
        clusters = clusters.dropna().unique().tolist()  # list of clusters [3,1,4,2 ...]

    clusters.sort()

    p_values = {}

    for feature in features:
        sub_df = df[[feature, "Clusters"]].copy()  # feature values and their corresponding cluster values
        sub_df = sub_df.dropna()

        clusters_values = []

        for cluster in clusters:
            # index of clusters_values == index of clusters
            clusters_values.append(sub_df[sub_df["Clusters"] == cluster][feature].values.tolist())

        clusters_number = len(clusters_values)

        for i in range(clusters_number):
            for j in range(i + 1, clusters_number):
                try:
                    p_value = stats.ttest_ind(clusters_values[i], clusters_values[j])[1]
                    p_values[(feature, clusters[i], clusters[j])] = p_value
                except Exception as e:
                    p_values[(feature, clusters[i], clusters[j])] = 1.0

    return p_values


def levene(df: pd.DataFrame, clusters: list = None, features: list = None) -> dict:
    """
        Perform Levene test for equal variances.
        The Levene test tests the null hypothesis that all input samples are from
        populations with equal variances. Levene’s test is an alternative to Bartlett’s test
        bartlett in the case where there are significant deviations from normality.

        Parameters
        ----------
        df: DataFrame
        features: list
        clusters: list

        Returns
        -------
        p-values : dict
            key: tuple(feature, cluster_i, cluster_j)
            value: the two-tailed p-value

        Notes
        -----
        Three variations of Levene’s test are possible.
        The possibilities and their recommended usages are:
            ‘median’ : Recommended for skewed (non-normal) distributions>
            ‘mean’ : Recommended for symmetric, moderate-tailed distributions.
            trimmed’ : Recommended for heavy-tailed distributions.
        The test version using the mean was proposed in the original article of Levene
        while the median and trimmed mean have been studied by Brown and Forsythe,
        sometimes also referred to as Brown-Forsythe test.

        References
        ----------
        .. [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35a.htm
        .. [2] Levene, H. (1960). In Contributions to Probability and Statistics:
               Essays in Honor of Harold Hotelling, I. Olkin et al. eds.,
               Stanford University Press, pp. 278-292.
        .. [3] Brown, M. B. and Forsythe, A. B. (1974), Journal of the
               American Statistical Association, 69, 364-367
    """

    if features is None:
        features = df.columns[:-1].copy()  # get the names of all features except clusters name

    if clusters is None:
        clusters = df["Clusters"].copy()
        clusters = clusters.dropna().unique().tolist()  # list of clusters [3,1,4,2 ...]

    clusters.sort()

    p_values = {}

    for feature in features:
        sub_df = df[[feature, "Clusters"]].copy()  # feature values and their corresponding cluster values
        sub_df = sub_df.dropna()

        clusters_values = []

        for cluster in clusters:
            # index of clusters_values == index of clusters
            clusters_values.append(sub_df[sub_df["Clusters"] == cluster][feature].values.tolist())

        clusters_number = len(clusters_values)

        for i in range(clusters_number):
            for j in range(i + 1, clusters_number):
                try:
                    p_value = stats.levene(clusters_values[i], clusters_values[j])[1]
                    p_values[(feature, clusters[i], clusters[j])] = p_value
                except Exception as e:
                    p_values[(feature, clusters[i], clusters[j])] = 1.0

    return p_values


def f_oneway(df: pd.DataFrame, clusters: list = None, features: list = None) -> dict:
    """
        Perform one-way ANOVA.
        The one-way ANOVA tests the null hypothesis that two or more groups have
        the same population mean.  The test is applied to samples from two or
        more groups, possibly with differing sizes.

        Parameters
        ----------
        df: DataFrame
        features: list
        clusters: list

        Returns
        -------
        p-values : dict
            key: feature
            value: the associated p-value from the F distribution

        Notes
        -----
        The ANOVA test has important assumptions that must be satisfied in order
        for the associated p-value to be valid.
            1. The samples are independent.
            2. Each sample is from a normally distributed population.
            3. The population standard deviations of the groups are all equal.  This
               property is known as homoscedasticity.

        References
        ----------
        .. [1] R. Lowry, "Concepts and Applications of Inferential Statistics",
               Chapter 14, 2014, http://vassarstats.net/textbook/
        .. [2] G.W. Heiman, "Understanding research methods and statistics: An
               integrated introduction for psychology", Houghton, Mifflin and
               Company, 2001.
        .. [3] G.H. McDonald, "Handbook of Biological Statistics", One-way ANOVA.
               http://www.biostathandbook.com/onewayanova.html
    """

    if features is None:
        features = df.columns[:-1].copy()  # get the names of all features except clusters name

    if clusters is None:
        clusters = df["Clusters"].copy()
        clusters = clusters.dropna().unique().tolist()  # list of clusters [3,1,4,2 ...]

    clusters.sort()

    p_values = {}

    for feature in features:
        sub_df = df[[feature, "Clusters"]].copy()  # feature values and their corresponding cluster values
        sub_df = sub_df.dropna()

        clusters_values = []

        for cluster in clusters:
            # index of clusters_values == index of clusters
            clusters_values.append(sub_df[sub_df["Clusters"] == cluster][feature].values.tolist())

        try:
            p_value = stats.f_oneway(*(x for x in clusters_values))[1]
        except Exception as e:
            p_value = 1.0

        p_values[feature] = p_value

    return p_values


def tukeys_test(df: pd.DataFrame, clusters: list = None, features: list = None) -> dict:
    """
        Tukey’s range test to compare means of all pairs of groups

        Parameters
        ----------
        df: DataFrame
        features: list
        clusters: list

        Returns
        -------
        p-values : dict
            key: feature
            value: a results class containing relevant data and some post-hoc calculations

        Notes
        -----
        1. The observations being tested are independent within and among the groups.
        2. The groups associated with each mean in the test are normally distributed.
        3. There is equal within-group variance across the groups associated with each
           mean in the test (homogeneity of variance).

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Tukey%27s_range_test
    """

    if features is None:
        features = df.columns[:-1].copy()  # get the names of all features except clusters name

    if clusters is None:
        clusters = df["Clusters"].copy()
        clusters = clusters.dropna().unique().tolist()  # list of clusters [3,1,4,2 ...]

    clusters.sort()

    p_values = {}
    pairs = list(combinations(clusters, 2))

    for feature in features:
        sub_df = df[[feature, "Clusters"]].copy()
        sub_df = sub_df.dropna()
        table = pairwise_tukeyhsd(sub_df[sub_df["Clusters"].isin(clusters)][feature],
                                  sub_df[sub_df["Clusters"].isin(clusters)]["Clusters"])

        for i in range(len(pairs)):
            p_value = table.__dict__["pvalues"][i]
            key = (feature, pairs[i][0], pairs[i][1])
            p_values[key] = p_value

    return p_values


def ttest_rel(df: pd.DataFrame, clusters: list = None, features: list = None) -> dict:
    """
        Calculate the t-test on TWO RELATED samples of scores, clusters[i] and clusters[j].
        This is a two-sided test for the null hypothesis that 2 related or
        repeated samples have identical average (expected) values.

        Parameters
        ----------
        df: DataFrame
        features: list
        clusters: list

        Returns
        -------
        p-values : dict
           key: tuple(feature, cluster_i, cluster_j)
           value: the two-sided p-value

        Notes
        -----
        Examples for use are scores of the same set of student in
        different exams, or repeated sampling from the same units. The
        test measures whether the average score differs significantly
        across samples (e.g. exams). If we observe a large p-value, for
        example greater than 0.05 or 0.1 then we cannot reject the null
        hypothesis of identical average scores. If the p-value is smaller
        than the threshold, e.g. 1%, 5% or 10%, then we reject the null
        hypothesis of equal averages. Small p-values are associated with
        large t-statistics.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/T-test#Dependent_t-test_for_paired_samples
    """

    if features is None:
        features = df.columns[:-1].copy()  # get the names of all features except clusters name

    if clusters is None:
        clusters = df["Clusters"].copy()
        clusters = clusters.dropna().unique().tolist()  # list of clusters [3,1,4,2 ...]

    clusters.sort()

    p_values = {}

    for feature in features:
        sub_df = df[[feature, "Clusters"]].copy()  # feature values and their corresponding cluster values
        sub_df = sub_df.dropna()

        clusters_values = []

        for cluster in clusters:
            # index of clusters_values == index of clusters
            clusters_values.append(sub_df[sub_df["Clusters"] == cluster][feature].values.tolist())

        clusters_number = len(clusters_values)

        for i in range(clusters_number):
            for j in range(i + 1, clusters_number):
                try:
                    p_value = stats.ttest_rel(clusters_values[i], clusters_values[j])[1]
                    p_values[(feature, clusters[i], clusters[j])] = p_value
                except Exception as e:
                    p_values[(feature, clusters[i], clusters[j])] = 1.0

    return p_values


def anovaRM(df: pd.DataFrame, depvar: str, subject: str, within: list = None) -> pd.DataFrame:
    """
        Repeated measures Anova using least squares regression
        The full model regression residual sum of squares is used to compare
        with the reduced model for calculating the within-subject effect sum of squares.
        Currently, only fully balanced within-subject designs are supported. Calculation of
        between-subject effects and corrections for violation of sphericity are not yet implemented.

        Parameters
        ----------
        df: DataFrame
        depvar: str
             The dependent variable in data
        subject: str
             Specify the subject id
        within: list(str)
             The within-subject factors

        Returns
        ----------
        pd.DataFrame

        Notes
        ----------
        This implementation currently only supports fully balanced designs.
        If the data contain more than one observation per subject and cell of
        the design, these observations need to be aggregated into a single
        observation before the Anova is calculated, either manually or by passing
        an aggregation function via the aggregate_func keyword argument.
        Note that if the input data set was not balanced before performing
        the aggregation, the implied heteroscedasticity of the data is ignored.

        References
        ----------
        .. [1] Rutherford, Andrew. Anova and ANCOVA: a GLM approach. John Wiley & Sons, 2011.
    """

    anova_result = AnovaRM(data=df, depvar=depvar, subject=subject, within=within).fit()
    anova_result = anova_result.summary()
    anova_result = anova_result.tables[0]

    return anova_result


def z_test(df: pd.DataFrame, clusters: list = None, features: list = None, val: float = 0, alt: str = "two-sided") -> dict:
    """
        Test for mean based on normal distribution, one or two samples
        In the case of two samples, the samples are assumed to be independent.

        Parameters
        ----------
        df: DataFrame
        features: list
        clusters: list
        val: float
             In the one sample case, value is the mean of x1 under the Null hypothesis.
             In the two sample case, value is the difference between mean of
             x1 and mean of x2 under the Null hypothesis.
             The test statistic is x1_mean - x2_mean - value.
        alt: str
             The alternative hypothesis, H1, has to be one of the following
             ‘two-sided’: H1: difference in means not equal to value (default)
             ‘larger’ : H1: difference in means larger than value
             ‘smaller’ : H1: difference in means smaller than value

        Returns
        ----------
        p-values : dict
           key: tuple(feature, cluster_i, cluster_j)
           value: p-value of the t-test.

        Notes
        ----------
        1. Nuisance parameters should be known, or estimated with high accuracy
           (an example of a nuisance parameter would be the standard deviation in a one-sample location test).
           Z-tests focus on a single parameter, and treat all other unknown parameters as being fixed at their
           true values. In practice, due to Slutsky's theorem, "plugging in" consistent estimates of nuisance
           parameters can be justified. However if the sample size is not large enough for these estimates to be
           reasonably accurate, the Z-test may not perform well.
        2. The test statistic should follow a normal distribution. Generally, one appeals to the central limit theorem
           to justify assuming that a test statistic varies normally. There is a great deal of statistical research on
           the question of when a test statistic varies approximately normally. If the variation of the test statistic
           is strongly non-normal, a Z-test should not be used.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Z-test
    """

    if features is None:
        features = df.columns[:-1].copy()  # get the names of all features except clusters name

    if clusters is None:
        clusters = df["Clusters"].copy()
        clusters = clusters.dropna().unique().tolist()  # list of clusters [3,1,4,2 ...]

    clusters.sort()

    p_values = {}

    for feature in features:
        sub_df = df[[feature, "Clusters"]].copy()  # feature values and their corresponding cluster values
        sub_df = sub_df.dropna()

        clusters_values = []

        for cluster in clusters:
            # index of clusters_values == index of clusters
            clusters_values.append(sub_df[sub_df["Clusters"] == cluster][feature].values.tolist())

        clusters_number = len(clusters_values)

        if clusters_number > 1:
            for i in range(clusters_number):
                for j in range(i + 1, clusters_number):
                    try:
                        p_value = ztest(clusters_values[i], clusters_values[j], value=val, alternative=alt)[1]
                        p_values[(feature, clusters[i], clusters[j])] = p_value
                    except Exception as e:
                        p_values[(feature, clusters[i], clusters[j])] = 1.0
        elif clusters_number == 1:
            try:
                p_value = ztest(clusters_values[0], value=val, alternative=alt)[1]
            except Exception as e:
                p_value = 1.0
            p_values[feature] = p_value

    return p_values


def mannwhitneyu(df: pd.DataFrame, clusters: list = None, features: list = None, alt: str = "two-sided") -> dict:
    """
        Compute the Mann-Whitney rank test on samples clusters[i] and clusters[j].

        Parameters
        ----------
        df: DataFrame
        features: list
        clusters: list
        alt: str
            Defines the alternative hypothesis. Default is ‘two-sided’.
            Let F(u) and G(u) be the cumulative distribution functions of the distributions
            underlying x and y, respectively. Then the following alternative hypotheses are available:
           ‘two-sided’: the distributions are not equal, i.e. F(u) ≠ G(u) for at least one u.
           ‘less’: the distribution underlying x is stochastically less than the distribution
            underlying y, i.e. F(u) > G(u) for all u.
           ‘greater’: the distribution underlying x is stochastically greater than the distribution
            underlying y, i.e. F(u) < G(u) for all u.

        Returns
        -------
        p-values : dict
            key: tuple(feature, cluster_i, cluster_j)
            value: p-value assuming an asymptotic normal distribution. One-sided or
                   two-sided, depending on the choice of `alternative`.

         Notes
         -----
         Use only when the number of observation in each sample is > 20 and
         you have 2 independent samples of ranks. Mann-Whitney U is
         significant if the u-obtained is LESS THAN or equal to the critical
         value of U.
         This test corrects for ties and by default uses a continuity correction.

         References
         ----------
         .. [1] https://en.wikipedia.org/wiki/Mann-Whitney_U_test
         .. [2] H.B. Mann and D.R. Whitney, "On a Test of Whether one of Two Random
                Variables is Stochastically Larger than the Other," The Annals of
                Mathematical Statistics, vol. 18, no. 1, pp. 50-60, 1947.
    """

    if features is None:
        features = df.columns[:-1].copy()  # get the names of all features except clusters name

    if clusters is None:
        clusters = df["Clusters"].copy()
        clusters = clusters.dropna().unique().tolist()  # list of clusters [3,1,4,2 ...]

    clusters.sort()

    p_values = {}

    for feature in features:
        sub_df = df[[feature, "Clusters"]].copy()  # feature values and their corresponding cluster values
        sub_df = sub_df.dropna()

        clusters_values = []

        for cluster in clusters:
            # index of clusters_values == index of clusters
            clusters_values.append(sub_df[sub_df["Clusters"] == cluster][feature].values.tolist())

        clusters_number = len(clusters_values)

        for i in range(clusters_number):
            for j in range(i + 1, clusters_number):
                try:
                    p_value = stats.mannwhitneyu(clusters_values[i], clusters_values[j], alternative=alt)[1]
                    p_values[(feature, clusters[i], clusters[j])] = p_value
                except Exception as e:
                    p_values[(feature, clusters[i], clusters[j])] = 1.0

    return p_values


def kruskal_wallis_test(df: pd.DataFrame, clusters: list = None, features: list = None) -> dict:
    """
        Compute the Kruskal-Wallis H-test for independent samples.
        The Kruskal-Wallis H-test tests the null hypothesis that the population
        median of all of the groups are equal.  It is a non-parametric version of
        ANOVA.  The test works on 2 or more independent samples, which may have
        different sizes.  Note that rejecting the null hypothesis does not
        indicate which of the groups differs.  Post hoc comparisons between
        groups are required to determine which groups are different.

        Parameters
        ----------
        df: DataFrame
        features: list
        clusters: list

        Returns
        -------
        p-values : dict
            key: tuple(feature, cluster_i, cluster_j)
            value: The p-value for the test using the assumption that H has a chi
                   square distribution. The p-value returned is the survival function of
                   the chi square distribution evaluated at H.

        Notes
        -----
        Due to the assumption that H has a chi square distribution, the number
        of samples in each group must not be too small.  A typical rule is
        that each sample must have at least 5 measurements.

        References
        ----------
        .. [1] W. H. Kruskal & W. W. Wallis, "Use of Ranks in
               One-Criterion Variance Analysis", Journal of the American Statistical
               Association, Vol. 47, Issue 260, pp. 583-621, 1952.
        .. [2] https://en.wikipedia.org/wiki/Kruskal-Wallis_one-way_analysis_of_variance
    """

    if features is None:
        features = df.columns[:-1].copy()  # get the names of all features except clusters name

    if clusters is None:
        clusters = df["Clusters"].copy()
        clusters = clusters.dropna().unique().tolist()  # list of clusters [3,1,4,2 ...]

    clusters.sort()

    p_values = {}

    for feature in features:
        sub_df = df[[feature, "Clusters"]].copy()  # feature values and their corresponding cluster values
        sub_df = sub_df.dropna()

        clusters_values = []

        for cluster in clusters:
            # index of clusters_values == index of clusters
            clusters_values.append(sub_df[sub_df["Clusters"] == cluster][feature].values.tolist())

        clusters_number = len(clusters_values)

        for i in range(clusters_number):
            for j in range(i + 1, clusters_number):
                try:
                    p_value = stats.kruskal(clusters_values[i], clusters_values[j])[1]
                    p_values[(feature, clusters[i], clusters[j])] = p_value
                except Exception as e:
                    p_values[(feature, clusters[i], clusters[j])] = 1.0
    return p_values


def wilcoxon(df: pd.DataFrame, clusters: list = None, features: list = None, val: float = None, alt="two-sided") -> dict:
    """
        Calculate the Wilcoxon signed-rank test.
        The Wilcoxon signed-rank test tests the null hypothesis that two
        related paired samples come from the same distribution. In particular,
        it tests whether the distribution of the differences clusters[i] - clusters[j] is symmetric
        about zero. It is a non-parametric version of the paired T-test.

        Parameters
        ----------
        df: DataFrame
        features: list
        clusters: list
        val: float
        alt: {"two-sided", "greater", "less"}, optional
             The alternative hypothesis to be tested, see Notes. Default is
             "two-sided".

        Returns
        -------
        p-values : dict
            key: tuple(feature, cluster_i, cluster_j)
            value: the p-value for the test depending on ``alternative``.

        References
        ----------
        .. [1] Pratt, J.W., Remarks on Zeros and Ties in the Wilcoxon Signed Rank Procedures,
               Journal of the American Statistical Association,
               Vol. 54, 1959, pp. 655-667. DOI:10.1080/01621459.1959.10501526
        .. [2] https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    """

    if features is None:
        features = df.columns[:-1].copy()  # get the names of all features except clusters name

    if clusters is None:
        clusters = df["Clusters"].copy()
        clusters = clusters.dropna().unique().tolist()  # list of clusters [3,1,4,2 ...]

    clusters.sort()

    p_values = {}

    for feature in features:
        sub_df = df[[feature, "Clusters"]].copy()  # feature values and their corresponding cluster values
        sub_df = sub_df.dropna()

        clusters_values = []

        for cluster in clusters:
            # index of clusters_values == index of clusters
            clusters_values.append(sub_df[sub_df["Clusters"] == cluster][feature].values.tolist())

        clusters_number = len(clusters_values)

        if clusters_number > 1:
            for i in range(clusters_number):
                for j in range(i + 1, clusters_number):
                    try:
                        p_value = stats.wilcoxon(clusters_values[i], clusters_values[j], alternative=alt)[1]
                        p_values[(feature, clusters[i], clusters[j])] = p_value
                    except Exception as e:
                        p_values[(feature, clusters[i], clusters[j])] = 1.0
        elif clusters_number == 1:
            y = [val]*len(clusters_values[0])
            try:
                p_value = stats.wilcoxon(clusters_values[0], y, alternative=alt)[1]
            except Exception as e:
                p_value = 1.0
            p_values[feature] = p_value

    return p_values


def friedmanchisquare(df: pd.DataFrame, clusters: list = None, features: list = None) -> dict:
    """
        Compute the Friedman test for repeated measurements.
        The Friedman test tests the null hypothesis that repeated measurements of
        the same individuals have the same distribution.  It is often used
        to test for consistency among measurements obtained in different ways.
        For example, if two measurement techniques are used on the same set of
        individuals, the Friedman test can be used to determine if the two
        measurement techniques are consistent.

        Parameters
        ----------
        df: DataFrame
        features: list
        clusters: list

        Returns
        -------
        p-values : dict
            key: feature
            value: the associated p-value assuming that the test statistic has a chi
                   squared distribution.

        Notes
        -----
        Due to the assumption that the test statistic has a chi squared
        distribution, the p-value is only reliable for n > 10 and more than
        6 repeated measurements.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Friedman_test
    """

    if features is None:
        features = df.columns[:-1].copy()  # get the names of all features except clusters name

    if clusters is None:
        clusters = df["Clusters"].copy()
        clusters = clusters.dropna().unique().tolist()  # list of clusters [3,1,4,2 ...]

    clusters.sort()

    p_values = {}

    for feature in features:
        sub_df = df[[feature, "Clusters"]].copy()  # feature values and their corresponding cluster values
        sub_df = sub_df.dropna()

        clusters_values = []

        for cluster in clusters:
            # index of clusters_values == index of clusters
            clusters_values.append(sub_df[sub_df["Clusters"] == cluster][feature].values.tolist())

        try:
            p_value = stats.friedmanchisquare(*(x for x in clusters_values))[1]
        except Exception as e:
            p_value = 1.0
        p_values[feature] = p_value

    return p_values


def chi2(df: pd.DataFrame, clusters: list = None, features: list = None, alt: str = None) -> dict:
    """
        Chi-square test of independence of variables in a contingency table.

        Parameters
        ----------
        df: DataFrame
        features: list
        clusters: list
        alt: str
            By default, the statistic computed in this test is Pearson’s chi-squared statistic.
            alt allows a statistic from the Cressie-Read power divergence family to be used instead.

        Returns
        -------
        p-values : dict
            key: tuple(feature, clusters_i, clusters_j)
            value: the p-value of the test

        References
        ----------
        .. [1] "Contingency table",
                https://en.wikipedia.org/wiki/Contingency_table
        .. [2] "Pearson's chi-squared test",
                https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
        .. [3] Cressie, N. and Read, T. R. C., "Multinomial Goodness-of-Fit
               Tests", J. Royal Stat. Soc. Series B, Vol. 46, No. 3 (1984),
               pp. 440-464.
    """

    if features is None:
        features = df.columns[:-1].copy()  # get the names of all features except clusters name

    if clusters is None:
        clusters = df["Clusters"].copy()
        clusters = clusters.dropna().unique().tolist()  # list of clusters [3,1,4,2 ...]

    clusters.sort()

    p_values = {}

    for feature in features:
        sub_df = df[[feature, "Clusters"]].copy()  # feature values and their corresponding cluster values
        sub_df = sub_df.dropna()

        clusters_values = []
        min_size = 0  # set equal size for crosstab
        p_values = {}

        for cluster in clusters:
            # index of clusters_values == index of clusters
            clusters_values.append(sub_df[sub_df["Clusters"] == cluster][feature].values.tolist())
            if len(clusters_values[-1]) < min_size or min_size == 0:
                min_size = len(clusters_values[-1])

        clusters_number = len(clusters_values)

        for i in range(clusters_number):
            for j in range(i + 1, clusters_number):
                tab = pd.crosstab(np.array(clusters_values[i][:min_size]),
                                  np.array(clusters_values[j][:min_size]))
                try:
                    stat, p_value, dof, expected = stats.chi2_contingency(tab, lambda_=alt)
                    p_values[(feature, clusters[i], clusters[j])] = p_value
                except Exception as e:
                    p_values[(feature, clusters[i], clusters[j])] = 1.0

    return p_values


def mcnemar(df: pd.DataFrame, clusters: list = None, features: list = None) -> dict:
    """
        McNemar test of homogeneity.

        Parameters
        ----------
        df: DataFrame
        features: list
        clusters: list

        Returns
        -------
        p-values : dict
            key: tuple(feature, cluster_i, cluster_j)
            value: p-value of the null hypothesis of equal marginal distributions

        Notes
        ----------
        This is a special case of Cochran’s Q test, and of the homogeneity test.
        The results when the chisquare distribution is used are identical, except for continuity correction.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/McNemar%27s_test
    """

    if features is None:
        features = df.columns[:-1].copy()  # get the names of all features except clusters name

    if clusters is None:
        clusters = df["Clusters"].copy()
        clusters = clusters.dropna().unique().tolist()  # list of clusters [3,1,4,2 ...]

    clusters.sort()

    p_values = {}

    for feature in features:
        sub_df = df[[feature, "Clusters"]].copy()  # feature values and their corresponding cluster values
        sub_df = sub_df.dropna()

        clusters_values = []

        for cluster in clusters:
            # index of clusters_values == index of clusters
            clusters_values.append(sub_df[sub_df["Clusters"] == cluster][feature].values.tolist())

        clusters_number = len(clusters_values)

        for i in range(clusters_number):
            for j in range(i + 1, clusters_number):
                try:
                    table = [clusters_values[i], clusters_values[j]]
                    res = contingency_tables.mcnemar(table)
                    p_values[(feature, clusters[i], clusters[j])] = res.pvalue
                except Exception as e:
                    p_values[(feature, clusters[i], clusters[j])] = 1.0

    return p_values


def cochrans_q(df: pd.DataFrame, clusters: list = None, features: list = None) -> dict:
    """
        Cochran’s Q test for identical binomial proportions

        Parameters
        ----------
        df: DataFrame
        features: list
        clusters: list

        Returns
        -------
        p-values : dict
            key: feature
            value: p_value from the chisquare distribution.

        Notes
        ----------
        Cochran’s Q is a k-sample extension of the McNemar test.
        If there are only two groups, then Cochran’s Q test and the McNemar test are equivalent.
        The procedure tests that the probability of success is the same for every group.
        The alternative hypothesis is that at least two groups have a different probability of success.
        In Wikipedia terminology, rows are blocks and columns are treatments.
        The number of rows N, should be large for the chisquare distribution to be a good approximation.
        The Null hypothesis of the test is that all treatments have the same effect.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Cochran_test SAS Manual for NPAR TESTS
    """

    if features is None:
        features = df.columns[:-1].copy()  # get the names of all features except clusters name

    if clusters is None:
        clusters = df["Clusters"].copy()
        clusters = clusters.dropna().unique().tolist()  # list of clusters [3,1,4,2 ...]

    clusters.sort()

    p_values = {}

    for feature in features:
        sub_df = df[[feature, "Clusters"]].copy()  # feature values and their corresponding cluster values
        sub_df = sub_df.dropna()

        clusters_values = []

        for cluster in clusters:
            # index of clusters_values == index of clusters
            clusters_values.append(sub_df[sub_df["Clusters"] == cluster][feature].values.tolist())

        try:
            table = [*(x for x in clusters_values)]
            res = contingency_tables.cochrans_q(table)
            p_values[feature] = res.pvalue
        except Exception as e:
            p_values[feature] = 1.0

    return p_values


def proportions_ztest(df: pd.DataFrame, clusters: list = None, features: list = None,
                      val: float = None, alt: str = 'two-sided', prop_var: bool = False) -> dict:
    """
        Test for proportions based on normal (z) test

        Parameters
        ----------
        df: DataFrame
        features: list
        clusters: list
        val: float, array_like or None, optional
             This is the value of the null hypothesis equal to the proportion
             in the case of a one sample test. In the case of a two-sample test,
             the null hypothesis is that prop[0] - prop[1] = value, where prop is
             the proportion in the two samples. If not provided value = 0 and the null
             is prop[0] = prop[1]
        alt: str in [‘two-sided’, ‘smaller’, ‘larger’]
             The alternative hypothesis can be either two-sided or one of the one- sided tests,
             smaller means that the alternative hypothesis is prop < value and larger means prop > value.
             In the two sample test, smaller means that the alternative hypothesis is p1 < p2
             and larger means p1 > p2 where p1 is the proportion of the first sample and p2 of the second one.
        prop_var: False or float in (0, 1)
            If prop_var is false, then the variance of the proportion estimate is calculated based on the sample
            proportion. Alternatively, a proportion can be specified to calculate this variance. Common use case is to
            use the proportion under the Null hypothesis to specify the variance of the proportion estimate.

        Returns
        ----------
        p-values : dict
            key: feature
            value: p-value for the z-test

        Notes
        ----------
        This uses a simple normal test for proportions. It should be the same
        as running the mean z-test on the data encoded 1 for event and 0 for no event
        so that the sum corresponds to the count.
        In the one and two sample cases with two-sided alternative, this test produces
        the same p-value as proportions_chisquare, since the chisquare is the distribution
        of the square of a standard normal distribution.
    """

    if features is None:
        features = df.columns[:-1].copy()  # get the names of all features except clusters name

    if clusters is None:
        clusters = df["Clusters"].copy()
        clusters = clusters.dropna().unique().tolist()  # list of clusters [3,1,4,2 ...]

    clusters.sort()

    p_values = {}

    for feature in features:
        sub_df = df[[feature, "Clusters"]].copy()  # feature values and their corresponding cluster values
        sub_df = sub_df.dropna()

        count = []
        nobs = []

        for cluster in clusters:
            nobs.append(len(sub_df[sub_df["Clusters"] == cluster][feature]))
            count.append(len(sub_df[(sub_df["Clusters"] == cluster) & (sub_df[feature] == 1)][feature]))

        try:
            p_value = proportion.proportions_ztest(count, nobs, value=val, alternative=alt, prop_var=prop_var)[1]
            p_values[feature] = p_value
        except Exception as e:
            p_values[feature] = 1.0

    return p_values

