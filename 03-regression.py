from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd


DATA_PATH = 'output/'
IMAGE_PATH_REGRESSION = 'images/linear_regression/regression_plot/'
IMAGE_PATH_RISIDUALS = 'images/linear_regression/risiduals/'
FILE = 'total_cases_with_socioecon.csv'


def read_data():
    covid_df = pd.read_csv(DATA_PATH + FILE)
    return covid_df


def select_variables(covid_df):

    dependent_variable = ['death']

    independent_variables = [
        'pop_density',
        'cdc_svi_overall_ranking',
        'em_total_households_with_income_mean_earnings',
        'em_total_pop_median_age',
        'et_over_60_population',
        'em_over_60_pop_median_age',
        'e_total_housing_units_more_than_one_occupant_per_room',
        'e_total_pop_poverty_status_below_100_percent',
        'e_total_pop_over_60_responsible_for_grandchildren_live_together',
        'e_over_60_pop_employment_status_in_civilian_labor_force',
        'e_over_60_pop_poverty_status_below_100_percent',
    ]

    covid_df = covid_df[independent_variables + dependent_variable]
    return covid_df


def check_linearity(covid_df):
    dependent = list(covid_df.columns)
    dependent.remove('death')

    for variable in dependent:
        reg = stats.linregress(covid_df[variable], covid_df['death'])
        plt.figure(figsize=(24, 8))
        plt.plot(covid_df[variable], covid_df['death'], 'b.', alpha=0.5)
        plt.plot(covid_df[variable], reg.intercept + reg.slope * covid_df[variable], 'r', linewidth=2)
        plt.savefig(IMAGE_PATH_REGRESSION + variable + ".png")
        plt.close()


def output_regression_details(covid_df):
    """
    For analysis only. Prints out the statistical summary for the given regression.
    """
    data = pd.DataFrame({'y': covid_df['death'], 'x': covid_df['pop_density'], 'intercept': 1})
    results = sm.OLS(data['y'], data[['x', 'intercept']]).fit()
    print(results.summary())

def check_risidual_normality(covid_df):

    dependent = list(covid_df.columns)
    dependent.remove('death')

    for variable in dependent:
        reg = stats.linregress(covid_df[variable], covid_df['death'])
        residuals = covid_df['death'] - (reg.slope * covid_df[variable] + reg.intercept)
        # Check the pvalue for normality of the risidual
        # print(stats.normaltest(residuals).pvalue)

        plt.figure(figsize=(24, 8))
        plt.hist(residuals, bins=len(covid_df.index))
        plt.title("Histogram Risiduals - " + variable)
        plt.xlabel(variable)
        plt.ylabel("Deaths")
        plt.savefig(IMAGE_PATH_RISIDUALS + variable + ".png")
        plt.close()


def output_regression(covid_df):
    dependent = list(covid_df.columns)
    dependent.remove('death')

    for variable in dependent:
        reg = stats.linregress(covid_df[variable], covid_df['death'])
        print(variable)
        print("pvalue: " + str(reg.pvalue))
        print("R^2: " + str(reg.rvalue**2))
        print()


def output_non_linear_regression(covid_df):
    dependent = list(covid_df.columns)
    dependent.remove('death')

    cdc_svi = np.array(covid_df['cdc_svi_overall_ranking'].values)
    deaths = np.array(covid_df['death'].values)
    fit = np.polyfit(cdc_svi, deaths, 4)
    poly = np.poly1d(fit)

    plt.figure(figsize=(24, 8))
    xp = np.linspace(0, 1, 100)
    plt.plot(cdc_svi, deaths, 'b.', alpha=0.8)
    plt.plot(xp, poly(xp), '-')
    plt.savefig(IMAGE_PATH_REGRESSION + "cdc_svi_overall_ranking_" + "polynomial.png")
    plt.close()

    y_true = deaths
    y_pred = poly(cdc_svi)
    score = r2_score(y_true, y_pred)
    print("cdc_svi_overall_ranking: Polynomial Degree 4")
    print("R^2: " + str(score))

def main():

    covid_df = read_data()
    covid_df = select_variables(covid_df)
    """
    ------------------------------------------------------------------------------------------------------------------------
    PART 1: OLS
    ------------------------------------------------------------------------------------------------------------------------
    """
    check_linearity(covid_df)
    check_risidual_normality(covid_df)
    # output_regression_details(covid_df)
    output_regression(covid_df)

    """
    The assumptions for OLS are:
    (1) The sample is representative of the population.
    (2) The relationship between the variables is linear.
    (3) The residuals are normally distributed and iid.

    ASSUMPTIONS: 
    (1) The sample is the entire population data.

    (2) Upon inspecting the regression plots, the relation is clearly linear.

    (3) Out of interest, the risidual of all variables is clearly not normal.Obviously, transforming the the variable for a right
    skewed distribution with something like np.log() will have no impact on the risidual. Using the statistical summary
    of the regression, it can be shown that the distributions have high kurtosis and are skewed to the right. For example,
    Below is the output for populatioon density. It has significant kurtosis and skewness (right). For reference, a normal 
    distribution has a kurtosis of approximatley 3 and skewness of 0.

                                    OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      y   R-squared:                       0.575
    Model:                            OLS   Adj. R-squared:                  0.566
    Method:                 Least Squares   F-statistic:                     64.95
    Date:                Tue, 28 Jul 2020   Prob (F-statistic):           1.79e-10
    Time:                        12:13:48   Log-Likelihood:                -468.14
    No. Observations:                  50   AIC:                             940.3
    Df Residuals:                      48   BIC:                             944.1
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    x              7.0023      0.869      8.059      0.000       5.255       8.749
    intercept   -247.4002    536.970     -0.461      0.647   -1327.051     832.251
    ==============================================================================
    Omnibus:                       37.539   Durbin-Watson:                   2.051
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              173.188
    Skew:                           1.760   Prob(JB):                     2.47e-38
    Kurtosis:                      11.411   Cond. No.                         816.
    ==============================================================================

    However upon visually inspecting the histogram plot of the risidual, the majority of the variables appear 'normal enough'
    to get value out of the regression. First, n >= 40, and second, this is the population data and not the sample
    data.

    FINDINGS: 
    We find that many of the variables are linear (pvalue < 0.05), however, in general their variance is not well explained.
    The most significant finding is population density with a pvalue of 1.7934825125225636e-10 and an R^2 of 0.575.
    """
    """
    ------------------------------------------------------------------------------------------------------------------------
    PART 2: Non-Linear Regression
    ------------------------------------------------------------------------------------------------------------------------
    """
    output_non_linear_regression(covid_df)

    """
    Maybe one of the variables which clearly does not have a linear relationship can be modeled using using Polynomial 
    Regression. cdc_svi_overall_ranking is a metric provided by the CDC (Centre for Dieses Control and Prevention) based 
    on socio economic status, household composition and disability, minortity status and language, and housing type and
    transportation. Looking at the scatterplot, it seems like there is (a bit) of a pattern in the data.

    It turns out that the best resulted is given by a 4th degree polynomial, and for the most part it is no more promising
    tank the linear regression.
    """


if __name__ == '__main__':
    main()
