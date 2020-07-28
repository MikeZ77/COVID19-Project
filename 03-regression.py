from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = 'output/'
IMAGE_PATH = 'images/linear_regression/'
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
        plt.figure(figsize=(24, 8))
        plt.plot(covid_df[variable], covid_df['death'], 'b.', alpha=0.5)
        plt.savefig(IMAGE_PATH + variable + ".png")


def perform_regression(covid_df):
    reg = stats.linregress(covid_df['pop_density'], covid_df['death'])
    print(reg.pvalue)
    print(reg.rvalue**2)


def check_risidual_normality():
    pass


def main():
    """
    The assumptions for OLS are:
    (1) The sample is representative of the population.
    (2) The relationship between the variables is linear.
    (3) The residuals are normally distributed and iid.
    """

    covid_df = read_data()
    covid_df = select_variables(covid_df)
    check_linearity(covid_df)
    perform_regression(covid_df)


if __name__ == '__main__':
    main()
