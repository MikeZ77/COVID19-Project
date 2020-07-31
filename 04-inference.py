import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
import matplotlib.pyplot as plt
import itertools

DATA_PATH = 'output/'
FILE_1 = 'daily_cases_wtih_race.csv'
FILE_2 = 'total_cases_with_socioecon.csv'
IMAGE_PATH_HIST = 'images/deaths_race/'
pd.set_option('display.max_rows', 300)


def read_data():
    covid_df = pd.read_csv(DATA_PATH + FILE_1, index_col=[0])
    socioecon_df = pd.read_csv(DATA_PATH + FILE_2)
    covid_df['date'] = pd.to_datetime(covid_df['date'].astype(str), format='%Y%m%d')
    return covid_df, socioecon_df


def reformat_headers(socioecon_df):
    columns_to_rename = {
        'e_total_pop_race_white': 'total_white',
        'e_total_pop_race_black': 'total_black',
        'e_total_pop_race_hispanic': 'total_hispanic',
        'e_total_pop_race_asian': 'total_asian',
        'e_total_pop_race_native_pop': 'total_aian',
        'e_total_pop_race_pacific_islander': 'total_nhpi'
    }
    socioecon_df = socioecon_df.rename(columns=columns_to_rename)
    socioecon_df = socioecon_df[
        ['state',
         'total_white',
         'total_black',
         'total_hispanic',
         'total_asian',
         'total_aian',
         'total_nhpi',
         ]
    ]
    return socioecon_df


def clean_for_race_inference(covid_df, socioecon_df):

    covid_df = covid_df[
        ['date',
         'state',
         'deaths_white',
         'deaths_black',
         'deaths_ethnicity_hispanic',
         'deaths_asian',
         'deaths_aian',
         'deaths_nhpi',
         ]
    ]
    population_mapping = {
        'deaths_white': 'total_white',
        'deaths_black': 'total_black',
        'deaths_ethnicity_hispanic': 'total_hispanic',
        'deaths_asian': 'total_asian',
        'deaths_aian': 'total_aian',
        'deaths_nhpi': 'total_nhpi',
    }

    dates_df = covid_df[['date']].drop_duplicates()
    # Keep only those States that have data for all races on a given day
    covid_df = covid_df.dropna().reset_index(drop=True)
    # print(covid_df)
    """
    There are 8 States which have considtantly reported Race based data: AK, CA, CO, GA, IL, LA, MN, NC, WA
    since 2020-04-26. That is our sample. Also, we are more concerned about the deaths as a
    percentage of the popultion.
    """
    covid_df = covid_df[covid_df['date'] >= '2020-04-26']
    states = ['AK', 'CA', 'CO', 'GA', 'IL', 'LA', 'MN', 'NC', 'WA']
    daily_covid = pd.DataFrame()

    for state in states:
        # Take the diff of each state to get the daily deaths
        covid_state = covid_df[covid_df['state'] == state].drop(columns=['date', 'state']).diff(-1)[:-1]

        # Divide daily deaths by the total population of the state
        for col_name in list(covid_state.columns):
            covid_state[col_name] = covid_state[col_name] / \
                socioecon_df.loc[socioecon_df.index[socioecon_df['state'] == state][0], population_mapping[col_name]]

        # checks that the index are matching: covid_state['state1'] = state
        daily_covid = daily_covid.append(covid_state)

    covid_df = covid_df[['date', 'state']]
    # Merge the dataframes by their index which has not changed
    covid_df = pd.merge(covid_df, daily_covid, left_index=True, right_index=True).reset_index(drop=True)

    return covid_df


def check_inference_conditions(covid_df):
    """
    ANOVA has three conditions, but two we need to check:
    (1) Groups are normally distributed
    (2) Groups have equal variance.
    """
    removed_nan_col = pd.DataFrame()
    removed_nan_col_compare = pd.DataFrame()
    covid_df = covid_df.drop(columns=['date', 'state'])

    for col_name in list(covid_df.columns):
        removed_nan_col['temp_col'] = covid_df[col_name].dropna().reset_index(drop=True)
        print(col_name)
        print("Normal pvalue: " + str(stats.normaltest(removed_nan_col['temp_col']).pvalue))
        print()

        removed_nan_col = pd.DataFrame()
        removed_nan_col_compare = pd.DataFrame()
    # Note, if sample sizes are equal, the t-test/ANOVA is robust enough to deal with it (Heteroskedacity)
    for subset in itertools.combinations(list(covid_df.columns), 2):

        removed_nan_col['temp_col'] = covid_df[subset[0]].dropna().reset_index(drop=True)
        removed_nan_col_compare['temp_col'] = covid_df[subset[1]].dropna().reset_index(drop=True)

        print(subset[0] + " - " + subset[1])
        print(stats.levene(removed_nan_col['temp_col'], removed_nan_col_compare['temp_col']).pvalue)
        print()

        removed_nan_col = pd.DataFrame()
        removed_nan_col_compare = pd.DataFrame()


def save_sample_population_distribution(covid_df):
    covid_df = covid_df.drop(columns=['date', 'state'])
    for col_name in list(covid_df.columns):
        plt.figure(figsize=(24, 8))
        plt.hist(covid_df[col_name].dropna().reset_index(drop=True), bins=len(covid_df.index))
        plt.title("Histogram - " + col_name)
        plt.xlabel(col_name)
        plt.ylabel("Deaths per State Population")
        plt.savefig(IMAGE_PATH_HIST + col_name + ".png")
        plt.close()


def sample_population_adjustments(covid_df):
    covid_df = covid_df.replace(0, np.NaN)
    # Lets count the number of non-nan values so we can get the new sample numbers
    print(covid_df['deaths_white'].count())
    print(covid_df['deaths_black'].count())
    print(covid_df['deaths_ethnicity_hispanic'].count())
    print(covid_df['deaths_asian'].count())
    print(covid_df['deaths_aian'].count())
    print(covid_df['deaths_nhpi'].count())
    """
    You can see that deaths_aian and deaths_nhpi have many days where 0 deaths are recorded and
    is beyond the 1.5 ratio of deaths_white, deaths_black, deaths_ethnicity_hispanic and deaths_asian.
    Much of these data points may be legitimate, since these populations are so small. That being said,
    the data is very far from being normal, and cannot be transformed. For this reason, they should be
    removed from the analysis.
    """
    covid_df = covid_df.drop(columns=['deaths_aian', 'deaths_nhpi'])
    return covid_df


def transform_data(covid_df):
    """
    The data appears to be skewed right skewed. try to transform the data. 
    """
    scale_df = covid_df.drop(columns=['date', 'state'])
    # Normalize the data since we cannot log or sqrt negative values
    scale_df = (scale_df - scale_df.min()) / (scale_df.max() - scale_df.min())

    scale_df = np.log(scale_df + 1)

    covid_df[['deaths_white', 'deaths_black', 'deaths_ethnicity_hispanic', 'deaths_asian']] = scale_df
    print(covid_df)
    return covid_df


def main():
    covid_df, socioecon_df = read_data()
    socioecon_df = reformat_headers(socioecon_df)
    covid_df = clean_for_race_inference(covid_df, socioecon_df)
    check_inference_conditions(covid_df)
    save_sample_population_distribution(covid_df)

    """
    FINDINGS: Clearly, the data is not normal. Some of the combinations of variables have equal variance.
    Looking at the histograms, this confirms this, and that the data is heavily skewed to the right. This is
    because on many days, 0 deaths are reported. There also appears to be corrections in the data on some days
    (i.e. negative values). 

    Note: In order to do a t-test or ANOVA, the sample sizes must be approximately equal. At the worst, this must
    be a ratio of 1.5/1 between the two samples for unequal variance. If the variance is equal, unequal sample size
    is irrelvant. I think it makes sense to leave the negative adjustments, since they are relatively few.
    """
    covid_df = sample_population_adjustments(covid_df)
    check_inference_conditions(covid_df)
    save_sample_population_distribution(covid_df)

    """
    The data is still not normal. Lets see if transmorming it can help.
    """

    covid_df = transform_data(covid_df)
    check_inference_conditions(covid_df)
    save_sample_population_distribution(covid_df)

    """
    We now have a distribution that looks much closer to normal, and has equal variance. It is still skewed to
    the right though. An obvious problem is the 0 deaths that we removed. It looks like a small number of days
    do have 0 deaths, and the remaining are just days that are not reported.
    """


if __name__ == '__main__':
    main()
