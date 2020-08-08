import pandas as pd
import numpy as np
import sys


OUTPUT_PATH = 'output/'
INPUT_PATH = 'raw_data/'


def standardize_headers(data_store_df):
    """
    Remove camelcase from daily_df.
    Lowercase all the df for consistancy.
    """
    daily_df = data_store_df['daily']

    updated_header = list()
    daily_header = list(daily_df.columns)
    for col in daily_header:
        for l in col:
            if l.isupper():
                col = col[:col.index(l)] + '_' + col[col.index(l):]
        updated_header.append(col)

    daily_df.columns = updated_header
    for df in data_store_df.values():
        df.columns = df.columns.str.lower()

    return data_store_df


def combine_daily_race(data_store_df):
    daily_df = data_store_df['daily']
    race_df = data_store_df['daily race']

    daily_race_df = daily_df.merge(race_df, how='left', on=['date', 'state'])

    return daily_race_df


def grouped_daily(data_store_df):
    daily_df = data_store_df['daily']

    daily_grouped_df = daily_df.groupby('state').agg('max')
    # DROP columns which are not cummulative
    non_cummulative_cols = [
        'hospitalized_currently',
        'in_icu_currently',
        'on_ventilator_currently',
        'death_increase',
        'hospitalized_increase',
        'positive_increase',
        'negative_increase',
        'total_test_results_increase',
        'date',
    ]
    daily_grouped_df.drop(non_cummulative_cols, axis=1, inplace=True)

    return daily_grouped_df


def grouped_daily_race(daily_race_df):
    daily_race_grouped_df = daily_race_df.groupby('state').agg('max')
    return daily_race_grouped_df


def replace_state_map(data_store_df):
    """
    Adds the mapping to the socioeconomic data. For example, 'ALABAMA' is replaced with 'AL'.
    This is to allow the two different sources to be merged in combine_socio_econ_and_total_race().
    """
    socio_econ_df = data_store_df['daily socioecon']
    socio_econ_over_60_df = data_store_df['daily socioecon over 60']
    state_mapping_df = data_store_df['state mapping']

    state_mapping_df = state_mapping_df.rename(columns={'state_upper': 'state'})

    mapped_df = socio_econ_df.merge(state_mapping_df, how='left', on='state')
    socio_econ_df = mapped_df.drop('state', axis=1)
    socio_econ_df = socio_econ_df.rename(columns={'state_short': 'state'})

    mapped_df = socio_econ_over_60_df.merge(state_mapping_df, how='left', on='state')
    socio_econ_over_60_df = mapped_df.drop('state', axis=1)
    socio_econ_over_60_df = socio_econ_over_60_df.rename(columns={'state_short': 'state'})

    data_store_df['daily socioecon'] = socio_econ_df
    data_store_df['daily socioecon over 60'] = socio_econ_over_60_df

    return data_store_df


def socio_econ_grouped_by_state(data_store_df):
    """
    Groups the socioeconomic data by state instead of by county. Note that 'cdc_svi_overall_ranking'
    and 'pop_density' use the mean instead of the sum.
    """
    socio_econ_df = data_store_df['daily socioecon']
    socio_econ_over_60_df = data_store_df['daily socioecon over 60']
    state_mapping_df = data_store_df['state mapping']

    socio_econ_headers = list(socio_econ_df.columns)
    socio_econ_headers.remove('cdc_svi_overall_ranking')
    socio_econ_headers.remove('em_total_pop_median_age')
    socio_econ_headers.remove('pop_density')
    socio_econ_headers.remove('em_total_housing_units_avg_household_size_owned')

    socio_econ_headers_to_average = [
        'cdc_svi_overall_ranking',
        'em_total_pop_median_age',
        'pop_density',
        'em_total_housing_units_avg_household_size_owned'
    ]

    socio_econ_avg_df = socio_econ_df.groupby('state')[socio_econ_headers_to_average].agg('mean')
    socio_econ_total_df = socio_econ_df.groupby('state')[socio_econ_headers].agg('sum')
    socio_econ_df = socio_econ_total_df.merge(socio_econ_avg_df, how='left', on='state')

    socio_econ_over_60_avg_df = socio_econ_over_60_df.groupby('state')['em_over_60_pop_median_age'].agg('mean')
    socio_econ_over_60_df = socio_econ_over_60_df.drop('em_over_60_pop_median_age', axis=1)
    socio_econ_over_60_df = socio_econ_over_60_df.groupby('state').agg('sum')
    socio_econ_over_60_df = socio_econ_over_60_df.merge(socio_econ_over_60_avg_df, how='left', on='state')

    data_store_df['daily socioecon'] = socio_econ_df
    data_store_df['daily socioecon over 60'] = socio_econ_over_60_df

    return data_store_df


def combine_socio_econ_and_total_race(total_race_df, data_store_df):
    """
    Merges the total dataframe with the total socioeconomic dataframe where 'total' is grouped
    by State.
    """
    socio_econ_df = data_store_df['daily socioecon']
    socio_econ_over_60_df = data_store_df['daily socioecon over 60']

    grouped_socio_econ_df = socio_econ_df.merge(socio_econ_over_60_df, how='left', on='state')
    socio_econ_total_race = grouped_socio_econ_df.merge(total_race_df, how='left', on='state')

    return socio_econ_total_race


def input_raw_data(data_store_df):

    data_store_df['daily'] = pd.read_csv(INPUT_PATH + sys.argv[1])
    data_store_df['daily race'] = pd.read_csv(INPUT_PATH + sys.argv[2])
    data_store_df['daily socioecon'] = pd.read_csv(INPUT_PATH + sys.argv[3])
    data_store_df['daily socioecon over 60'] = pd.read_csv(INPUT_PATH + sys.argv[4])
    data_store_df['state mapping'] = pd.read_csv(INPUT_PATH + sys.argv[5])

    # Feature descriptions: https://covidtracking.com/data-definitions
    data_store_df['daily'] = data_store_df['daily'][
        ['date',
         'state',
         'positive',
         'negative',
         'hospitalizedCurrently',
         'inIcuCurrently',
         'onVentilatorCurrently',
         'hospitalizedCumulative',
         'inIcuCumulative',
         'onVentilatorCumulative',
         'recovered',
         'deathIncrease',
         'death',
         'hospitalizedIncrease',
         'hospitalized',
         'positiveIncrease',
         'negativeIncrease',
         'positiveTestsViral',
         'negativeTestsViral',
         'totalTestResultsIncrease',
         'dataQualityGrade']
    ]
    """
    AIAN =  American Indian and Alaska Native
    NHPI =  Native Hawaiian and Pacific Islander
    Cases_LatinX includes Hispanic  
    """
    data_store_df['daily race'] = data_store_df['daily race'][
        ['Date',
         'State',
         'Cases_White',
         'Cases_Black',
         'Cases_LatinX',
         'Cases_Asian',
         'Cases_AIAN',
         'Cases_NHPI',
         'Cases_Multiracial',
         'Cases_Ethnicity_Hispanic',
         'Cases_Other',
         'Cases_Unknown',
         'Deaths_White',
         'Deaths_Black',
         'Deaths_LatinX',
         'Deaths_Asian',
         'Deaths_AIAN',
         'Deaths_NHPI',
         'Deaths_Multiracial',
         'Deaths_Ethnicity_Hispanic',
         'Deaths_Other',
         'Deaths_Unknown'
         ]
    ]
    """
    SOURCE: https://www.kaggle.com/jtourkis/us-county-level-acs-features-for-covid-analysis

    CDC_SVI_Overall_Ranking from https://svi.cdc.gov/. Social vulnerability refers to the 
    resilience of communities when confronted by external stresses on human health, stresses 
    such as natural or human-caused disasters, or disease outbreaks.CDC's Social 
    Vulnerability Index uses 15 U.S. census variables at tract level to help local officials 
    identify communities that may need support in preparing for hazards; or recovering from disaster.
    """
    data_store_df['daily socioecon'] = data_store_df['daily socioecon'][
        ['STATE',
         'Pop_Density',
         'CDC_SVI_Overall_Ranking',
         'ET_Total_Population',
         'EM_Total_Pop_Median_Age',
         'E_Total_Pop_RACE_White',
         'E_Total_Pop_RACE_Black',
         'E_Total_Pop_RACE_Native_Pop',
         'E_Total_Pop_RACE_Asian',
         'E_Total_Pop_RACE_Pacific_Islander',
         'E_Total_Pop_RACE_Other_Race',
         'E_Total_Pop_RACE_Two_or_More_Races',
         'E_Total_Pop_RACE_Hispanic',
         'E_Total_Pop_in_Households_Householder_Parent',
         'E_Total_Pop_Over_30_RESPONSIBLE_FOR_GRANDCHILDREN_Live_Together_and_Responsible',
         'E_Total_Households_TYPE_Nonfamily_Householder_Living_Alone',
         'E_Total_HOUSING_UNITS_More_Than_One_Occupant_Per_Room',
         'E_Total_Pop_POVERTY_STATUS_Below_100_Percent',
         'EM_Total_HOUSING_UNITS_Avg_Household_Size_Owned',
         'E_Total_Pop_Over_16_EMPLOYMENT_STATUS_In_Civilian_Labor_Force_Employed',
         'EM_Total_Households_WITH_INCOME_Mean_Earnings'
         ]
    ]

    """
    SOURCE: https://www.kaggle.com/jtourkis/us-county-level-acs-features-for-covid-analysis
    """

    data_store_df['daily socioecon over 60'] = data_store_df['daily socioecon over 60'][
        ['STATE',
         'ET_Over_60_Population',
         'EM_Over_60_Pop_Median_Age',
         'E_Over_60_RACE_White',
         'E_Over_60_RACE_Black',
         'E_Over_60_RACE_Native_Pop',
         'E_Over_60_RACE_Asian',
         'E_Over_60_RACE_Pacific_Islander',
         'E_Over_60_RACE_Other_Race',
         'E_Over_60_RACE__Two_or_More_Races',
         'E_Over_60_RACE_Hispanic',
         'E_Over_60_Households_TYPE_Family',
         'E_Total_Pop_Over_60_RESPONSIBLE_FOR_GRANDCHILDREN_Live_Together',
         'E_Pop_Over_60_DISABILITY_STATUS_Yes',
         'E_Over_60_Pop_EMPLOYMENT_STATUS_in_Civilian_Labor_Force',
         'E_Over_60_Pop_POVERTY_STATUS_Below_100_Percent',
         'EM_Over_60_HOUSING_UNITS_Avg_Household_Size_Owned',
         ]
    ]

    data_store_df['state mapping'] = data_store_df['state mapping'][
        [
            'state_upper',
            'state_short'
        ]
    ]

    return data_store_df


def output_cleaned_data(output):
    for df in output:
        df.to_csv(OUTPUT_PATH + df.name)


def main():

    #--------------------------------------------- CLEAN DATA -------------------------------------------------
    """
    Note: dataQualityGrade is a grade based on 5 factors: Reporting total, Testing total, Outcomes total, and 
          Demographic total. It is based on the amount of this data, and not on the quality of the data itself.
          Therefore, it makes the most sense to remove States ourselves that are not reporting the data or 
          interpolate if possible (as opposed to relying on the dataQualityGrade alone).  
    """
    data_store_df = {
        'daily': pd.DataFrame(),
        'daily race': pd.DataFrame(),
        'daily socioecon': pd.DataFrame(),
        'daily socioecon over 60': pd.DataFrame(),
        'state mapping': pd.DataFrame()}

    data_store_df = input_raw_data(data_store_df)
    data_store_df = standardize_headers(data_store_df)

    daily_race_df = combine_daily_race(data_store_df)
    total_race_df = grouped_daily_race(daily_race_df)
    total_df = grouped_daily(data_store_df)

    data_store_df = replace_state_map(data_store_df)
    data_store_df = socio_econ_grouped_by_state(data_store_df)

    socio_econ_total_race = combine_socio_econ_and_total_race(total_race_df, data_store_df)

    # REFERENCE: this is the current output selection
    daily_race_df.name = 'daily_cases_wtih_race.csv'
    total_race_df.name = 'total_cases_with_race.csv'
    socio_econ_total_race.name = 'total_cases_with_socioecon.csv'
    output = [daily_race_df, total_race_df, socio_econ_total_race]
    output_cleaned_data(output)


if __name__ == '__main__':
    main()
