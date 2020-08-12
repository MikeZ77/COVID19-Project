import glob
import numpy as np
import pandas as pd


def main():

    # Read in daily cases csv from another folder
    # https://stackoverflow.com/questions/36156284/exact-folder-to-import-csv-to-python
    # https://stackoverflow.com/questions/36519086/how-to-get-rid-of-unnamed-0-column-in-a-pandas-dataframe
    statesDailyCases = pd.read_csv(r'/home/cs17/miniconda3/envs/weather/caseData/daily_cases.csv', parse_dates = ['date'], index_col=0)

    # https://note.nkmk.me/en/python-pandas-dataframe-rename/
    statesDailyCases = statesDailyCases.rename(columns = {'positive': 'cumulative_positive', 'negative': 'cumulative_negative','death': 'cumulative_death'})

    # Filter by range of dates: march 1st - july 15th
    # https://www.interviewqs.com/ddi_code_snippets/select_pandas_dataframe_rows_between_two_dates
    startDate = '2020-03-26'
    endDate = '2020-07-15'
    dateRange = (statesDailyCases['date'] >= startDate) & (statesDailyCases['date'] <= endDate)
    statesDailyCases = statesDailyCases.loc[dateRange]

    # Filter by states to be used
    # https://stackoverflow.com/questions/12096252/use-a-list-of-values-to-select-rows-from-a-pandas-dataframe
    stateList = ['FL', 'NY', 'OH', 'GA', 'VA', 'AZ', 'MA', 'TN', 'MD', 'CO', 'MN', 'SC']
    statesDailyCases = statesDailyCases[statesDailyCases['state'].isin(stateList)]

    # Selection the column attributes to be used
    statesDailyCases = statesDailyCases[['date', 'state', 'cumulative_positive', 'cumulative_negative', 'hospitalized_cumulative', 'cumulative_death']]


    #-----------------------------------------------------------------------------------------------------------------------------#

    
    # Read in all csv state weather files and vertically concatenate them
    # https://medium.com/@kadek/elegantly-reading-multiple-csvs-into-pandas-e1a76843b688
    statesDailyWeather = pd.concat([pd.read_csv(f, parse_dates = ['date']) for f in glob.glob('*.csv')], ignore_index = True)

    # Remove trailing whitespace in 'state' column
    # http://www.datasciencemadesimple.com/strip-space-column-pandas-dataframe-leading-trailing-2/
    statesDailyWeather['state'] = statesDailyWeather['state'].str.strip()

    # write out the concatenation into a file
    statesDailyWeather.to_csv('allStatesWeather.csv', index = False)

    # Transform date to month
    # http://www.datasciencemadesimple.com/get-year-from-date-pandas-python-2/
    #states['date'] = states['date'].dt.to_period('M')
    
    # Group aggregation by states and month
    # https://stackoverflow.com/questions/19078325/naming-returned-columns-in-pandas-aggregate-function/43897124
    # https://jamesrledoux.com/code/group-by-aggregate-pandas
    # https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
    # https://pandas.pydata.org/pandas-docs/stable/whatsnew/v0.25.0.html#enhancements
    statesGroupedByDate = statesDailyWeather.groupby(['state', 'date'])\
    .agg({'awnd':'mean', 'prcp':'mean', 'tmin':'mean','tmax':'mean', 'tavg':'mean', 'tobs':'mean'})\
    .rename(columns = {'awnd': 'avg_daily_wind_speed', 'prcp': 'avg_daily_precip', 'tmin': 'avg_daily_min_tmp',\
                         'tmax': 'avg_daily_max_tmp', 'tavg': 'avg_daily_tmp', 'tobs': 'avg_current_tmp'})
    statesGroupedByDate = statesGroupedByDate.reset_index()

    # Write out the daily average weather measurements for all 25 states on the list
    statesGroupedByDate.to_csv('StatesWeatherDateAggregate.csv', index = False)


    #-----------------------------------------------------------------------------------------------------------------------------#


    # Left join daily cases with daily weather
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
    result = pd.merge(statesDailyCases, statesGroupedByDate, how = 'left', on = ['state', 'date'])

    result.to_csv('DailyData.csv', index = False)


if __name__ == "__main__":
    main()