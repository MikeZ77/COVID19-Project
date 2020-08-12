import numpy as np 
import pandas as pd 


def main():

    
    # Read weather csv files by state - each state has 1 or more csv files
    # 25 states were collected, ranked in the order of most populous
    california1 = pd.read_csv('california1.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    california1 = california1[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    california1['read'] = 'CA'
   
    california2 = pd.read_csv('california2.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    california2 = california2[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    california2['read'] = 'CA'

    texas1 = pd.read_csv('texas1.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    texas1 = texas1[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    texas1['read'] = 'TX'

    texas2 = pd.read_csv('texas2.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    texas2 = texas2[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    texas2['read'] = 'TX'

    texas3 = pd.read_csv('texas3.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    texas3 = texas3[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    texas3['read'] = 'TX'

    florida = pd.read_csv('florida.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    florida = florida[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    florida['read'] = 'FL'

    nyc = pd.read_csv('nyc.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    nyc = nyc[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    nyc['read'] = 'NY'

    penn = pd.read_csv('penn.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    penn = penn[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    penn['read'] = 'PA'

    illinois = pd.read_csv('illinois.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    illinois = illinois[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    illinois['read'] = 'IL'

    ohio = pd.read_csv('ohio.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    ohio = ohio[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    ohio['read'] = 'OH'

    georgia = pd.read_csv('georgia.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    georgia = georgia[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    georgia['read'] = 'GA'

    ncarolina = pd.read_csv('ncarolina.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    ncarolina = ncarolina[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    ncarolina['read'] = 'NC'

    michigan = pd.read_csv('michigan.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    michigan = michigan[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    michigan['read'] = 'MI'

    jersey = pd.read_csv('jersey.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    jersey = jersey[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    jersey['read'] = 'NJ'

    virginia = pd.read_csv('virginia.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    virginia = virginia[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    virginia['read'] = 'VA'

    washington = pd.read_csv('washington.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    washington = washington[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    washington['read'] = 'WA'

    arizona = pd.read_csv('arizona.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    arizona = arizona[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    arizona['read'] = 'AZ'

    mass = pd.read_csv('mass.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    mass = mass[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    mass['read'] = 'MA'

    tenn = pd.read_csv('tenn.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    tenn = tenn[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    tenn['read'] = 'TN'

    indiana = pd.read_csv('indiana.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    indiana = indiana[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    indiana['read'] = 'IN'

    missouri = pd.read_csv('missouri.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    missouri = missouri[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    missouri['read'] = 'MO'

    maryland = pd.read_csv('maryland.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    maryland = maryland[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    maryland['read'] = 'MD'

    wisconsin = pd.read_csv('wisconsin.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    wisconsin = wisconsin[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    wisconsin['read'] = 'WI'

    colorado1 = pd.read_csv('colorado1.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    colorado1 = colorado1[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    colorado1['read'] = 'CO'

    colorado2 = pd.read_csv('colorado2.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    colorado2 = colorado2[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    colorado2['read'] = 'CO'

    minnesota = pd.read_csv('minnesota.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    minnesota = minnesota[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    minnesota['read'] = 'MN'

    scarolina = pd.read_csv('scarolina.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    scarolina = scarolina[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    scarolina['read'] = 'SC'

    alabama = pd.read_csv('alabama.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    alabama = alabama[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    alabama['read'] = 'AL'

    louisiana = pd.read_csv('louisiana.csv', parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    louisiana = louisiana[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    louisiana['read'] = 'LA'


    # Vertically stacked all the dataframes by turning them into a list first
    # https://stackoverflow.com/questions/41181779/merging-2-dataframes-vertically
    WeatherStackedDF = pd.concat([florida, nyc, penn, illinois, ohio, georgia,california1, california2, 
                                texas1, texas2, texas3, ncarolina, michigan, jersey, virginia, washington, 
                                arizona, mass, tenn, indiana, missouri, maryland, wisconsin, colorado1, colorado2, 
                                minnesota, scarolina, alabama, louisiana], ignore_index=True)

    # Change column names to lower case
    WeatherStackedDF.columns = map(str.lower, WeatherStackedDF.columns)

    # Extract state abbreviation from string
    #https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/
    weatherDF = WeatherStackedDF.assign(state = WeatherStackedDF['name'].str[-5:-3])
    del weatherDF['name']

    # extract month from date - to impute by month
    # https://stackoverflow.com/questions/32490629/getting-todays-date-in-yyyy-mm-dd-in-python
    weatherDF['month'] = pd.DatetimeIndex(weatherDF['date']).month
    
    # convert date to the format: yyyymmdd
    # https://www.geeksforgeeks.org/python-pandas-series-dt-strftime/
    # https://www.kite.com/python/answers/how-to-change-the-pandas-datetime-format-in-python
    # https://stackoverflow.com/questions/28133018/convert-pandas-series-to-datetime-in-a-dataframe
    weatherDF['date'] = weatherDF['date'].dt.strftime('%Y%m%d')

    by_state = weatherDF.groupby("state")
    #by_state.get_group("PA")


    # Verify each file contains its matching state
    for state, frame in by_state:
        print(f"First 2 entries for {state!r}")
        print("------------------------")
        print(frame.head(2), end="\n\n")


if __name__ == "__main__":
    main()