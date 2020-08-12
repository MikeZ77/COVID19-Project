import sys
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.experimental import enable_iterative_imputer # https://stackoverflow.com/questions/55846680/can-not-import-iterativeimputer-from-sklearn-impute
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


def main(readFile, writeFile):

    # Read a weather csv file
    # https://stackoverflow.com/questions/24251219/pandas-read-csv-low-memory-and-dtype-options
    # https://stackoverflow.com/questions/29376026/find-mixed-types-in-pandas-columns
    weatherFile = pd.read_csv(readFile, parse_dates = ['DATE'], sep=',', error_bad_lines=False, index_col=False, low_memory=False)
    weatherColumns = weatherFile[['NAME', 'DATE', 'AWND', 'PRCP', 'TMIN', 'TMAX', 'TAVG', 'TOBS']]
    
    # change column names to lower case
    # https://chrisalbon.com/python/data_wrangling/pandas_lowercase_column_names/
    weatherColumns.columns = map(str.lower, weatherColumns.columns)

    # http://www.datasciencemadesimple.com/return-first-n-character-from-left-of-column-in-pandas-python/
    #weatherColumns['name'] = weatherColumns['name'].str[-5:-2]
    
    # state abbreviation
    #https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/
    weatherDF = weatherColumns.assign(state = weatherColumns['name'].str[-5:-2])
    del weatherDF['name']
    
    # extract month from date - to impute by month
    # https://stackoverflow.com/questions/32490629/getting-todays-date-in-yyyy-mm-dd-in-python
    weatherDF['month'] = pd.DatetimeIndex(weatherDF['date']).month
    
    # convert date to a given format
    # https://www.geeksforgeeks.org/python-pandas-series-dt-strftime/
    # https://www.kite.com/python/answers/how-to-change-the-pandas-datetime-format-in-python
    # https://stackoverflow.com/questions/28133018/convert-pandas-series-to-datetime-in-a-dataframe
    weatherDF['date'] = weatherDF['date'].dt.strftime('%Y%m%d')

    # single column mean imputation
    # https://stackoverflow.com/questions/19966018/pandas-filling-missing-values-by-mean-in-each-group
    weatherDF['awnd'] = weatherDF['awnd'].fillna(weatherDF.groupby('month')['awnd'].transform('mean')).round(decimals=2)
    weatherDF['prcp'] = weatherDF['prcp'].fillna(weatherDF.groupby('month')['prcp'].transform('mean')).round(decimals=2)

    # multiple columns imputation using a regressor
    X = weatherDF.iloc[:,3:7].values 
    imp = IterativeImputer(estimator = BayesianRidge(), max_iter = 15, missing_values = np.nan) # default estimator is bayesian ridge
    imputed_df = imp.fit_transform(X)
    imputed_df = pd.DataFrame(imputed_df, columns = weatherDF.iloc[:,3:7].columns).round(decimals=2)
 
    # concatenate dfs in order
    # https://stackoverflow.com/questions/29221502/pandas-selecting-discontinuous-columns-from-a-dataframe
    # https://stackoverflow.com/questions/39534676/typeerror-first-argument-must-be-an-iterable-of-pandas-objects-you-passed-an-o
    weather_df_cleaned = pd.concat([weatherDF.iloc[:,0:3], imputed_df, weatherDF.iloc[:,7]], axis=1) 

    #titanic = pd.DataFrame(imp.fit_transform(weatherDF), columns=weatherDF.columns)
    # https://stackoverflow.com/questions/28161356/sort-pandas-dataframe-by-date
    #weatherColumns = weatherColumns.sort_values(by = ['date'])

    weather_df_cleaned.to_csv(writeFile, index = False)


if __name__=='__main__':
    readFile = sys.argv[1]
    writeFile = sys.argv[2]
    main(readFile, writeFile)