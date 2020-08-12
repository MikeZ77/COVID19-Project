import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
from scipy import stats
from scipy.special import boxcox, inv_boxcox


def make_comparison_dataframe(historical, forecast):
    """ Join the history with the forecast.
        The resulting dataset will contain columns 'yhat', 'yhat_lower', 'yhat_upper' and 'y'.
    """
    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))


def calculate_forecast_errors(df, prediction_size):
    """Calculate MAPE and MAE of the forecast.
    
       Args:
           df: joined dataset with 'y' and 'yhat' columns.
           prediction_size: number of days at the end to predict.
    """
    
    # Make a copy
    df = df.copy()
    
    # Now we calculate the values of e_i and p_i according to the formulas given in the article above.
    df['e'] = df['y'] - df['yhat']
    df['p'] = 100 * df['e'] / df['y']
    
    # Recall that we held out the values of the last `prediction_size` days
    # in order to predict them and measure the quality of the model. 
    
    # Now cut out the part of the data which we made our prediction for.
    predicted_part = df[-prediction_size:]
    
    # Define the function that averages absolute error values over the predicted part.
    error_mean = lambda error_name: np.mean(np.abs(predicted_part[error_name]))
    
    # Now we can calculate MAPE and MAE and return the resulting dictionary of errors.
    return {'MAPE': error_mean('p'), 'MAE': error_mean('e')}



def inverse_boxcox(y, lambda_):
    return np.exp(y) if lambda_ == 0 else np.exp(np.log(lambda_ * y + 1) / lambda_)



def main():

    data = pd.read_csv('DailyData.csv', parse_dates = ['date'])
    #test = df.sort('one', ascending=False)

    # https://stackoverflow.com/questions/27673231/why-should-i-make-a-copy-of-a-data-frame-in-pandas
    state_df = data[data['state'] == 'MA'].copy()
    state_df['daily_positive'] = state_df['cumulative_positive'] - state_df['cumulative_positive'].shift(-1)
    state_df['daily_negative'] = state_df['cumulative_negative'] - state_df['cumulative_negative'].shift(-1)
    state_df['daily_hospitalized'] = state_df['hospitalized_cumulative'] - state_df['hospitalized_cumulative'].shift(-1)
    state_df['daily_death'] = state_df['cumulative_death'] - state_df['cumulative_death'].shift(-1)
    
    # https://stackoverflow.com/questions/36518027/pandas-creating-new-data-frame-from-only-certain-columns
    state = state_df[['date', 'daily_death']].copy()
    
    # https://stackoverflow.com/questions/11346283/renaming-columns-in-pandas
    state.rename({'date': 'ds', 'daily_death': 'y'}, axis = 1, inplace = True)
    
    # https://stackoverflow.com/questions/35240528/reverse-dataframes-rows-order-with-pandas
    # https://stackoverflow.com/questions/16167829/in-pandas-how-can-i-reset-index-without-adding-a-new-column
    state =  state[::-1]
    
    # https://stackoverflow.com/questions/16396903/delete-the-first-three-rows-of-a-dataframe-in-pandas
    state = state.iloc[1:]

    # https://stackoverflow.com/questions/23307301/replacing-column-values-in-a-pandas-dataframe
    #az_df.y[az_df.y <= 1] = 1 
    df = state[state.y > 1].reset_index(drop=True)

    # Split dataset into original, the first part of the data, and prediction part, at the end of the timeline.
    # The last 5 days removed from the dataset are to be used later as the prediction target
    prediction_size = 8
    train_df = df[:-prediction_size]

    model = Prophet(daily_seasonality = True) #instantiate Prophet
    model.fit(train_df); #fit the model with your dataframe

    future_data = model.make_future_dataframe(prediction_size)
    forecast_data = model.predict(future_data)
    
    model.plot(forecast_data, xlabel = 'date', ylabel = 'death').savefig('1.png')

    model.plot_components(forecast_data).savefig('2.png')

    '''
    # log transform the ‘y’ variable to a try to convert non-stationary data to stationary
    # Save a copy of the original data
    az_df['y_original'] = az_df['y'] 
    # log-transform y
    az_df['y'] = np.log(az_df['y'])

    model = Prophet(daily_seasonality=True) #instantiate Prophet
    model.fit(az_df); #fit the model with your dataframe

    # Make a future dataframe for the next 7 days
    future_data = model.make_future_dataframe(periods=7)
    forecast_data = model.predict(future_data)
    
    model.plot(forecast_data, xlabel = 'date', ylabel = 'log death').savefig('1.png')


    
    #print(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    model.plot_components(forecast_data).savefig('2.png')
    '''


    '''
    # forcast in the context of original data
    forecast_data_orig = forecast_data # make sure we save the original forecast data
    forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
    forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
    forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])
    model.plot(forecast_data_orig).savefig('3.png')

    az_df['y_log']=az_df['y'] #copy the log-transformed data to another column
    az_df['y']=az_df['y_original'] #copy the original data to 'y'
    model.plot(forecast_data_orig).savefig('4.png')
   

    forecast_data[['yhat','yhat_upper','yhat_lower']] = forecast_data[['yhat','yhat_upper','yhat_lower']].apply(lambda x: inv_boxcox(x, lam))
    model.plot(forecast_data).savefig('3.png')
    '''


    cmp_df = make_comparison_dataframe(df, forecast_data)
    for err_name, err_value in calculate_forecast_errors(cmp_df, prediction_size).items():
        print(err_name, err_value)



    train_df2 = train_df.copy().set_index('ds')
    train_df2['y'], lambda_prophet = stats.boxcox(train_df2['y'])
    train_df2.reset_index(inplace=True)

    m2 = Prophet(daily_seasonality = True)
    m2.fit(train_df2)
    future2 = m2.make_future_dataframe(periods=prediction_size)
    forecast2 = m2.predict(future2)

    m2.plot(forecast2, xlabel = 'date', ylabel = 'death').savefig('3.png')
    m2.plot_components(forecast2).savefig('4.png')


    for column in ['yhat', 'yhat_lower', 'yhat_upper']:
        forecast2[column] = inverse_boxcox(forecast2[column], lambda_prophet)

    cmp_df2 = make_comparison_dataframe(df, forecast2)
    for err_name, err_value in calculate_forecast_errors(cmp_df2, prediction_size).items():
        print(err_name, err_value)




if __name__ == "__main__":
    main()