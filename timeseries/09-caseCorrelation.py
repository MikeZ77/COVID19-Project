import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns

def main():

    data = pd.read_csv('DailyData.csv', parse_dates = ['date'])
    data.reset_index(drop=True, inplace=True)    
    data.columns = ['date', 'state', 'cumulative_positive', 'cumulative_negative', 'hospitalized_cumulative', 'cumulative_death', 'daily_precip', 'current_tmp', 'daily_max_tmp', 'daily_wind_speed', 'daily_min_tmp', 'daily_tmp']

    # https://stackoverflow.com/questions/27673231/why-should-i-make-a-copy-of-a-data-frame-in-pandas
    state_df = data[data['state'] == 'MA'].copy()
    state_df['daily_positive'] = state_df['cumulative_positive'] - state_df['cumulative_positive'].shift(-1)
    state_df['daily_negative'] = state_df['cumulative_negative'] - state_df['cumulative_negative'].shift(-1)
    state_df['daily_hospitalized'] = state_df['hospitalized_cumulative'] - state_df['hospitalized_cumulative'].shift(-1)
    state_df['daily_death'] = state_df['cumulative_death'] - state_df['cumulative_death'].shift(-1)
    
    # https://stackoverflow.com/questions/36518027/pandas-creating-new-data-frame-from-only-certain-columns
    state = state_df[['daily_positive', 'daily_hospitalized', 'daily_death','daily_tmp', 'daily_wind_speed', 'daily_precip']].copy()
    state = state.dropna()
    #print(state)

    # https://stackoverflow.com/questions/56876795/change-marker-size-in-seaborn-pairplot-with-kind-reg
    # https://stackoverflow.com/questions/50722972/change-the-regression-line-colour-of-seaborns-pairplot
    #plt.figure(figsize=(20,20))
    g = sns.pairplot(state, diag_kind = 'kde', kind = 'reg', plot_kws= dict(line_kws = dict(color = 'red'), scatter_kws=dict(s=2)), height=10, aspect=0.3)
    '''
    state.corr()['daily_death'].sort_values(ascending = False)
    scatter_matrix(state, alpha=0.4, diagonal = None, figsize=(20,20))
    #scatter_matrix(state.loc[:,'daily_positive':'avg_daily_precip'], alpha=0.5)
    '''
    plt.suptitle('Massachusetts')
    plt.show()
    

if __name__ == "__main__":
    main()