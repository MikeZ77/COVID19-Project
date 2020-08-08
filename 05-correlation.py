import pandas as pd
import numpy as np
import plotly.graph_objects as go
from kaleido.scopes.plotly import PlotlyScope

DATA_PATH = 'output/'
FILE_1 = 'daily_cases_wtih_race.csv'
IMAGE_PATH = 'images/correlation/'
scope = PlotlyScope()
pd.set_option('display.max_rows', 3000)
# pd.set_option('display.max_columns', 20)
# pd.set_option('display.width', 20)


def read_data():
    covid_df = pd.read_csv(DATA_PATH + FILE_1, index_col=[0])
    covid_df['date'] = pd.to_datetime(covid_df['date'].astype(str), format='%Y%m%d')

    return covid_df


def compute_covariance_matrix(covid_df):
    """
    States are CA, CO, WA
    """
    covid_df = covid_df[covid_df['date'] >= '2020-05-03']
    headers = ['date',
               'cases_white',
               'cases_black',
               'cases_asian',
               'cases_aian',
               'cases_nhpi',
               'cases_multiracial',
               'cases_ethnicity_hispanic',
               'deaths_white',
               'deaths_black',
               'deaths_asian',
               'deaths_aian',
               'deaths_nhpi',
               'deaths_multiracial',
               'deaths_ethnicity_hispanic',
               ]
    covid_df = covid_df[headers]
    covid_df = covid_df.dropna(thresh=len(headers) - 1).reset_index(drop=True)
    # There is an entry with a comma (object type) in cases_white which ends up getting dropped.
    covid_df['cases_white'] = covid_df['cases_white'].astype(float)
    covid_df = covid_df.groupby('date').sum().diff().iloc[1:]

    return covid_df


def create_heatmap(covid_df):

    covid_df = covid_df.reset_index(drop=True)

    covariance_matrix = covid_df.corr()
    cases_corr_matrix = covariance_matrix.iloc[0:7, 0:7]
    death_corr_matrix = covariance_matrix.iloc[7:, 7:]
    cases_death_corr_matrix = covariance_matrix.iloc[0:7, 7:]

    np.fill_diagonal(cases_corr_matrix.values, np.nan)
    np.fill_diagonal(death_corr_matrix.values, np.nan)

    cases_fig = go.Figure(
        data=go.Heatmap(
            z=cases_corr_matrix.values.tolist(),
            x=list(cases_corr_matrix.columns.values),
            y=list(cases_corr_matrix.columns.values),
            hoverongaps=False,
            colorbar=dict(title='Correlation')),
    )

    cases_fig.update_layout(
        title_text='Heatmap Cases by Race',
    )

    with open(IMAGE_PATH + "cases_corr_matrix.png", "wb") as file:
        file.write(scope.transform(cases_fig, format="png"))

    death_fig = go.Figure(data=go.Heatmap(
        z=death_corr_matrix.values.tolist(),
        x=list(death_corr_matrix.columns.values),
        y=list(death_corr_matrix.columns.values),
        hoverongaps=False,
        colorbar=dict(title='Correlation')),
    )

    death_fig.update_layout(
        title_text='Heatmap Death by Race',
    )

    with open(IMAGE_PATH + "death_corr_matrix.png", "wb") as file:
        file.write(scope.transform(death_fig, format="png"))

    cases_death_fig = go.Figure(data=go.Heatmap(
        z=cases_death_corr_matrix.values.tolist(),
        x=list(cases_death_corr_matrix.columns.values),
        y=list(cases_death_corr_matrix.index.values),
        hoverongaps=False,
        colorbar=dict(title='Correlation')),
    )

    cases_death_fig.update_layout(
        title_text='Heatmap Death and Cases by Race',
    )

    with open(IMAGE_PATH + "cases_death_corr_matrix.png", "wb") as file:
        file.write(scope.transform(cases_death_fig, format="png"))


def main():
    covid_df = read_data()
    covid_df = compute_covariance_matrix(covid_df)
    covid_df = create_heatmap(covid_df)


if __name__ == '__main__':
    main()
