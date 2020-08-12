from kaleido.scopes.plotly import PlotlyScope
import plotly.graph_objects as go
import datetime as dt
import pandas as pd

DATA_PATH = 'output/'
IMAGE_PATH = 'images/maps/'
FILE = 'total_cases_with_socioecon.csv'
scope = PlotlyScope()


def get_latest_date(date_df):
    current_date = str(date_df.values[0])
    current_date = dt.datetime.strptime(current_date, '%Y%m%d').date().isoformat()
    return current_date


def read_data():
    covid_df = pd.read_csv(DATA_PATH + FILE)
    return covid_df


def set_data_set(covid_df):
    map_features = ['state', 'death', 'pop_density', 'cdc_svi_overall_ranking', 'et_over_60_population', 'date']
    covid_df = covid_df[map_features]
    return covid_df

def plot_deaths(covid_df):

    current_date = get_latest_date(covid_df['date'])

    fig = go.Figure(data=go.Choropleth(
        locations=covid_df['state'],
        z=covid_df['death'].astype(int),
        locationmode='USA-states',
        colorscale='Reds',
        # colorbar_title="Deaths",
        # marker_line_color='white'
    ))

    fig.update_layout(
        title_text='Total Deaths per US State',
        geo_scope='usa',
    )

    with open(IMAGE_PATH + "deaths_us_map.png", "wb") as file:
        file.write(scope.transform(fig, format="png"))


def plot_death_per_population_over_60(covid_df):

    covid_df['death_by_population'] = covid_df['death'] / covid_df['et_over_60_population']
    current_date = get_latest_date(covid_df['date'])

    fig = go.Figure(data=go.Choropleth(
        locations=covid_df['state'],
        z=covid_df['death_by_population'].astype(float),
        locationmode='USA-states',
        colorscale='Reds',
        # colorbar_title="Deaths / Pop. 60+",
    ))

    fig.update_layout(
        title_text='Relative Deaths by Population Over 60',
        geo_scope='usa',
    )

    with open(IMAGE_PATH + "deaths_per_over_60_map.png", "wb") as file:
        file.write(scope.transform(fig, format="png"))


def plot_death_per_population_density(covid_df):

    covid_df['death_by_density'] = covid_df['death'] / covid_df['pop_density']
    current_date = get_latest_date(covid_df['date'])

    fig = go.Figure(data=go.Choropleth(
        locations=covid_df['state'],
        z=covid_df['death_by_density'].astype(float),
        locationmode='USA-states',
        colorscale='Reds',
        # colorbar_title="Deaths / Pop. Density",
    ))

    fig.update_layout(
        title_text='Relative Deaths by Population Density',
        geo_scope='usa',
    )

    with open(IMAGE_PATH + "deaths_per_pop_density.png", "wb") as file:
        file.write(scope.transform(fig, format="png"))

def main():

    covid_df = read_data()
    covid_df = set_data_set(covid_df)
    plot_deaths(covid_df)
    plot_death_per_population_over_60(covid_df)
    plot_death_per_population_density(covid_df)


if __name__ == '__main__':
    main()
