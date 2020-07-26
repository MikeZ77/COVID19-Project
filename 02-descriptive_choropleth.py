from kaleido.scopes.plotly import PlotlyScope
import plotly.graph_objects as go
import datetime as dt
import pandas as pd

DATA_PATH = 'output/'
IMAGE_PATH = 'images/'
FILE = 'total_cases_with_socioecon.csv'


def read_data():
    covid_df = pd.read_csv(DATA_PATH + FILE)
    return covid_df


def set_data_set(covid_df):
    map_features = ['state', 'death', 'pop_density', 'cdc_svi_overall_ranking', 'date']
    covid_df = covid_df[map_features]
    return covid_df

def map_features(covid_df):

    current_date = str(covid_df['date'].values[0])
    current_date = dt.datetime.strptime(current_date, '%Y%m%d').date().isoformat()

    scope = PlotlyScope()

    fig = go.Figure(data=go.Choropleth(
        locations=covid_df['state'],  # Spatial coordinates
        z=covid_df['death'].astype(int),  # Data to be color-coded
        locationmode='USA-states',  # set of locations match entries in `locations`
        colorscale='Reds',
        colorbar_title="Deaths",
        # marker_line_color='white'
    ))

    fig.update_layout(
        title_text='Total Deaths per US State as of ' + current_date,
        geo_scope='usa',  # limite map scope to USA
    )

    with open(IMAGE_PATH + "deaths_us_map.png", "wb") as file:
        file.write(scope.transform(fig, format="png"))


def main():

    covid_df = read_data()
    covid_df = set_data_set(covid_df)
    map_features(covid_df)


if __name__ == '__main__':
    main()
