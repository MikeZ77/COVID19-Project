import pandas as pd
import numpy as np
import sys


def main():

        # Feature descriptions: https://covidtracking.com/data-definitions

    daily_df = pd.read_csv(sys.argv[1])
    race_df = pd.read_csv(sys.argv[2])

    daily_df = daily_df[
        ['date',
         'state',
         'positive',
         'negative',
         'hospitalizedCurrently',
         'inIcuCurrently',
         'onVentilatorCurrently',
         'recovered',
         'death',
         'hospitalized',
         'positiveTestsViral',
         'negativeTestsViral',
         'totalTestResults',
         'dataQualityGrade']
    ]
    print(daily_df)
    race_df = race_df[
        ['Date',
         'State',
         'Cases_White',
         'Cases_Black',
         'Cases_LatinX',
         'Cases_Asian',
         'Cases_Multiracial'
         ]
    ]

    # Combine dataframes (Group by Date and State)
    # Remove States that have gaps in data > 15%. Otherwise interpolate the missing data.
    # Remove States that have a dataQualityGrade of less than ...


if __name__ == '__main__':
    main()
