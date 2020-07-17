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
    """
	AIAN =  American Indian and Alaska Native
	NHPI = 	Native Hawaiian and Pacific Islander
	Cases_LatinX includes Hispanic	
    """
    race_df = race_df[
        ['Date',
         'State',
         'Cases_White',
         'Cases_Black',
         'Cases_LatinX',
         'Cases_Asian',
         'Cases_AIAN',
         'Cases_NHPI',
         'Cases_Multiracial',
         'Cases_Other',
         'Cases_Unknown',
         'Deaths_White',
         'Deaths_Black',
         'Deaths_LatinX',
         'Deaths_Asian',
         'Deaths_Asian',
         'Deaths_AIAN',
         'Deaths_NHPI',
         'Deaths_Multiracial',
         'Deaths_Other',
         'Deaths_Unknown'
         ]
    ]
    print(daily_df)
    print(race_df)
    """
	Note: dataQualityGrade is a grade based on 5 factors: Reporting total, Testing total, Outcomes total, and 
		  Demographic total. It is based on the amount of this data, and not on the quality of the data itself.
		  Therefore, it makes the most sense to remove States ourselves that are not reporting the data or 
		  interpolate if possible (as opposed to relying on the dataQualityGrade alone).  
    """

    # Combine dataframes (Group by Date and State)
    # Remove States that have gaps in data > 15% (?). Otherwise interpolate the missing data.


if __name__ == '__main__':
    main()
