NOTE:
Steps to run the project and dependancies. Format: python script.py arg1 arg2 ... #The expected output. python has been left implicit 
since this can vary depending on your environment.

RUN:
01-clean_data.py states_daily.csv ethnicity_daily.csv socioecnomic_data.csv socioeconoic_data_over_60.csv 50_us_states_mapping.csv
#Expect: cleaned base .csv files in the output folder
 
02-descriptive_choropleth.py
#Expect: choropleth images in the images/maps folder
    		
03-regression.py
#Expect: regression images in the images/linear_regression folder
				
04-inference.py
#Expect: inference images in the images/deaths_race folder
				
05-correlation.py
#Expect: correlation images in the images/correlation folder				

The following scripts are located in the folder: weather/raw data NOAA
06-NOAADataCleaning.py
#Expect: all state data to be validated
				
07-weatherCleanup.py statename outputname
#Expect: all 25 imputed states to be generated in 25 csvs	

The following script is located in the folder: weather
08-statesCombined.py
#Expect: the following 3 files generated - allStatesWeather.csv, StatesWeatherDateAggregate.csv, DailyData.csv     			

The following scripts are located in the folder: timeseries
09-caseCorrelation.py
#Expect: a combination of pair matrix scatter plot
				
10-covidCases.py				
#Expect: univariate timeseries prediction model, breakdown of the model by trend and seasonality and Errors for each input state				
 
SETUP:
pip install python==3.6.11
pip install pandas==1.0.3
pip install numpy==1.18.1
pip install plotly==4.9.0
pip install kaleido==0.0.1
pip install matplotlib==3.1.3
pip install statsmodels
pip install scipy==1.5.0
pip install scikit-learn==0.23.1
pip install ephem==3.7.7.1
pip install pystan==2.19.1.1
pip install fbprophet==0.6
pip install seaborn==0.9.0




