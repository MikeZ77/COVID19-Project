RUN:
python 01-clean_data.py states_daily.csv ethnicity_daily.csv socioecnomic_data.csv socioeconoic_data_over_60.csv 50_us_states_mapping.csv
02-descriptive_choropleth.py
03-regression.py
04-inference.py
05-correlation.py

RUN:
the following scripts in the folder raw data NOAA
06-NOAADataCleaning.py
07-weatherCleanup.py

the following script in the folder weather
08-statesCombined.py

the follwing scripts in the folder timeseries
09-caseCorrelation.py
10-covidCases.py
 
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




