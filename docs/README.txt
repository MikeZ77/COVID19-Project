RUN:
python 01-clean_data.py states_daily.csv ethnicity_daily.csv socioecnomic_data.csv socioeconoic_data_over_60.csv 50_us_states_mapping.csv
02-descriptive_choropleth.py
03-regression.py
04-inference.py
05-correlation.py
 
SETUP:
pip install pandas
pip install plotly==4.9.0
pip install kaleido==0.0.1
pip install matplotlib
pip install statsmodels




