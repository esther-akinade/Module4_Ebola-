
# %%
# Module 4 Main Function Code
from main_functions import convert_cumulative_to_SIR
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
 ##Load the Ebola_Sierra Leone Data 
data = pd.read_csv('ebola_sierra_leone_data_2014_2015_cumulative.csv')
## Display the first few rows of the dataset
print(data.head())

data = data.drop(index=0)  # drop the 'Country/Region' row
data.columns = ['date', 'confirmed_cases']
data['date'] = pd.to_datetime(data['date'])

# %%
import numpy as np, pandas as pd 
from datetime import datetime, timedelta
from main_functions import convert_cumulative_to_SIR

# SL Ebola Cases 
N =  # SL population in millions 
df_full = pd.read_csv("ebola_sierra_leone_data_2014_2015_cumulative.csv")
# Convert date column
df_full['date'] = pd.to_datetime(df_full['date'])

data_sir = convert_cumulative_to_SIR(
    df_full,
    date_col='date',
    cumulative_col='confirmed_cases',
    population=N,
    infectious_period=21,
    new_case_col='new_cases',
    I_col='I_est',
    R_col='R_est',
    S_col='S_est'
)

df_full = data_sir
df_full['I_est'] = df_full['I_est'] / 1e6  # convert to millions

# Correct date filter (Aug 2014 → Dec 2015)
df_full = df_full[
    (df_full['date'] >= '2014-08-29') &
    (df_full['date'] <= '2015-12-29')
]

df_full.head()
# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

plt.plot(data_sir['date'], data_sir['I_est'], 'o-')
plt.xlabel('date'); plt.ylabel('Infections (in millions)')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
plt.show()

# %%
