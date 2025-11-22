##############################################
# %% RUN AS AN INTERACTIVE NOTEBOOK
from main_functions import convert_cumulative_to_SIR
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# %%
# Load the COVID-19 dataset
data = pd.read_csv('Module_4/Data/time_series_covid19_confirmed_global.csv')
# Display the first few rows of the dataset
print(data.head())

# Preprocess the data to get the total confirmed cases over time for US
data = data.loc[data["Country/Region"] == "US"].transpose().reset_index()
data = data.drop(index=0)  # drop the 'Country/Region' row
data.columns = ['date', 'confirmed_cases']
data['date'] = pd.to_datetime(data['date'])


# %%
# Plot the confirmed cases over time for first 2 months of data (until shutdown)
plt.figure(figsize=(10, 6))
plt.plot(data['date'],
         data['confirmed_cases'],
         label='Confirmed Cases',
         marker="o")
plt.ylim(0, 10000)
plt.xlabel('Date')
plt.xlim(pd.Timestamp('2020-01-21'), pd.Timestamp('2020-03-17'))
# Format ticks as M/D (no leading zeros)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%-m/%-d'))
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.ylabel('Number of Cases')
plt.title('US COVID-19 Confirmed Cases Over Time')
plt.legend()
plt.show()


# %%
# HOW WOULD YOU PREDICT THE TREND OF COVID-19 CASES IN THE FOLLOWING DAYS?
# 1. March 18, 2020 (1 day)
# Write out your answer here (in words, not code)

# 2. April 17, 2020 (1 month)
# Write out your answer here (in words, not code)

# 3. March 17, 2021 (1 year)
# Write out your answer here (in words, not code)

##############################################
# %%
# Let's look at new infections (incidence) over time: I(t)

# notice that we are taking the difference between days to get new cases
data['new_cases'] = data['confirmed_cases'].diff().fillna(0)

plt.figure(figsize=(10, 6))
plt.plot(data['date'],
         data['new_cases'],
         label='New Confirmed Cases',
         marker="o")
plt.xlabel('Date')
plt.ylim(0, 2100)
plt.xlim(pd.Timestamp('2020-01-21'), pd.Timestamp('2020-03-17'))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%-m/%-d'))
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.ylabel('Number of New Cases')
plt.title('New US COVID-19 Cases Over Time')
plt.legend()
plt.show()

# %%
# Let's look at new infections over time for the first year

plt.figure(figsize=(10, 6))
plt.plot(data['date'],
         data['new_cases'],
         label='New Cases',
         marker="o")
plt.xlabel('Date')
plt.xlim(pd.Timestamp('2020-01-21'), pd.Timestamp('2021-03-17'))
plt.ylim(0, 400000)

# Format ticks as M/Y (no leading zeros)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%-m/%-y'))
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=12))
plt.ylabel('Number of Cases')
plt.title('New US COVID-19 Cases Over Time')
plt.legend()
plt.show()

# %%
# new cases each day does not represent the number of currently infectious individuals I(t)
# Let's use the convert_cumulative_to_SIR function to approximate S(t), I(t), and R(t) from the data
population = 331002651  # US population approx as of 2020
data_sir = convert_cumulative_to_SIR(
    data,
    date_col='date',
    cumulative_col='confirmed_cases',
    population=population,
    infectious_period=14,
    new_case_col='new_cases',
    I_col='I_est',
    R_col='R_est',
    S_col='S_est')


# %%
# Plot the Infectious population over time for first year after shutdown
plt.figure(figsize=(10, 6))

plt.plot(data_sir['date'],
         data_sir['I_est'],
         label='Infectious (I)',
         color='red')

plt.xlabel('Date')
plt.xlim(pd.Timestamp('2020-01-21'), pd.Timestamp('2021-03-17'))
plt.ylim(0, 4e6)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%-m/%-y'))
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=12))
plt.ylabel('Percent of Individuals')
plt.title('Estimated Infections of COVID-19 in US')
plt.legend()
plt.show()
# %%
# Plot the SIR estimates over time
plt.figure(figsize=(10, 6))
plt.plot(data_sir['date'],
         data_sir['S_est'],
         label='Susceptible (S)',
         color='blue')
plt.plot(data_sir['date'],
         data_sir['I_est'],
         label='Infectious (I)',
         color='red')
plt.plot(data_sir['date'],
         data_sir['R_est'],
         label='Recovered (R)',
         color='green')
plt.xlabel('Date')
# plt.ylim(0, 8e9)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%-m/%-y'))
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=12))
plt.ylabel('Percent of Individuals')
plt.title('Approximated SIR Populations of COVID-19 in US')
plt.legend()
plt.show()

# %%
# Save I(t) estimates to CSV for use in Module 4 Example Notebook
# data_sir[['date', 'I_est']].to_csv(
#     'Module_4/Data/US_COVID19_I_estimates.csv', index=False)

# %%
# What assumptions are being made in the SIR model? (You can copy these to the limitations section of your project Jupyter notebook)
