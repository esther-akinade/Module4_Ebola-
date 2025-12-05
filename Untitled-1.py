## LOAD YOUR DATASET HERE. 
from main_functions import convert_cumulative_to_SIR, euler_sir
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# 1. Load the Ebola Sierra Leone dataset
df = pd.read_csv("ebola_sierra_leone_data_2014_2015_cumulative.csv")
df['date'] = pd.to_datetime(df['date'])

# Population in thousands
N = 6876000 / 1e3   

# Convert cumulative data → S,I,R estimates
data_sir = convert_cumulative_to_SIR(
    df,
    date_col='date',
    cumulative_col='confirmed_cases',
    population=N,
    infectious_period=21,
    new_case_col='new_cases',
    I_col='I_est',
    R_col='R_est',
    S_col='S_est'
)

# Convert to thousands for plotting
data_sir['I_est'] = data_sir['I_est'] / 1e3

# Extract arrays
I_obs_full = data_sir['I_est'].values.astype(float)
t_full = np.linspace(0, len(I_obs_full)-1, len(I_obs_full))

I0 = I_obs_full[0]
R0 = 0.0
S0 = N - I0 - R0

# ----------------------------------------------------------
# 2️⃣  **Use FIRST HALF of the data for optimization**
# ----------------------------------------------------------
mid = len(I_obs_full) // 2

I_obs = I_obs_full[:mid]
t_obs = t_full[:mid]

def SSE(params):
    beta, gamma = params
    S_pred, I_pred, R_pred = euler_sir(beta, gamma, S0, I0, R0, t_obs, N)
    return np.mean((I_pred - I_obs) ** 2)

initial_guess = [0.2, 0.1]
bounds = [(0.0001, 5), (0.0001, 2)]

result = minimize(SSE, initial_guess, bounds=bounds, method='L-BFGS-B')
beta_hat, gamma_hat = result.x

print("\n=======================")
print(" OPTIMIZATION RESULTS")
print("=======================")
print("Fitted beta :", beta_hat)
print("Fitted gamma:", gamma_hat)
print("Minimum SSE (first half) :", result.fun)

# ----------------------------------------------------------
# 3️⃣  **Simulate forward using optimized β,γ over FULL time**
# ----------------------------------------------------------
S_fit_full, I_fit_full, R_fit_full = euler_sir(
    beta_hat, gamma_hat, S0, I0, R0, t_full, N
)

# ----------------------------------------------------------
# 4️⃣  **Plot observed data vs model prediction**
# ----------------------------------------------------------
plt.figure(figsize=(12,6))

# Full observed
plt.plot(t_full, I_obs_full, 'o', label='Observed I(t)', markersize=4)

# Model prediction
plt.plot(
    t_full, I_fit_full, '-', linewidth=2,
    label=f'Forecasted SIR Model (β={beta_hat:.3f}, γ={gamma_hat:.3f})'
)

# Mark first-half fitting region
plt.axvline(mid, color='gray', linestyle='--', label='End of Training Window')

plt.xlabel("days")
plt.ylabel("Infections (thousands)")
plt.title("SIR Model Fitted on First Half → Forecast on Full Period")
plt.legend()
plt.tight_layout()
plt.show()
# Use euler's method and your optimization routine above to find new gamma and beta on the  
# FIRST HALF of the data, then simulate the SIR model forward in time using those parameters and plot.