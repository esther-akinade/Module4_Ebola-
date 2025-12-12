## LOAD YOUR DATASET HERE.
from main_functions import convert_cumulative_to_SIR
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the Ebola Sierra Leone dataset
df = pd.read_csv("ebola_sierra_leone_data_2014_2015_cumulative.csv")
print(df.head())   # show first few rows

# Ensure date column is actual datetime
df['date'] = pd.to_datetime(df['date'])

# 2. Convert cumulative cases to S, I, R estimates
N = 6876000   # population in thousands 

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
data_sir['I_est'] = data_sir['I_est'] 

print(data_sir.head())  # confirm values look correct

# 3. Plot S, I, R over time
plt.plot(data_sir['date'], data_sir['I_est'], marker='o')

plt.xlabel('Date')
plt.ylabel('Infections (Thousands)')
plt.title('Ebola Infection Estimate (I_est) Over Time')

# Rotate x-axis labels for readability
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Using the euler_SIR function defined earlier, we can simulate the SIR model over time.
from main_functions import euler_sir
from scipy.optimize import minimize


# Set up observed arrays
I_obs = data_sir['I_est'].values.astype(float)
t_obs = np.linspace(0, len(I_obs)-1, len(I_obs))

I0_obs = data_sir.iloc[0]['I_est']
R0_obs = 0.0
S0_obs = N- I0_obs - R0_obs
# Plug in guesses for gamma and beta, plot the model predictions against the data, and calculate SSE.

beta1 = 3 #random guess for beta
gamma1 = 2.9 #random guess for gamma
beta2 = 3.1 #alternative guesses
gamma2 = 3 #alternative guesses

S1, I1, R1 = euler_sir(beta1, gamma1, S0_obs, I0_obs, R0_obs, t_obs, N)
S2, I2, R2 = euler_sir(beta2, gamma2, S0_obs, I0_obs, R0_obs, t_obs, N)

plt.plot(t_obs, I1, label=f'beta={beta1},gamma={gamma1}', marker='x')
plt.plot(t_obs, I2, label=f'beta={beta2},gamma={gamma2}', marker='s')
plt.plot(t_obs, I_obs, 'o', label='Observed')

plt.legend()
plt.xlabel('days')
plt.ylabel('Infections (in thousands)')
plt.title('Demo: effect of beta/gamma on I(t)')
plt.show()

# Optimization: 
def SSE(params):
    beta, gamma = params
    S_pred, I_pred, R_pred = euler_sir(beta, gamma, S0_obs, I0_obs, R0_obs, t_obs, N)
    return np.mean((I_pred - I_obs)**2)

initial_guess = [0.2, 0.1]   # starting point
bounds = [(0.0001, 5), (0.0001, 2)]  # sensible parameter ranges

result = minimize(SSE, initial_guess, bounds=bounds, method='L-BFGS-B')

beta_hat, gamma_hat = result.x
print("\n=======================")
print(" OPTIMIZATION RESULTS")
print("=======================")
print("Fitted beta :", beta_hat)
print("Fitted gamma:", gamma_hat)
print("Minimum SSE :", result.fun)

S_fit, I_fit, R_fit = euler_sir(beta_hat, gamma_hat, S0_obs, I0_obs, R0_obs, t_obs, N)
plt.figure(figsize=(10,5))

plt.plot(t_obs, I_obs, 'o', label='Observed I(t)')
plt.plot(t_obs, I_fit, '-', label=f'Fitted Model β={beta_hat:.3f}, γ={gamma_hat:.3f}')
plt.plot(t_obs, I1, '--', label='Initial guess #1')
plt.plot(t_obs, I2, '--', label='Initial guess #2')

plt.xlabel('days')
plt.ylabel('Infections (in thousands)')
plt.title('Observed vs Fitted SIR Model')
plt.legend()
plt.show()

print("\nFinal fitted SSE:", np.mean((I_fit - I_obs)**2))

# Use an optimization routine to minimize SSE and find the best-fitting parameters.
print("SSE (beta1,gamma1):", np.mean((I1 - I_obs)**2))
print("SSE (beta2,gamma2):", np.mean((I2 - I_obs)**2))

#1. Split into first half of data
n_half = len(I_obs) // 2

I_obs_first = I_obs[:n_half]
t_first = np.linspace(0, len(I_obs_first)-1, len(I_obs_first))

I0_first = I_obs_first[0]
R0_first = 0.0
S0_first = N - I0_first - R0_first

# 2. Define the SSE function for first-half fitting
def SSE_first(params):
    beta, gamma = params
    
    # Reject invalid parameter regions
    if beta <= 0 or gamma <= 0:
        return 1e12

    try:
        S_pred, I_pred, R_pred = euler_sir(
            beta, gamma,
            S0_first, I0_first, R0_first,
            t_first, N
        )
    except:
        # If solver or code crashes: huge penalty
        return 1e12

    # If Euler produced NaN or Inf → penalize
    if np.any(np.isnan(I_pred)) or np.any(np.isinf(I_pred)):
        return 1e12
    
    # If model values explode unrealistically → penalize
    if np.max(I_pred) > 1e9:  
        return 1e12

    return np.mean((I_pred - I_obs_first)**2)

# 3. Optimize using first-half only
initial_guess = [0.2, 0.1]
bounds = [(0.0001, 5), (0.0001, 2)]

result_first = minimize(SSE_first, initial_guess, bounds=bounds, method='L-BFGS-B')

beta_half, gamma_half = result_first.x

print("\n==============================")
print(" FIT ON FIRST HALF OF DATA")
print("==============================")
print("beta_half  =", beta_half)
print("gamma_half =", gamma_half)
print("SSE_half   =", result_first.fun)

# 4. Simulate forward over FULL dataset using the half-fit parameters
S_full_fit, I_full_fit, R_full_fit = euler_sir(
    beta_half,
    gamma_half,
    S0_obs,     # initial S from full dataset
    I0_obs,     # initial I from full dataset
    R0_obs,
    t_obs,      # full time array
    N
)

# 5. Plot predicted full trajectory vs real data
plt.figure(figsize=(10,5))

plt.plot(t_obs, I_obs, 'o', label='Observed I(t)')
plt.plot(t_obs, I_full_fit, '-', label=f'Simulated using half-data fit β={beta_half:.3f}, γ={gamma_half:.3f}')

plt.axvline(n_half, color='gray', linestyle='--', label='Halfway Point')

plt.xlabel('Days')
plt.ylabel('I(t) (in thousands)')
plt.title('Using First-Half Fit to Predict Full SIR Dynamics')
plt.legend()
plt.show()
# Calculating SSE between model predictions and data on the SECOND HALF of the data.
# Second-half observed data
I_obs_second = I_obs[n_half:]
t_second = np.linspace(0, len(I_obs_second)-1, len(I_obs_second))

# Initial conditions for second half:
I0_second = I_obs_second[0]
R0_second = R0_obs              # still zero because your R_est=0 in dataset
S0_second = N - I0_second - R0_second

# Simulate SIR starting at beginning of second half
S_second_pred, I_second_pred, R_second_pred = euler_sir(
    beta_half,
    gamma_half,
    S0_second,
    I0_second,
    R0_second,
    t_second,
    N
)

# Compute SSE on second half ONLY
SSE_second_half = np.mean((I_second_pred - I_obs_second)**2)

print("\n==============================")
print(" SSE ON SECOND HALF OF DATA")
print("==============================")
print("β (from first half) =", beta_half)
print("γ (from first half) =", gamma_half)
print("SSE (second half)   =", SSE_second_half)
# Using scipy's solve_ivp function with the runge-kutta solver, re-implement the SIR model simulation, find optimal gamma and beta again, and plot the results.

from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Define SIR ODE system
# -----------------------------
def SIR_ODE(t, y, beta, gamma, N):
    S, I, R = y
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return [dS, dI, dR]


# -----------------------------
# 2. Wrapper to simulate SIR using solve_ivp
# -----------------------------
def simulate_sir_ivp(beta, gamma, S0, I0, R0, t_eval, N):
    sol = solve_ivp(
        fun=lambda t, y: SIR_ODE(t, y, beta, gamma, N),
        t_span=(t_eval[0], t_eval[-1]),
        y0=[S0, I0, R0],
        t_eval=t_eval,
        method="RK45"
    )
    return sol.y  # returns (S(t), I(t), R(t))


# -----------------------------
# 3. Define SSE function for optimization
# -----------------------------
def SSE_ivp(params):
    beta, gamma = params
    
    # Reject invalid parameter ranges
    if beta <= 0 or gamma <= 0:
        return 1e12

    try:
        S_pred, I_pred, R_pred = simulate_sir_ivp(
            beta, gamma, S0_obs, I0_obs, R0_obs, t_obs, N
        )
    except:
        return 1e12
    
    # Handle any numerical failures
    if np.any(np.isnan(I_pred)) or np.any(np.isinf(I_pred)):
        return 1e12

    return np.mean((I_pred - I_obs)**2)


# -----------------------------
# 4. Optimize β and γ using RK45 solver
# -----------------------------
initial_guess = [0.2, 0.1]
bounds = [(0.0001, 5), (0.0001, 2)]

result_ivp = minimize(SSE_ivp, initial_guess, bounds=bounds, method='L-BFGS-B')

beta_ivp, gamma_ivp = result_ivp.x

print("\n========================================")
print("  OPTIMIZATION RESULTS USING solve_ivp")
print("========================================")
print("Fitted beta (RK45):  ", beta_ivp)
print("Fitted gamma (RK45): ", gamma_ivp)
print("Minimum SSE (RK45):  ", result_ivp.fun)


# -----------------------------
# 5. Simulate with fitted RK45 parameters
# -----------------------------
S_ivp, I_ivp, R_ivp = simulate_sir_ivp(
    beta_ivp, gamma_ivp, S0_obs, I0_obs, R0_obs, t_obs, N
)

# -----------------------------
# 6. Plot Results
# -----------------------------
plt.figure(figsize=(10,5))

plt.plot(t_obs, I_obs, 'o', label='Observed I(t)')
plt.plot(t_obs, I_ivp, '-', label=f'RK45 Fit: β={beta_ivp:.3f}, γ={gamma_ivp:.3f}')

plt.xlabel("Days")
plt.ylabel("Infections (Thousands)")
plt.title("SIR Model Fit Using solve_ivp (Runge–Kutta)")
plt.legend()
plt.show()
# SSE comparison between Euler's method and RK4 (solve_ivp) on the SECOND HALF of the data.
# -----------------------------
# 1. Define second-half data
# -----------------------------
I_obs_second = I_obs[n_half:]
t_second = np.linspace(0, len(I_obs_second)-1, len(I_obs_second))

# Initial conditions for second half
I0_second = I_obs_second[0]
R0_second = R0_obs            # still 0
S0_second = N - I0_second - R0_second

# -----------------------------
# 2. Euler prediction on second half
# -----------------------------
S_euler_second, I_euler_second, R_euler_second = euler_sir(
    beta_half, gamma_half,
    S0_second, I0_second, R0_second,
    t_second, N
)

SSE_euler_second = np.mean((I_euler_second - I_obs_second)**2)

# -----------------------------
# 3. RK45 prediction on second half
# -----------------------------
S_rk45_second, I_rk45_second, R_rk45_second = simulate_sir_ivp(
    beta_ivp, gamma_ivp,
    S0_second, I0_second, R0_second,
    t_second, N
)

SSE_rk45_second = np.mean((I_rk45_second - I_obs_second)**2)

# -----------------------------
# 4. Print comparison
# -----------------------------
print("\n========================================")
print(" SSE ON SECOND HALF OF DATA")
print("========================================")
print(f"Euler SSE (first-half fit): {SSE_euler_second:.4f}")
print(f"RK45 SSE (full-data fit)   : {SSE_rk45_second:.4f}")

# -----------------------------
# 5. Optional plot comparison
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(t_second, I_obs_second, 'o', label='Observed')
plt.plot(t_second, I_euler_second, '--', label='Euler Prediction (β_half, γ_half)')
plt.plot(t_second, I_rk45_second, '-', label='RK45 Prediction (β_ivp, γ_ivp)')

plt.axvline(0, color='gray', linestyle='--', alpha=0.5)  # start of second half
plt.xlabel('Days (Second Half)')
plt.ylabel('Infections (Thousands)')
plt.title('Comparison of Euler vs RK45 on Second Half')
plt.legend()
plt.show()

from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. SEIR ODE System
# ------------------------------------------------------------
def SEIR_ODE(t, y, beta, sigma, gamma, N):
    S, E, I, R = y
    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return [dS, dE, dI, dR]


# ------------------------------------------------------------
# 2. Simulation Wrapper Using solve_ivp
# ------------------------------------------------------------
def simulate_seir_ivp(beta, sigma, gamma, S0, E0, I0, R0, t_eval, N):
    sol = solve_ivp(
        fun=lambda t, y: SEIR_ODE(t, y, beta, sigma, gamma, N),
        t_span=(t_eval[0], t_eval[-1]),
        y0=[S0, E0, I0, R0],
        t_eval=t_eval,
        method='RK45',
        max_step=1.0
    )
    return sol.y  # returns (S(t), E(t), I(t), R(t))


# ------------------------------------------------------------
# 3. Define SSE for the SEIR Model
# ------------------------------------------------------------
def SSE_seir(params):
    beta, sigma, gamma = params
    
    # Reject invalid parameter regions
    if beta <= 0 or sigma <= 0 or gamma <= 0:
        return 1e12

    try:
        S_pred, E_pred, I_pred, R_pred = simulate_seir_ivp(
            beta, sigma, gamma,
            S0_obs, E0_est, I0_obs, R0_obs,
            t_obs, N
        )
    except:
        return 1e12

    if np.any(np.isnan(I_pred)):
        return 1e12
    
    return np.mean((I_pred - I_obs)**2)


# ------------------------------------------------------------
# 4. Initial Conditions for SEIR
# ------------------------------------------------------------
# You already have S0_obs, I0_obs, R0_obs from your previous code

# For E0, assume some small initial exposure level
E0_est = I0_obs * 2   # (or experiment with 1–5× I0)

print("Initial E0 chosen as:", E0_est)

# ------------------------------------------------------------
# 5. Fit Parameters (beta, sigma, gamma)
# ------------------------------------------------------------
initial_guess = [0.3, 0.2, 0.1]   # beta, sigma, gamma
bounds = [(0.0001, 5), (0.0001, 1), (0.0001, 1)]

result_seir = minimize(SSE_seir, initial_guess, bounds=bounds, method='L-BFGS-B')

beta_seir, sigma_seir, gamma_seir = result_seir.x

print("\n====================================")
print("       SEIR OPTIMIZATION RESULTS")
print("====================================")
print("Fitted beta  :", beta_seir)
print("Fitted sigma :", sigma_seir)
print("Fitted gamma :", gamma_seir)
print("Minimum SSE  :", result_seir.fun)

# ------------------------------------------------------------
# 6. Simulate SEIR with Fitted Parameters
# ------------------------------------------------------------
S_fit, E_fit, I_fit, R_fit = simulate_seir_ivp(
    beta_seir, sigma_seir, gamma_seir,
    S0_obs, E0_est, I0_obs, R0_obs,
    t_obs, N
)

# ------------------------------------------------------------
# 7. Plot Observed vs Fitted I(t)
# ------------------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(t_obs, I_obs, 'o', label='Observed I(t)')
plt.plot(t_obs, I_fit, '-', label=f'SEIR Fit β={beta_seir:.3f}, σ={sigma_seir:.3f}, γ={gamma_seir:.3f}')

plt.xlabel('Days')
plt.ylabel('Infections (Individuals)')
plt.title('SEIR Model Fit (solve_ivp / RK45)')
plt.legend()
plt.show()

print("\nFinal SSE (SEIR):", np.mean((I_fit - I_obs)**2))