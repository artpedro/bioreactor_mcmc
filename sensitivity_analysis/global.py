# %%
#!pip install joblib SALib tqdm
from joblib import Parallel, delayed
import numpy as np
from scipy.integrate import odeint
from SALib.sample import sobol
from tqdm import tqdm
import scipy

# %%
def enzymic_amox(t,y, 
kcat1,
kcat2,
Km1,
Km2,  
Tmax, 
Ken,  
kAB,  
kAN,  
kAOH, 
kNH):
    FAB = 0
    FNH = 0 
    
    CAB = y[0]
    CAN = y[1]
    CNH = y[2]
    CAOH = y[3]

    Cez = 1

    # Consumo de ester
    VAB = (kcat1*CAB*Cez)/((Km1*(1 + (CAN/kAN) + (CAOH/kAOH))) + CAB)
    
    # Hidrolise de amoxicilina
    VAN = (kcat2*CAN*Cez)/((Km2*(1 + (CAB/kAB) + (CNH/kNH) + (CAOH/kAOH))) + CAN)
    
    # Enzima saturada com 6-apa
    X   = CNH/(Ken + CNH)
    
    # Sintese enzimatica
    VS  = VAB*Tmax*X

    # Hidrolise de ester
    Vh1 = (VAB - VS) 

    dy = np.zeros(4)

    # C. ester
    dy[0] = ((-(VS - VAN) - (Vh1 + VAN)) + FAB) 
    
    # C. amox
    dy[1] = (VS - VAN)                         
    
    # C. 6-apa
    dy[2] = (-(VS - VAN) + FNH)                
    
    # C. POHPG
    dy[3] =  (Vh1 + VAN)
    
    return np.array(dy)      

# %%
def ode15s_amox(P, CI, t):
    return scipy.integrate.solve_ivp(enzymic_amox,t_span=(t[0],t[-1]),t_eval=t,y0=CI,method='RK45',args=P).y.T

# %%
def model_output(params):
    kcat1, kcat2, Km1, Km2, Tmax, Ken, kAB, kAN, kAOH, kNH, CAB, CNH = params
    initial_state = [CAB, 0.0, CNH, 0.0]
    sol = ode15s_amox(params[:-2],initial_state,t_eval)
    return sol  # Transpose to get time points as rows and variables as columns


# %%
kcat1        = 0.178 #Constante catalítica do consumo do éster (mmol/i.u. per min)
 
kcat2        = 0.327 #Constante catalítica da hidrólise da amoxicilina (mmol/i.u. per min)
 
Km1          = 7.905 #Constante de Michaelis-Menten ou constante de afinidade para consumo do éster(mM) 
 
Km2          = 12.509 #Constante de Michaelis-Menten ou constante de afinidade para hidrólise da amoxicilina(mM)
 
Tmax         = 0.606 #Taxa de conversão máxima do complexo acil-enzima-núcleo em produto
 
Ken          = 14.350 #Constante de adsorção do 6-APA
 
kAB          = 3.78 #Constante de inibição do éster (POHPGME)(mM)
 
kAN          = 9.174 #Constante de inibição da amoxicilina (mM)
 
kAOH         = 10.907 #Constante de inibição do POHPG, produto da hidr�lise da amoxicilina (mM)
 
kNH          = 62.044 #Constante de inibição do 6-APA

P = [kcat1,kcat2,Km1,Km2,Tmax,Ken,kAB,kAN,kAOH,kNH]

# Function to calculate bounds for each parameter based on P
def calculate_bounds(P, factor=2.6):
    bounds = []
    for i, param in enumerate(P):
        if i == 4:  # Tmax (special case)
            bounds.append([0, 1])
        else:
            lower_bound = param * (1 - factor)
            upper_bound = param * (1 + factor)
            if lower_bound < 0:
                lower_bound = 0.01
            bounds.append([lower_bound, upper_bound])

    return bounds

# Calculate bounds for the parameters in P
param_bounds = calculate_bounds(P)
print(param_bounds)
# Keep the bounds for initial concentrations unchanged
initial_conc_bounds = [
    [30, 80],  # CAB
    [5, 100],  # CNH
]

# Define the problem with dynamic bounds
problem = {
    'num_vars': 12,
    'names': [
        'kcat1',
        'kcat2',
        'Km1',
        'Km2',  
        'Tmax', 
        'Ken',  
        'kAB',  
        'kAN',  
        'kAOH', 
        'kNH'
    ],
    'bounds': np.array(param_bounds + initial_conc_bounds)
}
print(problem['bounds'].shape)

# %%
# Generate samples using Sobol sequence
param_values = sobol.sample(problem, 2048) 
t_eval = np.linspace(0, 300, 101)

print("N samples: ",param_values.shape)

# %%
# Use parallel processing to evaluate the model output for all parameter sets
num_cores = 6  # Use all available cores
results = Parallel(n_jobs=num_cores)(delayed(model_output)(params) for params in param_values)

# %%
results_clean = []
for i in results:
    if i.shape[0] == 101:
        results_clean.append(i)
    else:
        results_clean.append(np.zeros((101,4)))
for i in results_clean:
    if i.shape[1] != 4:  
        print(i.shape)

# %%
from SALib.analyze import sobol  # Correctly import analyze
import numpy as np

Y = np.array(results_clean)
print(f"Shape of Y: {Y.shape}")
Si_list = []
for i in range(Y.shape[1]):  # Iterate over time points
    for j in range(Y.shape[2]):  # Iterate over each output variable (4 outputs)
        print(f"Processing time point {i}/{Y.shape[1]}, output {j}/{Y.shape[2]}", end='\r')

        # Perform Sobol analysis on the output variable j at time point i
        Si = sobol.analyze(problem, Y[:, i, j], calc_second_order=True, print_to_console=False)
        
        # Store the result for each time point and output variable
        Si_list.append(Si)

        # Optionally print the first-order sensitivity indices for debugging
        print(f"S1 at time point {i}, output {j}: {Si['S1']}")


# %%
import pickle

# Store the Si_list to a file
with open('si_list_2048_IC.pkl', 'wb') as f:
    pickle.dump(Si_list, f)

# %%
import os
import matplotlib.pyplot as plt
n_timepoints = Y.shape[1]
n_outputs = Y.shape[2]
output_dir = "sobol_plots"
os.makedirs(output_dir, exist_ok=True)

# Generate and save the plots for each output
for j in range(n_outputs):  # Loop over each output variable
    S1_values = np.array([Si_list[i]['S1'] for i in range(j, len(Si_list), n_outputs)])
    ST_values = np.array([Si_list[i]['ST'] for i in range(j, len(Si_list), n_outputs)])

    # Plot S1 (First-order sensitivity indices) for each parameter across time points
    plt.figure(figsize=(12, 6))
    for k in range(S1_values.shape[1]):  # Loop over parameters
        plt.plot(range(n_timepoints), S1_values[:, k], label=f'S1 - {problem["names"][k]}')
    plt.title(f'First-order Sobol indices (S1) for Output {j}')
    plt.xlabel('Time Point')
    plt.ylabel('Sobol Index (S1)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'output_{j}_S1_2048.png'))
    plt.close()

    # Plot ST (Total-order sensitivity indices) for each parameter across time points
    plt.figure(figsize=(12, 6))
    for k in range(ST_values.shape[1]):  # Loop over parameters
        plt.plot(range(n_timepoints), ST_values[:, k], label=f'ST - {problem["names"][k]}')
    plt.title(f'Total-order Sobol indices (ST) for Output {j}')
    plt.xlabel('Time Point')
    plt.ylabel('Sobol Index (ST)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'output_{j}_ST.png'))
    plt.close()

output_dir  # Return the output directory where plots are saved


