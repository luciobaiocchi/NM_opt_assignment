import numpy as np
import time  # <--- Import necessario
from broyden import BroydenProblem
from methods import NewtonMethods
from utils import analyze_convergence, plot_convergence

# --- CONFIGURAZIONE ---
N = 100000000  # 10 Milioni: attenzione alla RAM!
K_MAX = 200
TOL = 1e-8
X0_SEED = 358616

# Setup Iniziale
np.random.seed(X0_SEED)
x0 = np.random.uniform(-2, 0, N)

print(f"Problem Size: {N}")
print(f"Initial Cost: {BroydenProblem.func(x0):.4e}")
'''
# ==========================================
# 1. ESECUZIONE TRUNCATED NEWTON
# ==========================================
print("\n" + "="*50)
print(f"START: Truncated Newton...")

# --- AVVIO TIMER ---
start_time_tn = time.perf_counter()

xk_tn, fxk_tn, gradxk_norm_tn, k_tn, hist_tn = NewtonMethods.truncated_newton(
    x0, 
    BroydenProblem.func, 
    BroydenProblem.gradient, 
    BroydenProblem.hessian_sparse,
    alpha0=1.0,
    kmax=K_MAX,
    tolgrad=TOL,
    c1=1e-4, 
    rho=0.5, 
    btmax=50
)

# --- STOP TIMER ---
end_time_tn = time.perf_counter()
execution_time_tn = end_time_tn - start_time_tn

print(f"DONE. Tempo impiegato (Truncated): {execution_time_tn:.4f} secondi")
print("="*50)

# analyze_convergence(hist_tn)
plot_convergence(hist_tn, "Truncated Newton") # Decommenta se vuoi il grafico

'''
# ==========================================
# 2. ESECUZIONE MODIFIED NEWTON (BANDED)
# ==========================================
print("\n" + "="*50)
print(f"START: Modified Newton (Banded)...")

# --- AVVIO TIMER ---
start_time_mn = time.perf_counter()

xk_mn, fxk_mn, gradxk_norm_mn, k_mn, hist_mn = NewtonMethods.modified_newton_banded(
    x0, 
    BroydenProblem.func, 
    BroydenProblem.gradient, 
    BroydenProblem.hessian_sparse,
    alpha0=1.0,
    kmax=K_MAX,
    tolgrad=TOL,
    c1=1e-4, 
    rho=0.5, 
    btmax=50
)

# --- STOP TIMER ---
end_time_mn = time.perf_counter()
execution_time_mn = end_time_mn - start_time_mn

print(f"DONE. Tempo impiegato (Modified): {execution_time_mn:.4f} secondi")
print("="*50)

# analyze_convergence(hist_mn)
plot_convergence(hist_mn, "Modified Newton (Banded)") # Decommenta se vuoi il grafico

# --- CONFRONTO FINALE TEMPI ---
print("\n" + "*"*30)
print("RIEPILOGO TEMPI DI ESECUZIONE")
print("*"*30)
print(f"Truncated Newton:       {execution_time_tn:.4f} s")
print(f"Modified Newton Banded: {execution_time_mn:.4f} s")
print("*"*30)