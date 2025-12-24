import numpy as np
import time
from broyden import BroydenProblem
from banded_trig import BandedTrigonometric
from methods import NewtonMethods
from utils import analyze_convergence, plot_convergence

# --- CONFIGURAZIONE ---
N = 100000  # 100 Milioni: attenzione alla RAM!
K_MAX = 200
TOL = 1e-8
SEED = 358616

# Setup Iniziale

np.random.seed(SEED)
#x0_bro = np.full(N,-1)
#x0_bro = np.full(N,-1) # punto di partenza per broyden con tutti -1
#x0_trig = np.random.uniform(0, 2, N)
x0_bro = np.random.uniform(-2, 0, N) 
x0_trig = np.random.uniform(0, 2, N) # punto di partenza per trigonometric con tutti 1

hypercube_bro = np.random.uniform(-2, 0, (5, N))
hypercube_tri = np.random.uniform(0, 2, (5, N))
n_list = [2, 100, 1000, 10000, 100000]
print(f"Problem Size: {N}")
print(f"Initial Cost Broyden: {BroydenProblem.func(x0_bro):.4e}")
print(f"Initial Cost Banded Trigonometric: {BandedTrigonometric.func(x0_trig):.4e}")

'''
# ==========================================
# 1. ESECUZIONE TRUNCATED NEWTON per BROYDEN
# ==========================================
print("\n" + "="*50)
print(f"START: Truncated Newton for Broyden Tridiagonal...")

# --- AVVIO TIMER ---
start_time_tn = time.perf_counter()

xk_tn, fxk_tn, gradxk_norm_tn, k_tn, hist_tn = NewtonMethods.truncated_newton(
    x0_bro, 
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

print(f"DONE. Tempo impiegato (Truncated) for Broyden Tridiagonal: {execution_time_tn:.4f} secondi")
print("="*50)

analyze_convergence(hist_tn)
plot_convergence(hist_tn, "Truncated Newton for Broyden Tridiagonal") # Decommenta se vuoi il grafico



# ==========================================
# 2. ESECUZIONE MODIFIED NEWTON (BANDED) per BROYDEN
# ==========================================
print("\n" + "="*50)
print(f"START: Modified Newton for Broyden Tridiagonal...")

# --- AVVIO TIMER ---
start_time_mn = time.perf_counter()

xk_mn, fxk_mn, gradxk_norm_mn, k_mn, hist_mn = NewtonMethods.modified_newton_banded(
    x0_bro, 
    BroydenProblem.func, 
    BroydenProblem.gradient, 
    BroydenProblem.hessian_sparse,
    alpha0=1.0,
    kmax=K_MAX,
    tolgrad=TOL,
    c1=1e-5, 
    rho=0.5, 
    btmax=50
)

# --- STOP TIMER ---
end_time_mn = time.perf_counter()
execution_time_mn = end_time_mn - start_time_mn

print(f"DONE. Tempo impiegato (Modified) for Broyden Tridiagonal: {execution_time_mn:.4f} secondi")
print("="*50)

analyze_convergence(hist_mn)
plot_convergence(hist_mn, "Modified Newton for Broyden Tridiagonal") # Decommenta se vuoi il grafico

'''

# ==========================================
# 3. ESECUZIONE TRUNCATED NEWTON per BANDED TRIGONOMETRIC
# ==========================================
print("\n" + "="*50)
print(f"START: Truncated Newton for Banded Trigonometric...")

# --- AVVIO TIMER ---
start_time_tn_banded = time.perf_counter()

xk_tn, fxk_tn, gradxk_norm_tn, k_tn, hist_tn_banded = NewtonMethods.truncated_newton(
    x0_trig, 
    BandedTrigonometric.func, 
    BandedTrigonometric.gradient_exact, 
    BandedTrigonometric.hessian_with_jacobian,
    alpha0=1.0,
    kmax=K_MAX,
    tolgrad=TOL,
    c1=1e-4, 
    rho=0.5, 
    btmax=50,
    dynamic=True,
    h=1e-4
)

# --- STOP TIMER ---
end_time_tn_banded = time.perf_counter()
execution_time_tn_banded = end_time_tn_banded - start_time_tn_banded

print(f"DONE. Tempo impiegato (Truncated) for Banded Trigonometric: {execution_time_tn_banded:.4f} secondi")
print("="*50)

analyze_convergence(hist_tn_banded)
plot_convergence(hist_tn_banded, "Truncated Newton for Banded Trigonometric") # Decommenta se vuoi il grafico


# ==========================================
# 4. ESECUZIONE TRUNCATED NEWTON per Trig
# ==========================================
print("\n" + "="*50)
print(f"START: Modified Newton for Banded Trigonometric...")
print(BandedTrigonometric.func(x0_trig))

# --- AVVIO TIMER ---
start_time_mn_banded = time.perf_counter()

xk_tn, fxk_tn, gradxk_norm_tn, k_tn, hist_mn_banded = NewtonMethods.modified_newton_bro(
    x0_trig, 
    BandedTrigonometric.func, 
    BandedTrigonometric.gradient_fd, 
    BandedTrigonometric.hessian_fd,
    alpha0=1.0,
    kmax=K_MAX,
    tolgrad=1e-6,
    c1=1e-4, 
    rho=0.5, 
    btmax=50
)

# --- STOP TIMER ---
end_time_mn_banded = time.perf_counter()
execution_time_mn_banded = end_time_mn_banded - start_time_mn_banded

print(f"DONE. Tempo impiegato (Modified) for Banded Trigonometric: {execution_time_mn_banded:.4f} secondi")
print("="*50)

analyze_convergence(hist_mn_banded)
plot_convergence(hist_mn_banded, "Modified Newton for Banded Trigonometric") # Decommenta se vuoi il grafico

'''

# --- CONFRONTO FINALE TEMPI ---
print("\n" + "*"*30)
print("RIEPILOGO TEMPI DI ESECUZIONE")
print("*"*30)
print(f"Truncated Newton Broyden:            {execution_time_tn:.4f} s")
print(f"Modified Newton Broyden:             {execution_time_mn:.4f} s")
print(f"Modified Newton Banded Trigonometric:{execution_time_mn_banded:.4f} s")
print(f"Truncated Newton Banded Trigonometric:{execution_time_tn_banded:.4f} s")

print("*"*30)

'''