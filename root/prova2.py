import numpy as np
import time  # <--- Import necessario
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
print("\n" + "="*100)
print("START"* 20)
print("\n" + "="*100)

np.random.seed(SEED)
h_values = [1e-4, 1e-8, 1e-12] #with k = 4,8,12
dynamic_h = [True, False] # True h = h * np.abs(x), False h=h
n_list = [2, 100, 1000, 10000, 100000]

#h=1e-4
for h in h_values:
    for N in n_list:
        for dynamic in dynamic_h:
            
            #x0_broyden = -np.ones(N)
            hypercube_bro = np.random.uniform(-2, 0, (5, N))
            #hypercube_bro = np.append(hypercube_bro,x0_broyden)
            
            hypercube_tri = np.random.uniform(0, 2, (5, N))
            
            point= 1
            for x in hypercube_bro:
                print("\n" + "="*50)
                print(f"Problem Size: {N}")
                print(f"Is h dynamic?: {dynamic}")
                print(f"H value: {h}")
                print(f"Starting point: point {point}")
                print(f"Initial Cost Broyden: {BroydenProblem.func(x):.4e}")
                # ==========================================
                # 1. ESECUZIONE TRUNCATED NEWTON per BROYDEN
                # ==========================================
                #print( "="*50)
                print(f"{'-'*15}START: Truncated Newton for Broyden Tridiagonal...{'-'*15}")
                # --- AVVIO TIMER ---
                start_time_tn = time.perf_counter()
                xk_tn, fxk_tn, gradxk_norm_tn, k_tn, hist_tn = NewtonMethods.truncated_newton(
                    x, 
                    BroydenProblem.func, 
                    BroydenProblem.exact_gradient, 
                    BroydenProblem.hessian_with_jacobian,
                    alpha0=1.0,
                    kmax=K_MAX,
                    tolgrad=TOL,
                    c1=1e-4, 
                    rho=0.5, 
                    btmax=50,
                    dynamic=dynamic,
                    h=h
                )
                # --- STOP TIMER ---
                end_time_tn = time.perf_counter()
                execution_time_tn = end_time_tn - start_time_tn
                point+=1
                print(f"DONE. Tempo impiegato (Truncated) for Broyden Tridiagonal: {execution_time_tn:.4f} secondi")
                print(f"Final Norm of the gradient: {gradxk_norm_tn}")
                print(f"Final value of f(x): {fxk_tn}")
                print(f"Final mean convergence rate: {analyze_convergence(hist_tn)}")
                print("="*50)
                #analyze_convergence(hist_tn)
                #plot_convergence(hist_tn, "Truncated Newton for Broyden Tridiagonal") # Decommenta se vuoi il grafico
