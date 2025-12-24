import numpy as np
import time  # <--- Import necessario
from broyden import BroydenProblem
from banded_trig import BandedTrigonometric
from methods import NewtonMethods
from utils import analyze_convergence, plot_convergence

# --- CONFIGURAZIONE ---
#N = 100000  # 100 Milioni: attenzione alla RAM!
K_MAX = 200
TOL = 1e-4
SEED = 358616

# Setup Iniziale

np.random.seed(SEED)
h_values = [1e-4, 1e-8, 1e-12] #with k = 4,8,12
dynamic_h = [True, False] # True h = h * np.abs(x), False h=h
n_list = [100, 1000, 10000, 100000]
h=1e-5
#for h in h_values:
for N in n_list:
    for dynamic in dynamic_h:
        hypercube_bro = np.random.uniform(-2, 0, (5, N))
        hypercube_tri = np.random.uniform(0, 2, (5, N))
        #hypercube_tri = np.ones(N)
        '''for x in hypercube_bro:
            print(f"Problem Size: {N}")
            print(f"Is dynamic: {dynamic}")
            print(f"H value: {h}")
            print(f"Initial Cost Broyden: {BroydenProblem.func(x):.4e}")
            print("\n" + "="*50)
            print(f"--------- START: Truncated Newton for Broyden Tridiagonal ---------")
            # --- AVVIO TIMER ---
            start_time_tn = time.perf_counter()
            xk_tn, fxk_tn, gradxk_norm_tn, k_tn, hist_tn = NewtonMethods.truncated_newton(
                x, 
                BroydenProblem.func, 
                BroydenProblem.gradient, 
                BroydenProblem.hessian_sparse,
                alpha0=1.0,
                kmax=K_MAX,
                tolgrad=TOL,
                c1=1e-4, 
                rho=0.5, 
                btmax=50,
                dynamic=dynamic,
                h=h
            )
            end_time_tn = time.perf_counter()
            execution_time_tn = end_time_tn - start_time_tn
            print(f"DONE. Tempo impiegato (Truncated) for Broyden Tridiagonal: {execution_time_tn:.4f} secondi")
        '''
        for x in hypercube_tri:
            print("\n" + "="*50)
            print("\n" + "="*50)
            print(f"Problem Size: {N}")
            print(f"Is dynamic: {dynamic}")
            print(f"H value: {h}")
            print(f"Initial Cost BandedTrigonometric: {BandedTrigonometric.func(x):.4e}")
            print(f"--------- START: modified_newton_single_diag for BandedTrigonometric ---------")
            # --- AVVIO TIMER ---
            start_time_tn = time.perf_counter()
            xk_tn, fxk_tn, gradxk_norm_tn, k_tn, hist_tn = NewtonMethods.modified_newton_trig(
                x, 
                BandedTrigonometric.func, 
                BandedTrigonometric.gradient_fd, 
                BandedTrigonometric.hessian_fd,
                alpha0=1.0,
                kmax=K_MAX,
                tolgrad=TOL,
                c1=1e-4, 
                rho=0.5, 
                btmax=50,
                dynamic=dynamic,
                h=h
            )
            end_time_tn = time.perf_counter()
            execution_time_tn = end_time_tn - start_time_tn
            print(f"DONE. Tempo impiegato (modified_newton_single_diag) for BandedTrigonometric: {execution_time_tn:.4f} secondi")

            
        

            #analyze_convergence(hist_tn)
            #plot_convergence(hist_tn, "Truncated Newton for Broyden Tridiagonal") # Decommenta se vuoi il grafico

    