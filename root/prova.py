import numpy as np
import time  # <--- Import necessario
from broyden import BroydenProblem
from banded_trig import BandedTrigonometric
from methods import NewtonMethods
from utils import analyze_convergence, plot_convergence

np.seterr(divide='ignore', invalid='ignore', over='ignore')


# --- CONFIGURAZIONE ---
N = 100000  # 100 Milioni: attenzione alla RAM!
K_MAX = 200
TOL = 1e-8
SEED = 358616

np.random.seed(SEED)
x0_bro = np.random.uniform(-2, 0, N) 

# Parametri richiesti dall'assignment
k_values = [4, 8, 12]
strategies = ["fixed", "variable"]

print(f"Problem Size: {N}")

# Wrapper per iniettare l'h corretto nel metodo hessian_sparse
# NewtonMethods.truncated_newton si aspetta una funzione hess(x), quindi usiamo una lambda o partial

for strat in strategies:
    for k in k_values:
        print(f"\n" + "="*60)
        print(f"TESTING: Strategy={strat.upper()}, k={k} (h approx 10^-{k})")
        print("="*60)

        # Definiamo la funzione Hessiana specifica per questa configurazione
        def hessian_wrapper(x):
            if strat == "fixed":
                # Caso 1: h fisso
                h_val = 10**(-k)
                return BroydenProblem.hessian_with_jacobian(x, h_arg=h_val)
            else:
                # Caso 2: h variabile h_i = 10^-k * |x_i|
                # Aggiungiamo un epsilon per evitare h=0 se x=0
                h_vec = (10**(-k)) * np.abs(x)
                # Fallback minimo per stabilità numerica (opzionale ma consigliato)
                h_vec[h_vec < 1e-16] = 10**(-k) 
                return BroydenProblem.hessian_with_jacobian(x, h_arg=h_vec)

        # Esecuzione (Esempio con Truncated Newton)
        # Nota: Passiamo hessian_wrapper come parametro hessian
        start_time_tn = time.perf_counter()

        xk, fx, gnorm, iters, hist = NewtonMethods.truncated_newton(
            x0_bro, 
            BroydenProblem.func, 
            BroydenProblem.gradient_exact, 
            hessian_wrapper,  # <--- Passiamo il wrapper configurato
            alpha0=1.0, 
            kmax=K_MAX, 
            tolgrad=TOL, 
            c1=1e-4, 
            rho=0.5, 
            btmax=50
        )
        
        end_time_tn = time.perf_counter()
        execution_time_tn = end_time_tn - start_time_tn

        print(f"Result: Iter={iters}, ||g||={gnorm:.4e}, Time={execution_time_tn:.4f}s")
        # Salva risultati per tabelle/grafici qui...