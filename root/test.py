import numpy as np
import time
from broyden import BroydenProblem
from banded_trig import BandedTrigonometric
from methods import NewtonMethods
from utils import analyze_convergence, plot_convergence

# --- CONFIGURAZIONE GLOBALE ---
N = 100000 
K_MAX = 200
SEED = 358616

def get_problem_configs(n_size):
    """
    Definisce la configurazione dei problemi (Funzione, Gradiente, Hessiana, x0, Tolleranza specifica)
    """
    np.random.seed(SEED)
    
    # Setup punti iniziali
    x0_bro = np.full(n_size, -1.0)
    x0_trig = np.ones(n_size)

    return [
        {
            "name": "Broyden Problem",
            "x0": x0_bro,
            "func": BroydenProblem.func,
            "grad": BroydenProblem.gradient,
            "hess": BroydenProblem.hessian_sparse, # Usa la sparsa per Broyden
            "tol": 1e-8
        },
        {
            "name": "Banded Trigonometric",
            "x0": x0_trig,
            "func": BandedTrigonometric.func,
            "grad": BandedTrigonometric.gradient,
            "hess": BandedTrigonometric.hessian,
            "tol": 1e-4 # Tolleranza specifica richiesta per questo problema
        }
    ]

def get_method_configs():
    """
    Definisce la lista dei metodi da testare.
    """
    return [
        {
            "name": "Truncated Newton",
            "solver": NewtonMethods.truncated_newton,
            "params": {"c1": 1e-4, "rho": 0.5, "btmax": 50}
        },
        {
            "name": "Modified Newton (Banded)",
            "solver": NewtonMethods.modified_newton_banded,
            # Nota: Modified Newton a volte richiede c1 diverso, qui uniformiamo o specifichiamo
            "params": {"c1": 1e-4, "rho": 0.5, "btmax": 50} 
        }
    ]

def run_single_test(problem, method, global_kmax):
    """
    Esegue un singolo test combinando un problema e un metodo.
    """
    print(f"\n{'='*60}")
    print(f"TEST: {method['name']} su {problem['name']}")
    print(f"{'='*60}")
    
    # Reset seed opzionale se i metodi sono stocastici, altrimenti non serve
    # np.random.seed(SEED) 

    # Parametri completi per il solver
    solver_args = {
        "x0": problem["x0"].copy(), # Importante: copiare x0 per non modificarlo per i test successivi
        "f": problem["func"],
        "gradf": problem["grad"],
        "hessf": problem["hess"],
        "alpha0": 1.0,
        "kmax": global_kmax,
        "tolgrad": problem["tol"],
        **method["params"] # Scompatta i parametri specifici del metodo
    }

    # --- AVVIO TIMER ---
    start_time = time.perf_counter()
    
    # Chiamata dinamica al solver
    xk, fxk, gradxk_norm, k, hist = method["solver"](**solver_args)

    # --- STOP TIMER ---
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"-> DONE. Iterazioni: {k}, Tempo: {elapsed_time:.4f} s")
    print(f"-> Costo Finale: {fxk:.4e}, Grad Norm: {gradxk_norm:.4e}")

    # Analisi e Plot
    analyze_convergence(hist)
    # Titolo grafico dinamico
    plot_title = f"{method['name']} - {problem['name']}"
    # plot_convergence(hist, plot_title) # Decommenta per mostrare i grafici

    return {
        "problem": problem["name"],
        "method": method["name"],
        "time": elapsed_time,
        "iter": k,
        "final_grad": gradxk_norm
    }

def print_summary(results):
    """
    Stampa una tabella riassuntiva dei tempi.
    """
    print("\n" + "*"*80)
    print(f"{'PROBLEM':<25} | {'METHOD':<25} | {'TIME (s)':<10} | {'ITERS':<5}")
    print("*"*80)
    for res in results:
        print(f"{res['problem']:<25} | {res['method']:<25} | {res['time']:<10.4f} | {res['iter']:<5}")
    print("*"*80)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    print(f"Generazione Dati (N={N})...")
    problems = get_problem_configs(N)
    methods = get_method_configs()
    
    results_summary = []

    # --- CICLO PRINCIPALE ---
    # Itera su ogni problema
    for prob in problems:
        print(f"\n>>> Processing Problem: {prob['name']} (Initial Cost: {prob['func'](prob['x0']):.4e})")
        
        # Itera su ogni metodo per quel problema
        for meth in methods:
            
            # Esecuzione Test
            result = run_single_test(prob, meth, K_MAX)
            results_summary.append(result)

    # Stampa tabella finale
    print_summary(results_summary)