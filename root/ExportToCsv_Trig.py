import numpy as np
import time
import csv

from banded_trig import BandedTrigonometric
from methods import NewtonMethods
from utils import analyze_convergence1, plot_convergence

# --- CONFIGURAZIONE ---
K_MAX = 200
TOL = 1e-4
SEED = 358616

np.random.seed(SEED)

h_values = [1e-4, 1e-8, 1e-12]
dynamic_h = [True, False]
n_list = [2, 1000, 10000, 100000]

# --- DEFINIZIONE FIELDNAMES ---
fieldnames_full = [
    "Point ID",
    "n",
    "h",
    "is_dynamic",
    "success",
    "k",
    "grad_norm_final",
    "fx_final",
    "convergence_rate",
    "time"
]

fieldnames_simple = [
    "Point ID",
    "n",
    "success",
    "k",
    "grad_norm_final",
    "fx_final",
    "convergence_rate",
    "time"
]

def process_batch(writer, method, gradient_fn, hessian_fn, N, h, dynamic, include_params):
    # 5 punti random + punto deterministico x0 = ones
    x0 = np.ones(N)
    hypercube_random = np.random.uniform(0, 2, (5, N))
    hypercube_trig = np.vstack([x0, hypercube_random])

    point_id = 0
    for x in hypercube_trig:
        # --- AVVIO TIMER ---
        start_time = time.perf_counter()

        xk, fxk, gradxk_norm, k, hist = method(
            x,
            BandedTrigonometric.func,
            gradient_fn,
            hessian_fn,
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
        execution_time = time.perf_counter() - start_time
        convergence_rate = analyze_convergence1(hist)
        
        success = (
            "yes"
            if (gradxk_norm is not None and gradxk_norm <= TOL and k < K_MAX)
            else "no"
        )
        
        row = {
            "Point ID": point_id,
            "n": N,
            "success": success,
            "k": k,
            "grad_norm_final": f"{gradxk_norm:.5g}" if gradxk_norm is not None else "",
            "fx_final": f"{fxk:.5g}",
            "convergence_rate": f"{convergence_rate:.5g}" if convergence_rate is not None else "",
            "time": f"{execution_time:.5g}"
        }

        if include_params:
            row["h"] = f"{h:.5g}"
            row["is_dynamic"] = dynamic

        writer.writerow(row)
        point_id += 1

def run_experiment(log, filename, method, gradient_fn, hessian_fn, fieldnames, use_h_dynamic_loops=True):
    print("\n" + "="*100)
    print(f"=========  {filename}  ==========")
    print(f"=========  {log}  ==========")
    print("="*100 + "\n")

    with open(filename, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        if use_h_dynamic_loops:
            for h in h_values:
                for N in n_list:
                    for dynamic in dynamic_h:
                        print('\n')
                        print(f"==================  {h, N, dynamic}  ==================")
                        print('\n')
                        process_batch(writer, method, gradient_fn, hessian_fn, N, h, dynamic, True)
        else:
            for N in n_list:
                process_batch(writer, method, gradient_fn, hessian_fn, N, None, None, False)

    print(f"\nCSV salvato correttamente come: {filename}")


# --- ESECUZIONE ---

# 1. TN, Exact Gradient, Hessian with Jacobian
run_experiment(
    " 1. TN, Exact Gradient, Hessian with Jacobian",
    "./results/results_tn_trig_exact_gradient.csv",
    NewtonMethods.truncated_newton,
    BandedTrigonometric.gradient_exact, 
    BandedTrigonometric.hessian_with_jacobian,
    fieldnames_full
)

# 2. MN, Exact Gradient, Hessian with Jacobian
run_experiment(
    "2. MN, Exact Gradient, Hessian with Jacobian",
    "./results/results_mn_trig_exact_gradient.csv",
    NewtonMethods.modified_newton_trig,
    BandedTrigonometric.gradient_exact,
    BandedTrigonometric.hessian_with_jacobian,
    fieldnames_full
)

# 3. TN, All Approx (FD)
run_experiment(
    "3. TN, All Approx (FD)",
    "./results/results_tn_trig_all_aprox.csv",
    NewtonMethods.truncated_newton,
    BandedTrigonometric.gradient_fd,
    BandedTrigonometric.hessian_fd,
    fieldnames_full
)

# 4. MN, All Approx (FD)
run_experiment(
    "4. MN, All Approx (FD)",
    "./results/results_mn_trig_all_aprox.csv",
    NewtonMethods.modified_newton_trig,
    BandedTrigonometric.gradient_fd,
    BandedTrigonometric.hessian_fd,
    fieldnames_full
)

# 5. TN, All Exact, No h/dynamic loops
run_experiment(
    "5. TN, All Exact, No h/dynamic loops",
    "./results/results_tn_trig_all_exact.csv",
    NewtonMethods.truncated_newton,
    BandedTrigonometric.gradient_exact,
    BandedTrigonometric.hessian_exact,
    fieldnames_simple,
    use_h_dynamic_loops=False
)

# 6. MN, All Exact, No h/dynamic loops
run_experiment(
    "6. MN, All Exact, No h/dynamic loops",
    "./results/results_mn_trig_all_exact.csv",
    NewtonMethods.modified_newton_trig,
    BandedTrigonometric.gradient_exact,
    BandedTrigonometric.hessian_exact,
    fieldnames_simple,
    use_h_dynamic_loops=False
)
