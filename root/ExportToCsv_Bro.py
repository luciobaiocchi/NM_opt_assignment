import numpy as np
import time
import csv
from multiprocessing import Process
from broyden import BroydenProblem
from methods import NewtonMethods
from utils import analyze_convergence1, plot_convergence

# --- CONFIGURAZIONE ---
K_MAX = 200
TOL = 1e-6
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

def process_batch(writer, method, gradient_fn, hessian_fn, N, h, dynamic, include_params, convergence_tail=None):
    # 5 punti random + punto deterministico x0 = -ones
    x0 = -np.ones(N)
    hypercube_random = np.random.uniform(-2, 0, (5, N))
    hypercube_bro = np.vstack([x0, hypercube_random])

    point_id = 0
    for x in hypercube_bro:
        # --- AVVIO TIMER ---
        start_time = time.perf_counter()

        xk, fxk, gradxk_norm, k, hist = method(
            x,
            BroydenProblem.func,
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
        
        if convergence_tail is not None:
            convergence_rate = analyze_convergence1(hist, tail=convergence_tail)
        else:
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

def run_experiment(log, filename, method, gradient_fn, hessian_fn, fieldnames, use_h_dynamic_loops=True, convergence_tail=None):
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
                        process_batch(writer, method, gradient_fn, hessian_fn, N, h, dynamic, True, convergence_tail)
        else:
            for N in n_list:
                process_batch(writer, method, gradient_fn, hessian_fn, N, None, None, False, convergence_tail)

    print(f"\nCSV salvato correttamente come: {filename}")


# --- ESECUZIONE ---

if __name__ == "__main__":
    processes = []

    # 1. TN, Exact Gradient, Hessian with Jacobian
    p1 = Process(target=run_experiment, args=(
        " 1. TN, Exact Gradient, Hessian with Jacobian",
        "./results/results_tn_broyden_exact_gradient.csv",
        NewtonMethods.truncated_newton,
        BroydenProblem.gradient_exact, 
        BroydenProblem.hessian_with_jacobian,
        fieldnames_full,
        True,
        3 # convergence_tail
    ))
    processes.append(p1)

    # 2. MN, Exact Gradient, Hessian with Jacobian
    p2 = Process(target=run_experiment, args=(
        "2. MN, Exact Gradient, Hessian with Jacobian",
        "./results/results_mn_broyden_exact_gradient.csv",
        NewtonMethods.modified_newton_bro,
        BroydenProblem.gradient_exact,
        BroydenProblem.hessian_with_jacobian,
        fieldnames_full
    ))
    processes.append(p2)

    # 3. TN, All Approx (FD)
    p3 = Process(target=run_experiment, args=(
        "3. TN, All Approx (FD)",
        "./results/results_tn_broyden_all_aprox.csv",
        NewtonMethods.truncated_newton,
        BroydenProblem.gradient_fd,
        BroydenProblem.hessian_fd,
        fieldnames_full
    ))
    processes.append(p3)

    # 4. MN, All Approx (FD)
    p4 = Process(target=run_experiment, args=(
        "4. MN, All Approx (FD)",
        "./results/results_mn_broyden_all_aprox.csv",
        NewtonMethods.modified_newton_bro,
        BroydenProblem.gradient_fd,
        BroydenProblem.hessian_fd,
        fieldnames_full
    ))
    processes.append(p4)

    # 5. TN, All Exact, No h/dynamic loops
    p5 = Process(target=run_experiment, args=(
        "5. TN, All Exact, No h/dynamic loops",
        "./results/results_tn_broyden_all_exact.csv",
        NewtonMethods.truncated_newton,
        BroydenProblem.gradient_exact,
        BroydenProblem.hessian_exact,
        fieldnames_simple,
        False
    ))
    processes.append(p5)

    # 6. MN, All Exact, No h/dynamic loops
    p6 = Process(target=run_experiment, args=(
        "6. MN, All Exact, No h/dynamic loops",
        "./results/results_mn_broyden_all_exact.csv",
        NewtonMethods.modified_newton_bro,
        BroydenProblem.gradient_exact,
        BroydenProblem.hessian_exact,
        fieldnames_simple,
        False
    ))
    processes.append(p6)

    # Start all processes
    print(f"Starting {len(processes)} experiments in parallel...")
    for p in processes:
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("\nAll experiments completed.")
