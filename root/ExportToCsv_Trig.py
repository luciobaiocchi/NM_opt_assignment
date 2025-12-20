import numpy as np
import time
import csv
from broyden import BroydenProblem
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
n_list = [2, 100, 1000, 10000, 100000]

# --- OUTPUT CSV ---
csv_filename = "./results/results_tn_trig_exact_gradient.csv"
csv_filename2 = "./results/results_mn_trig_exact_gradient.csv"
csv_filename3 = "./results/results_tn_trig_all_aprox.csv"
csv_filename4 = "./results/results_mn_trig_all_aprox.csv"
csv_filename5 = "./results/results_tn_trig_all_exact.csv"
csv_filename6 = "./results/results_mn_trig_all_exact.csv"


fieldnames = [
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

print("\n" + "="*100)
print("==================  START  ==================")
print(f"=========  {csv_filename}  ==========")
print("="*100 + "\n")

with open(csv_filename, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for h in h_values:
        for N in n_list:
            for dynamic in dynamic_h:

                # 5 punti random + punto deterministico x0 = ones
                x0 = np.ones(N)
                hypercube_random = np.random.uniform(0, 2, (5, N))
                hypercube_trig = np.vstack([x0, hypercube_random])

                point = 0

                for x in hypercube_trig:
                    # --- AVVIO TIMER ---
                    start_time_tn = time.perf_counter()

                    xk_tn, fxk_tn, gradxk_norm_tn, k_tn, hist_tn = NewtonMethods.truncated_newton(
                        x,
                        BandedTrigonometric.func,
                        BandedTrigonometric.exact_gradient,
                        BandedTrigonometric.hess_diag_fd_from_exact_grad,
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
                    execution_time_tn = time.perf_counter() - start_time_tn

                    convergence_rate = analyze_convergence1(hist_tn)
                    success = (
                        "yes"
                        if (gradxk_norm_tn is not None and gradxk_norm_tn <= TOL and k_tn < K_MAX)
                        else "no"
                    )

                    # --- SCRITTURA CSV ---
                    writer.writerow({
                        "Point ID": point,
                        "n": N,
                        "h": f"{h:.5g}",                         # notazione scientifica compatta
                        "is_dynamic": dynamic,
                        "success": success,
                        "k": k_tn,
                        "grad_norm_final": f"{gradxk_norm_tn:.5g}" if gradxk_norm_tn is not None else "",
                        "fx_final": f"{fxk_tn:.5g}",
                        "convergence_rate": f"{convergence_rate:.5g}" if convergence_rate is not None else "",
                        "time": f"{execution_time_tn:.5g}"
                    })

                    point+=1

print(f"\nCSV salvato correttamente come: {csv_filename}")




print("\n" + "="*100)
print("==================  START  ==================")
print(f"=========  {csv_filename2}  ==========")
print("="*100 + "\n")

with open(csv_filename2, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for h in h_values:
        for N in n_list:
            for dynamic in dynamic_h:

                # 5 punti random + punto deterministico x0 = ones
                x0 = np.ones(N)
                hypercube_random = np.random.uniform(0, 2, (5, N))
                hypercube_trig = np.vstack([x0, hypercube_random])

                point = 0

                for x in hypercube_trig:
                    # --- AVVIO TIMER ---
                    start_time_mn = time.perf_counter()

                    xk_mn, fxk_mn, gradxk_norm_mn, k_mn, hist_mn = NewtonMethods.modified_newton_single_diag(
                        x,
                        BandedTrigonometric.func,
                        BandedTrigonometric.exact_gradient,
                        BandedTrigonometric.hess_diag_fd_from_exact_grad,
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
                    execution_time_mn = time.perf_counter() - start_time_mn

                    convergence_rate = analyze_convergence1(hist_mn)
                    success = (
                        "yes"
                        if (gradxk_norm_mn is not None and gradxk_norm_mn <= TOL and k_mn < K_MAX)
                        else "no"
                    )

                    # --- SCRITTURA CSV ---
                    writer.writerow({
                        "Point ID": point,
                        "n": N,
                        "h": f"{h:.5g}",                         # notazione scientifica compatta
                        "is_dynamic": dynamic,
                        "success": success,
                        "k": k_mn,
                        "grad_norm_final": f"{gradxk_norm_mn:.5g}" if gradxk_norm_mn is not None else "",
                        "fx_final": f"{fxk_mn:.5g}",
                        "convergence_rate": f"{convergence_rate:.5g}" if convergence_rate is not None else "",
                        "time": f"{execution_time_mn:.5g}"
                    })

                    point+=1

print(f"\nCSV salvato correttamente come: {csv_filename2}")



print("\n" + "="*100)
print("==================  START  ==================")
print(f"=========  {csv_filename3}  ==========")
print("="*100 + "\n")

with open(csv_filename3, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for h in h_values:
        for N in n_list:
            for dynamic in dynamic_h:

                # 5 punti random + punto deterministico x0 = ones
                x0 = np.ones(N)
                hypercube_random = np.random.uniform(0, 2, (5, N))
                hypercube_trig = np.vstack([x0, hypercube_random])

                point = 0

                for x in hypercube_trig:
                    # --- AVVIO TIMER ---
                    start_time_tn = time.perf_counter()

                    xk_tn, fxk_tn, gradxk_norm_tn, k_tn, hist_tn = NewtonMethods.truncated_newton(
                        x,
                        BandedTrigonometric.func,
                        BandedTrigonometric.gradient,
                        BandedTrigonometric.hess_diag_fd_from_exact_grad,
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
                    execution_time_tn = time.perf_counter() - start_time_tn

                    convergence_rate = analyze_convergence1(hist_tn)
                    success = (
                        "yes"
                        if (gradxk_norm_tn is not None and gradxk_norm_tn <= TOL and k_tn < K_MAX)
                        else "no"
                    )

                    # --- SCRITTURA CSV ---
                    writer.writerow({
                        "Point ID": point,
                        "n": N,
                        "h": f"{h:.5g}",                         # notazione scientifica compatta
                        "is_dynamic": dynamic,
                        "success": success,
                        "k": k_tn,
                        "grad_norm_final": f"{gradxk_norm_tn:.5g}" if gradxk_norm_tn is not None else "",
                        "fx_final": f"{fxk_tn:.5g}",
                        "convergence_rate": f"{convergence_rate:.5g}" if convergence_rate is not None else "",
                        "time": f"{execution_time_tn:.5g}"
                    })

                    point+=1

print(f"\nCSV salvato correttamente come: {csv_filename3}")



print("\n" + "="*100)
print("==================  START  ==================")
print(f"=========  {csv_filename4}  ==========")
print("="*100 + "\n")

with open(csv_filename4, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for h in h_values:
        for N in n_list:
            for dynamic in dynamic_h:

                # 5 punti random + punto deterministico x0 = ones
                x0 = np.ones(N)
                hypercube_random = np.random.uniform(0, 2, (5, N))
                hypercube_trig = np.vstack([x0, hypercube_random])

                point = 0

                for x in hypercube_trig:
                    # --- AVVIO TIMER ---
                    start_time_mn = time.perf_counter()

                    xk_mn, fxk_mn, gradxk_norm_mn, k_mn, hist_mn = NewtonMethods.modified_newton_single_diag(
                        x,
                        BandedTrigonometric.func,
                        BandedTrigonometric.gradient,
                        BandedTrigonometric.hessian,
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
                    execution_time_mn = time.perf_counter() - start_time_mn

                    convergence_rate = analyze_convergence1(hist_mn)
                    success = (
                        "yes"
                        if (gradxk_norm_mn is not None and gradxk_norm_mn <= TOL and k_mn < K_MAX)
                        else "no"
                    )

                    # --- SCRITTURA CSV ---
                    writer.writerow({
                        "Point ID": point,
                        "n": N,
                        "h": f"{h:.5g}",                         # notazione scientifica compatta
                        "is_dynamic": dynamic,
                        "success": success,
                        "k": k_mn,
                        "grad_norm_final": f"{gradxk_norm_mn:.5g}" if gradxk_norm_mn is not None else "",
                        "fx_final": f"{fxk_mn:.5g}",
                        "convergence_rate": f"{convergence_rate:.5g}" if convergence_rate is not None else "",
                        "time": f"{execution_time_mn:.5g}"
                    })

                    point+=1

print(f"\nCSV salvato correttamente come: {csv_filename4}")


fieldnames2 = [
    "Point ID",
    "n",
    "success",
    "k",
    "grad_norm_final",
    "fx_final",
    "convergence_rate",
    "time"
]

print("\n" + "="*100)
print("==================  START  ==================")
print(f"=========  {csv_filename5}  ==========")
print("="*100 + "\n")

with open(csv_filename5, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames2)
    writer.writeheader()

    for N in n_list:
        # 5 punti random + punto deterministico x0 = ones
        x0 = np.ones(N)
        hypercube_random = np.random.uniform(0, 2, (5, N))
        hypercube_trig = np.vstack([x0, hypercube_random])
        point = 0
        for x in hypercube_trig:
            # --- AVVIO TIMER ---
            start_time_tn = time.perf_counter()
            xk_tn, fxk_tn, gradxk_norm_tn, k_tn, hist_tn = NewtonMethods.truncated_newton(
                x,
                BandedTrigonometric.func,
                BandedTrigonometric.exact_gradient,
                BandedTrigonometric.exact_hessian,
                alpha0=1.0,
                kmax=K_MAX,
                tolgrad=TOL,
                c1=1e-4,
                rho=0.5,
                btmax=50,
                dynamic=None,
                h=None
            )
            # --- STOP TIMER ---
            execution_time_tn = time.perf_counter() - start_time_tn
            convergence_rate = analyze_convergence1(hist_tn)
            success = (
                "yes"
                if (gradxk_norm_tn is not None and gradxk_norm_tn <= TOL and k_tn < K_MAX)
                else "no"
            )
            # --- SCRITTURA CSV ---
            writer.writerow({
                "Point ID": point,
                "n": N,
                "success": success,
                "k": k_tn,
                "grad_norm_final": f"{gradxk_norm_tn:.5g}" if gradxk_norm_tn is not None else "",
                "fx_final": f"{fxk_tn:.5g}",
                "convergence_rate": f"{convergence_rate:.5g}" if convergence_rate is not None else "",
                "time": f"{execution_time_tn:.5g}"
            })
            point+=1

print(f"\nCSV salvato correttamente come: {csv_filename5}")



print("\n" + "="*100)
print("==================  START  ==================")
print(f"=========  {csv_filename6}  ==========")
print("="*100 + "\n")

with open(csv_filename6, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames2)
    writer.writeheader()

    for N in n_list:
        # 5 punti random + punto deterministico x0 = ones
        x0 = np.ones(N)
        hypercube_random = np.random.uniform(0, 2, (5, N))
        hypercube_trig = np.vstack([x0, hypercube_random])
        point = 0
        for x in hypercube_trig:
            # --- AVVIO TIMER ---
            start_time_mn = time.perf_counter()
            xk_mn, fxk_mn, gradxk_norm_mn, k_mn, hist_mn = NewtonMethods.modified_newton_single_diag(
                x,
                BandedTrigonometric.func,
                BandedTrigonometric.exact_gradient,
                BandedTrigonometric.exact_hessian,
                alpha0=1.0,
                kmax=K_MAX,
                tolgrad=TOL,
                c1=1e-4,
                rho=0.5,
                btmax=50,
                dynamic=None,
                h=None
            )
            # --- STOP TIMER ---
            execution_time_mn = time.perf_counter() - start_time_mn
            convergence_rate = analyze_convergence1(hist_mn)
            success = (
                "yes"
                if (gradxk_norm_mn is not None and gradxk_norm_mn <= TOL and k_mn < K_MAX)
                else "no"
            )
            # --- SCRITTURA CSV ---
            writer.writerow({
                "Point ID": point,
                "n": N,
                "success": success,
                "k": k_mn,
                "grad_norm_final": f"{gradxk_norm_mn:.5g}" if gradxk_norm_mn is not None else "",
                "fx_final": f"{fxk_mn:.5g}",
                "convergence_rate": f"{convergence_rate:.5g}" if convergence_rate is not None else "",
                "time": f"{execution_time_mn:.5g}"
            })
            point+=1

print(f"\nCSV salvato correttamente come: {csv_filename6}")

