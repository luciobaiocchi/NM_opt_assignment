import numpy as np
import time
import csv
import matplotlib.pyplot as plt
import numpy.linalg as npl
import os
from multiprocessing import Process
from banded_trig import BandedTrigonometric
from methods import NewtonMethods
from utils import analyze_convergence1, plot_convergence

# --- CONFIGURAZIONE ---
K_MAX = 200
TOL = 1e-4
SEED = 358616
verbose=True
enable_plotting = True

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
    """
    Esegue un batch di esperimenti per una data configurazione (N, h, dynamic).
    Ritorna (success_count, point_count, histories_dict).
    histories_dict mappa label -> history per ogni punto di partenza.
    """
    # 5 punti random + punto deterministico x0 = ones
    x0 = np.ones(N)

    rng = np.random.default_rng(SEED) # FOR PROCESSING IN PARALLEL
    hypercube_random = rng.uniform(0, 2, (5, N))
    hypercube_trig = np.vstack([x0, hypercube_random])

    point_id = 0
    success_count = 0
    histories_dict = {}  # Per il plotting
    
    point_labels = ["Suggested", "Rnd 1", "Rnd 2", "Rnd 3", "Rnd 4", "Rnd 5"]
    
    for idx, x in enumerate(hypercube_trig):
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
            h=h,
            verbose=verbose
        )

        # --- STOP TIMER ---
        execution_time = time.perf_counter() - start_time
        convergence_rate = analyze_convergence1(hist)
        
        success = (
            "yes"
            if (gradxk_norm is not None and gradxk_norm <= TOL and k < K_MAX)
            else "no"
        )
        if success == "yes":
            success_count += 1
            # Salva history solo per punti convergenti (utile per il plot)
            if enable_plotting:
                histories_dict[point_labels[idx]] = hist
        
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
    
    return success_count, point_id, histories_dict

# ==========================================
# HELPER FUNCTIONS FOR PLOTTING
# ==========================================
def compute_rates_sequence(history):
    try:
        xs = [h['x'] for h in history]
    except (TypeError, KeyError):
        xs = [h[0] for h in history]

    if len(xs) < 3:
        return [], []

    errors = np.array([npl.norm(xs[k] - xs[k-1]) for k in range(1, len(xs))])
    
    # Filtro rumore numerico
    valid_mask = errors > 1e-16
    errors = errors[valid_mask]
    iters = np.arange(1, len(xs))[valid_mask]
    
    rates = []
    valid_iters = []
    
    for i in range(1, len(errors) - 1):
        e_km1 = errors[i-1]
        e_k   = errors[i]
        e_kp1 = errors[i+1]
        
        if e_k < 1e-12: break 
        if abs(e_k - e_km1) < 1e-8 * e_km1: continue 
            
        denom = np.log(e_k / e_km1)
        if abs(denom) < 1e-10: continue
            
        p_k = np.log(e_kp1 / e_k) / denom
        
        if -0.5 < p_k < 4.5: 
            rates.append(p_k)
            valid_iters.append(iters[i])
            
    return valid_iters, rates

def moving_average(data, window_size=3):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_single_config(N, h, dynamic, histories_dict, log, save_filename):
    """
    Crea un singolo plot per una configurazione (N, h, dynamic).
    histories_dict mappa label -> history per ogni punto di partenza.
    log è la stringa descrittiva dell'esperimento.
    """
    if not histories_dict:
        return
        
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    lines_plotted = False
    
    for i, (label, history) in enumerate(histories_dict.items()):
        iters, rates = compute_rates_sequence(history)
        
        if len(rates) > 1:
            col = colors[i % len(colors)]
            win = 3 if len(rates) < 20 else 5
            rates_smooth = moving_average(rates, window_size=win)
            iters_smooth = iters[len(iters) - len(rates_smooth):]
            
            ax.plot(iters_smooth, rates_smooth, color=col, alpha=0.9, linewidth=2, 
                    marker='o', markersize=4, label=label)
            lines_plotted = True

    ax.axhline(1.0, color='firebrick', linestyle='--', linewidth=1.5, alpha=0.6, label='Linear (1.0)')
    ax.axhline(2.0, color='forestgreen', linestyle='--', linewidth=1.5, alpha=0.6, label='Quadratic (2.0)')
    
    ax.set_xlabel("Iteration $k$", fontsize=12)
    ax.set_ylabel("Order $p_k$", fontsize=12)
    ax.set_ylim(-0.5, 3.5)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    if lines_plotted:
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha='center', transform=ax.transAxes, color='gray', fontsize=14)

    # Titolo con log, N, h, dynamic
    dynamic_str = "Dynamic" if dynamic else "Fixed"
    h_str = f"{h:.0e}" if h is not None else "N/A"
    title = f"{log}\n$N={N}$, $h={h_str}$, {dynamic_str}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_filename:
        directory = os.path.dirname(save_filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f" >> Creato Plot: {save_filename}")
        plt.close(fig)

def run_experiment(log, filename, method, gradient_fn, hessian_fn, fieldnames, use_h_dynamic_loops=True, generate_plot=False, is_exact=False):
    """
    Esegue un esperimento e salva i risultati in CSV.
    Se generate_plot=True e enable_plotting=True, genera un plot per ogni (N, h, dynamic).
    """
    if verbose:
        print("\n" + "="*100)
        print(f"=========  {log}  ==========")
        print("="*100 + "\n")

    # Creo una stringa pulita per i nomi file e sottocartella
    log_clean = log.replace(' ', '_').replace('.', '').replace(',', '').replace('(', '').replace(')', '')
    OUTPUT_DIR = f"./plots_summary/Trig/{log_clean}"
    
    # Creo la directory se non esiste
    if enable_plotting and generate_plot:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(filename, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total_successes = 0
        total_attempts = 0
        if use_h_dynamic_loops:
            for N in n_list:
                for h in h_values:
                    for dynamic in dynamic_h:
                        if verbose:
                            print('\n')
                            print(f"==================  {h, N, dynamic}  ==================")
                            print('\n')
                        s, t, histories = process_batch(writer, method, gradient_fn, hessian_fn, N, h, dynamic, True)
                        total_successes += s
                        total_attempts += t
                        
                        # Genera plot per questa configurazione
                        if enable_plotting and generate_plot and histories:
                            dynamic_str = "dynamic" if dynamic else "fixed"
                            h_str = f"{h:.0e}".replace('-', 'm').replace('+', 'p')
                            fname = f"{OUTPUT_DIR}/Trig_N{N}_h{h_str}_{dynamic_str}.png"
                            plot_single_config(N, h, dynamic, histories, log, fname)
        else:
            # Esperimento senza h/dynamic loops (Exact)
            for N in n_list:
                s, t, histories = process_batch(writer, method, gradient_fn, hessian_fn, N, None, None, False)
                total_successes += s
                total_attempts += t
                
                # Genera plot per Exact (no h/dynamic)
                if enable_plotting and generate_plot and histories:
                    fname = f"{OUTPUT_DIR}/Trig_N{N}_exact.png"
                    plot_single_config(N, None, None, histories, log, fname)

    print(f"\n=========  {log}  ==========")
    print(f"Experiment finished. Total successes: {total_successes}/{total_attempts}")
    print(f"CSV salvato correttamente come: {filename}\n")


# --- ESECUZIONE ---

if __name__ == "__main__":
    start_total = time.time()
    processes = []

    # 1. TN, Exact Gradient, Hessian with Jacobian
    p1 = Process(target=run_experiment, kwargs={
        "log": "1. TN, Exact Gradient, Hessian with Jacobian",
        "filename": "./results/results_tn_trig_exact_gradient.csv",
        "method": NewtonMethods.truncated_newton,
        "gradient_fn": BandedTrigonometric.gradient_exact,
        "hessian_fn": BandedTrigonometric.hessian_with_jacobian,
        "fieldnames": fieldnames_full,
        "use_h_dynamic_loops": True,
        "generate_plot": True,
        "is_exact": False
    })
    processes.append(p1)

    # 2. MN, Exact Gradient, Hessian with Jacobian
    p2 = Process(target=run_experiment, kwargs={
        "log": "2. MN, Exact Gradient, Hessian with Jacobian",
        "filename": "./results/results_mn_trig_exact_gradient.csv",
        "method": NewtonMethods.modified_newton_trig,
        "gradient_fn": BandedTrigonometric.gradient_exact,
        "hessian_fn": BandedTrigonometric.hessian_with_jacobian,
        "fieldnames": fieldnames_full,
        "use_h_dynamic_loops": True,
        "generate_plot": True,
        "is_exact": False
    })
    processes.append(p2)

    # 3. TN, All Approx (FD)
    p3 = Process(target=run_experiment, kwargs={
        "log": "3. TN, All Approx (FD)",
        "filename": "./results/results_tn_trig_all_aprox.csv",
        "method": NewtonMethods.truncated_newton,
        "gradient_fn": BandedTrigonometric.gradient_fd,
        "hessian_fn": BandedTrigonometric.hessian_fd,
        "fieldnames": fieldnames_full,
        "use_h_dynamic_loops": True,
        "generate_plot": True,
        "is_exact": False
    })
    processes.append(p3)

    # 4. MN, All Approx (FD)
    p4 = Process(target=run_experiment, kwargs={
        "log": "4. MN, All Approx (FD)",
        "filename": "./results/results_mn_trig_all_aprox.csv",
        "method": NewtonMethods.modified_newton_trig,
        "gradient_fn": BandedTrigonometric.gradient_fd,
        "hessian_fn": BandedTrigonometric.hessian_fd,
        "fieldnames": fieldnames_full,
        "use_h_dynamic_loops": True,
        "generate_plot": True,
        "is_exact": False
    })
    processes.append(p4)

    # 5. TN, All Exact, No h/dynamic loops
    p5 = Process(target=run_experiment, kwargs={
        "log": "5. TN, All Exact, No h/dynamic loops",
        "filename": "./results/results_tn_trig_all_exact.csv",
        "method": NewtonMethods.truncated_newton,
        "gradient_fn": BandedTrigonometric.gradient_exact,
        "hessian_fn": BandedTrigonometric.hessian_exact,
        "fieldnames": fieldnames_simple,
        "use_h_dynamic_loops": False,
        "generate_plot": True,
        "is_exact": True
    })
    processes.append(p5)

    # 6. MN, All Exact, No h/dynamic loops
    p6 = Process(target=run_experiment, kwargs={
        "log": "6. MN, All Exact, No h/dynamic loops",
        "filename": "./results/results_mn_trig_all_exact.csv",
        "method": NewtonMethods.modified_newton_trig,
        "gradient_fn": BandedTrigonometric.gradient_exact,
        "hessian_fn": BandedTrigonometric.hessian_exact,
        "fieldnames": fieldnames_simple,
        "use_h_dynamic_loops": False,
        "generate_plot": True,
        "is_exact": True
    })
    processes.append(p6)

    # Start all processes
    print(f"Starting {len(processes)} experiments in parallel...")
    for p in processes:
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    end_total = time.time()
    print(f"\nAll experiments completed in {end_total - start_total:.2f} seconds.")
