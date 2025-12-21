import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl
import os  # <--- NUOVO IMPORT NECESSARIO PER IL SALVATAGGIO

# --- Import your modules ---
from broyden import BroydenProblem
from banded_trig import BandedTrigonometric
from methods import NewtonMethods

# ==========================================
# 1. HELPER: Compute Experimental Rates p_k
# ==========================================
def compute_rates_sequence(history):
    """
    Computes the sequence of experimental convergence rates p_k.
    """
    try:
        xs = [h['x'] for h in history]
    except (TypeError, KeyError):
        xs = [h[0] for h in history]

    if len(xs) < 3:
        return [], []

    # Error proxy: ||x_k - x_{k-1}||
    errors = np.array([npl.norm(xs[k] - xs[k-1]) for k in range(1, len(xs))])
    
    # Filter numeric noise
    valid_mask = errors > 1e-16
    errors = errors[valid_mask]
    iters = np.arange(1, len(xs))[valid_mask]
    
    rates = []
    valid_iters_for_rates = []
    
    for i in range(1, len(errors) - 1):
        e_km1 = errors[i-1]
        e_k   = errors[i]
        e_kp1 = errors[i+1]
        
        # Stability checks
        if e_k < 1e-12: break 
        if abs(e_k - e_km1) < 1e-8 * e_km1: continue 
            
        denom = np.log(e_k / e_km1)
        if abs(denom) < 1e-10: continue
            
        p_k = np.log(e_kp1 / e_k) / denom
        
        # Keep physical values roughly in range
        if -0.5 < p_k < 4.5: 
            rates.append(p_k)
            valid_iters_for_rates.append(iters[i])
            
    return valid_iters_for_rates, rates

# ==========================================
# 2. HELPER: Smoothing Function
# ==========================================
def moving_average(data, window_size=3):
    """Calcola la media mobile per smussare il grafico."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# ==========================================
# 3. PLOTTING FUNCTION (IMPROVED & SAVING)
# ==========================================
def plot_rates_for_dimension(n, histories_dict, title_suffix, save_filename=None):
    """
    Genera il plot dei tassi di convergenza.
    Se save_filename è fornito, salva il file invece di mostrarlo.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use a high-contrast colormap
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories_dict)))
    
    lines_plotted = False
    
    for idx, (label, history) in enumerate(histories_dict.items()):
        iters, rates = compute_rates_sequence(history)
        
        if len(rates) > 1:
            color = colors[idx]
            
            # 1. Plot RAW data (Faint/Transparent)
            ax.plot(iters, rates, color=color, alpha=0.25, linewidth=1, linestyle='-')
            
            # 2. Plot SMOOTHED data (Solid/Thick)
            # Window size: 3 for short sequences, 5 for longer ones
            win = 3 if len(rates) < 20 else 5
            rates_smooth = moving_average(rates, window_size=win)
            
            # Adjust iters to match the convolution output size
            iters_smooth = iters[len(iters) - len(rates_smooth):]
            
            ax.plot(iters_smooth, rates_smooth, color=color, alpha=1.0, linewidth=2.5, 
                    label=f"{label} (Smooth)")
            
            # Mark the final point to see where it ended
            ax.scatter(iters_smooth[-1], rates_smooth[-1], color=color, s=40, zorder=5)
            
            lines_plotted = True

    # --- Reference Lines (Better Visibility) ---
    ax.axhline(1.0, color='firebrick', linestyle='--', linewidth=1.5, alpha=0.8, label='Linear (1.0)')
    ax.axhline(2.0, color='forestgreen', linestyle='--', linewidth=1.5, alpha=0.8, label='Quadratic (2.0)')
    
    # --- Aesthetics ---
    ax.set_title(f"Experimental Conv. Rates ($N={n}$)\n{title_suffix}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Iteration $k$", fontsize=11)
    ax.set_ylabel("Est. Order $p_k$", fontsize=11)
    
    # Focus on the relevant area (Linear to Quadratic)
    ax.set_ylim(-2, 4) 
    
    # Grid
    ax.grid(True, which='major', linestyle='-', alpha=0.6)
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)
    ax.minorticks_on()
    
    if lines_plotted:
        ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=9)
    else:
        ax.text(0.5, 0.5, "Insufficient convergence steps for plot", 
                ha='center', transform=ax.transAxes, color='gray')

    plt.tight_layout()

    # --- LOGICA DI SALVATAGGIO ---
    if save_filename:
        # Crea la directory se non esiste (sicurezza extra)
        directory = os.path.dirname(save_filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f"Salvataggio completato: {save_filename}")
        plt.close(fig) # IMPORTANTE: Chiude la figura per liberare la RAM
    else:
        plt.show()

# ==========================================
# 4. EXPERIMENT RUNNER
# ==========================================
def run_full_analysis():
    # --- CONFIG ---
    SEED = 358616
    np.random.seed(SEED)
    
    n_list = [1000, 10000, 100000] # Modifica a piacere
    K_MAX = 200
    TOL = 1e-4
    h_exponents = [4, 8, 12] 
    
    # Strategies: (Label, is_dynamic_flag)
    strategies = [("Fixed h", False), ("Dynamic h", True)]
    Problem = BandedTrigonometric
    Method = NewtonMethods.modified_newton_single_diag
    
    # Cartella Output
    OUTPUT_DIR = "./plots/BandedTrigMN"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Creata cartella di output: {OUTPUT_DIR}")
        
    print(f"Starting CLEAN Analysis...")

    for N in n_list:
        print(f"\nProcessing N = {N}...")

        # 1. Start Points
        start_points = []
        start_points.append( ("Suggested", np.ones(N)) )
        rand_pts = np.random.uniform(0, 2, (5, N))
        for i, pt in enumerate(rand_pts):
            start_points.append( (f"Rnd {i+1}", pt) )

        res_exact = {} 
        res_mixed = {strat: {k: {} for k in h_exponents} for strat, _ in strategies}
        res_full  = {strat: {k: {} for k in h_exponents} for strat, _ in strategies}

        for label, x0 in start_points:
            # A. Exact
            _, _, gnorm, _, hist = Method(
                x0, Problem.func, Problem.exact_gradient, Problem.exact_hessian,
                alpha0=1.0, kmax=K_MAX, tolgrad=TOL, c1=1e-4, rho=0.5, btmax=50
            )
            if gnorm <= TOL: res_exact[label] = hist

            for strat_name, is_dyn in strategies:
                for k_pow in h_exponents:
                    h_val = 10**(-k_pow)
                    
                    # B. Mixed FD
                    hess_wrapper = lambda x: Problem.hess_diag_fd_from_exact_grad(x, h=h_val, is_h_dynamic=is_dyn)
                    _, _, gnorm_m, _, hist_m = Method(
                        x0, Problem.func, Problem.exact_gradient, hess_wrapper,     
                        alpha0=1.0, kmax=K_MAX, tolgrad=TOL, c1=1e-4, rho=0.5, btmax=50, dynamic=is_dyn, h=h_val   
                    )
                    if gnorm_m <= TOL: res_mixed[strat_name][k_pow][label] = hist_m

                    # C. Full FD
                    _, _, gnorm_f, _, hist_f = Method(
                        x0, Problem.func, Problem.gradient, Problem.hessian, 
                        alpha0=1.0, kmax=K_MAX, tolgrad=TOL, c1=1e-4, rho=0.5, btmax=50, dynamic=is_dyn, h=h_val 
                    )
                    if gnorm_f <= TOL: res_full[strat_name][k_pow][label] = hist_f

        # --- Plots & Saving ---
        if res_exact:
            fname = f"{OUTPUT_DIR}/Rates_N{N}_Baseline_Exact.png"
            plot_rates_for_dimension(N, res_exact, "Baseline: Exact Gradient & Hessian", save_filename=fname)

        for strat_name, _ in strategies:
            # Pulisce il nome della strategia per il file (es. "Fixed h" -> "Fixed_h")
            safe_strat = strat_name.replace(" ", "_")
            
            for k_pow in h_exponents:
                # 1. Mixed
                data_mixed = res_mixed[strat_name][k_pow]
                if data_mixed:
                    title_m = f"Mixed FD ({strat_name}): Exact Grad + FD Hess ($h \\approx 10^{{-{k_pow}}}$)"
                    fname_m = f"{OUTPUT_DIR}/Rates_N{N}_Mixed_{safe_strat}_h1e-{k_pow}.png"
                    plot_rates_for_dimension(N, data_mixed, title_m, save_filename=fname_m)

                # 2. Full
                data_full = res_full[strat_name][k_pow]
                if data_full:
                    title_f = f"Full FD ({strat_name}): FD Grad ($h \\approx 10^{{-{k_pow}}}$) + FD Hess"
                    fname_f = f"{OUTPUT_DIR}/Rates_N{N}_Full_{safe_strat}_h1e-{k_pow}.png"
                    plot_rates_for_dimension(N, data_full, title_f, save_filename=fname_f)

if __name__ == "__main__":
    run_full_analysis()