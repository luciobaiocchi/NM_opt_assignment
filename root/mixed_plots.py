import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl
from matplotlib.ticker import MaxNLocator
import os
from utils import analyze_convergence1


# --- Import dei tuoi moduli ---
from broyden import BroydenProblem
from banded_trig import BandedTrigonometric
from methods import NewtonMethods

# ==========================================
# 1. HELPER: Calcolo Rates
# ==========================================
def compute_rates_sequence(history):
    try:
        xs = [h['x'] for h in history]
    except (TypeError, KeyError):
        xs = [h[0] for h in history]

    if len(xs) < 3:
        return [], []

    errors = np.array([npl.norm(xs[k] - xs[k-1]) for k in range(1, len(xs))])
    
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
        
        denom = np.log(e_k / e_km1)
        if abs(denom) < 1e-10: continue
            
        p_k = np.log(e_kp1 / e_k) / denom
        
        if -0.5 < p_k < 4.5: 
            rates.append(p_k)
            valid_iters.append(iters[i])
            
    return valid_iters, rates

# ==========================================
# 2. FUNZIONE DI PLOT (MODIFICATA PER SALVATAGGIO)
# ==========================================
def plot_combined_panel(problem_class, histories, title, x_bounds=(-2.5, 0.5), y_bounds=(-2.5, 0.5), save_filename=None):
    """
    Se save_filename è passato, salva il file. Altrimenti mostra a video.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    # --- SUBPLOT 1: CONTOUR ---
    grid_points = 100
    x = np.linspace(x_bounds[0], x_bounds[1], grid_points)
    y = np.linspace(y_bounds[0], y_bounds[1], grid_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pt = np.array([X[i, j], Y[i, j]])
            Z[i, j] = problem_class.func(pt)

    ax1.contourf(X, Y, Z, levels=30, cmap='coolwarm', alpha=0.4, linewidths=1)

    for idx, (lbl, hist) in enumerate(histories):
        try:
            xs = [h['x'][0] for h in hist]
            ys = [h['x'][1] for h in hist]
        except:
            xs = [h[0][0] for h in hist]
            ys = [h[0][1] for h in hist]
        
        ax1.plot(xs, ys, marker='o', markersize=4, linestyle='-', linewidth=2, 
                 label=lbl, color=colors[idx], alpha=1)
        ax1.plot(xs[0], ys[0], marker='x', color='black', markersize=6, markeredgewidth=2)
        ax1.plot(xs[-1], ys[-1], marker='*', color='red', markersize=10, zorder=5)

    ax1.set_title("Optimization Paths (Top View)", fontsize=12, fontweight='bold')
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_xlim(x_bounds)
    ax1.set_ylim(y_bounds)

    # --- SUBPLOT 2: RATES ---
    lines_plotted = False
    ax2.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Linear (1.0)')
    ax2.axhline(2.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Quadratic (2.0)')

    for idx, (lbl, hist) in enumerate(histories):
        iters, rates = compute_rates_sequence(hist)
        if len(rates) > 0:
            col = colors[idx]
            ax2.plot(iters, rates, marker='o', markersize=6, linestyle='-', linewidth=2, 
                     color=col, label=lbl, alpha=0.85)
            lines_plotted = True

    ax2.set_title("Experimental Convergence Rates ($p_k$)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Iteration $k$")
    ax2.set_ylabel("Est. Order")
    ax2.set_xlim(1.5, 7.5)
    ax2.set_ylim(0, 4)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.grid(True, which='both', linestyle=':', alpha=0.6)

    if lines_plotted:
        ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)
    else:
        ax2.text(0.5, 0.5, "Insufficient data for rates", ha='center', transform=ax2.transAxes)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- LOGICA DI SALVATAGGIO ---
    if save_filename:
        # Crea la directory se non esiste (sicurezza extra)
        directory = os.path.dirname(save_filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        plt.savefig(save_filename, dpi=150, bbox_inches='tight') # dpi=150 è un buon compromesso
        print(f"Salvataggio completato: {save_filename}")
        plt.close(fig) # IMPORTANTE: Chiude la figura per liberare la RAM
    else:
        plt.show()

# ==========================================
# 3. RUNNER PRINCIPALE
# ==========================================
def run_combined_analysis():
    SEED = 358616
    np.random.seed(SEED)
    N = 2
    K_MAX = 200
    TOL = 1e-4
    Problem = BandedTrigonometric
    Method = NewtonMethods.truncated_newton
    start_points = []
    
    #BRO 
    #bounds_x = (-2.5, 0.5)
    #bounds_y = (-2.5, 0.5)
    #OUTPUT_DIR = "plots_output/bro/mn"
    #OUTPUT_DIR = "plots_output/bro/tc"
    #start_points.append( ("Suggested", -np.ones(N)) )
    #rand_pts = np.random.uniform(-2, 0, (5, N))
    #for i, pt in enumerate(rand_pts):
    #    start_points.append( (f"Rnd{i+1}", pt) )
    
    #TRI
    #bounds_x = (-3.5, 2)
    #bounds_y = (-2, 2.5)
    bounds_x = (-10, 5)
    bounds_y = (-10, 5)
    #OUTPUT_DIR = "plots_output/tri/mn"
    OUTPUT_DIR = "plots_output/tri/tc"
    start_points.append( ("Suggested", np.ones(N)) )
    rand_pts = np.random.uniform(0, 2, (5, N))
    for i, pt in enumerate(rand_pts):
        start_points.append( (f"Rnd{i+1}", pt) )


    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Creata cartella di output: {OUTPUT_DIR}")
    
    
    h_exponents = [4, 8, 12]
    strategies = [("Fixed", False), ("Dynamic", True)]

    print(f"Starting Analysis (Saving to {OUTPUT_DIR})...")

    # 1. BASELINE
    hist_exact = []
    for lbl, x0 in start_points:
        _, _, _, _, h = Method(
            x0, Problem.func, Problem.gradient_exact, Problem.hessian_exact,
            alpha0=1.0, kmax=K_MAX, tolgrad=TOL, c1=1e-4, rho=0.5, btmax=50
        )
        hist_exact.append((lbl, h))
    
    # Salva baseline
    plot_combined_panel(
        Problem, hist_exact, 
        "Baseline: Exact Gradient & Hessian", 
        bounds_x, bounds_y,
        save_filename=f"{OUTPUT_DIR}/Baseline_Exact.png"
    )

    # 2. FINITE DIFFERENCES
    for strat_name, is_dyn in strategies:
        for k_pow in h_exponents:
            h_val = 10**(-k_pow)
            h_str = f"10^{{-{k_pow}}}"
            
            hist_mixed = []
            hist_full = []
            
            for lbl, x0 in start_points:
                # A. MIXED
                hess_wrapper = lambda x: Problem.hessian_with_jacobian(x, h=h_val, is_h_dynamic=is_dyn)
                _, _, _, _, h_m = Method(
                    x0, Problem.func, Problem.gradient_exact, hess_wrapper,
                    alpha0=1.0, kmax=K_MAX, tolgrad=TOL, c1=1e-4, rho=0.5, btmax=50,
                    dynamic=is_dyn, h=h_val
                )
                hist_mixed.append((lbl, h_m))
                
                # B. FULL                
                _, _, _, _, h_f = Method(
                    x0, Problem.func, Problem.gradient_fd, Problem.hessian_fd,
                    alpha0=1.0, kmax=K_MAX, tolgrad=TOL, c1=1e-4, rho=0.5, btmax=50,
                    dynamic=is_dyn, h=h_val
                )
                hist_full.append((lbl, h_f))    
            
            # Salva Mixed
            fname_mixed = f"{OUTPUT_DIR}/Mixed_{strat_name}_h1e-{k_pow}.png"
            plot_combined_panel(
                Problem, hist_mixed, 
                f"Mixed FD ({strat_name}): Exact Grad + FD Hess ($h \\approx {h_str}$)",
                bounds_x, bounds_y,
                save_filename=fname_mixed
            )
            
            # Salva Full
            fname_full = f"{OUTPUT_DIR}/Full_{strat_name}_h1e-{k_pow}.png"
            plot_combined_panel(
                Problem, hist_full, 
                f"Full FD ({strat_name}): FD Grad ($h \\approx {h_str}$) + FD Hess",
                bounds_x, bounds_y,
                save_filename=fname_full
            )

if __name__ == "__main__":
    run_combined_analysis()