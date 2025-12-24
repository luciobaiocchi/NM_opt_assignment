import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl
import os

# --- Import dei tuoi moduli ---
from broyden import BroydenProblem
from banded_trig import BandedTrigonometric
from methods import NewtonMethods

# ==========================================
# 1. HELPER: Calcolo Rates (Invariato)
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

# ==========================================
# 2. NUOVA FUNZIONE DI PLOT RIASSUNTIVA (GRID 2x2)
# ==========================================
def plot_summary_grid(n, datasets_dict, save_filename):
    """
    Crea una figura 2x2 con i 4 scenari principali.
    datasets_dict deve contenere 4 chiavi con i dizionari 'history'.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten() # Per iterare facilmente su ax1, ax2, ax3, ax4
    
    # Colormap consistente
    # Generiamo colori sufficienti per le diverse curve (Rnd1, Rnd2, ecc.)
    # Assumiamo max 10 curve per plot
    colors = plt.cm.tab10(np.linspace(0, 1, 10)) 

    # Titoli dei 4 pannelli (devono corrispondere all'ordine di inserimento)
    panel_titles = list(datasets_dict.keys())
    
    for idx, (ax, title) in enumerate(zip(axes, panel_titles)):
        histories = datasets_dict[title]
        lines_plotted = False
        
        # --- Plot delle singole traiettorie (Rnd1, Rnd2...) ---
        for i, (label, history) in enumerate(histories.items()):
            iters, rates = compute_rates_sequence(history)
            
            if len(rates) > 1:
                col = colors[i % len(colors)]
                
                # Smooth data
                win = 3 if len(rates) < 20 else 5
                rates_smooth = moving_average(rates, window_size=win)
                iters_smooth = iters[len(iters) - len(rates_smooth):]
                
                # Plot
                ax.plot(iters_smooth, rates_smooth, color=col, alpha=0.9, linewidth=2, 
                        marker='o', markersize=4, label=label)
                lines_plotted = True

        # --- Linee di Riferimento ---
        ax.axhline(1.0, color='firebrick', linestyle='--', linewidth=1.5, alpha=0.6, label='Linear (1.0)')
        ax.axhline(2.0, color='forestgreen', linestyle='--', linewidth=1.5, alpha=0.6, label='Quadratic (2.0)')
        
        # --- Estetica ---
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("Iteration $k$")
        ax.set_ylabel("Order $p_k$")
        ax.set_ylim(-0.5, 3.5) # Focus sulla zona d'interesse
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if idx == 0 and lines_plotted: # Legenda solo nel primo per pulizia (o in tutti se preferisci)
             ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
        elif not lines_plotted:
             ax.text(0.5, 0.5, "Insufficient data", ha='center', transform=ax.transAxes, color='gray')

    fig.suptitle(f"Convergence Rates Summary - Banded Trigonometric ($N={n}$)", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Lascia spazio per il suptitle
    
    # Salvataggio
    if save_filename:
        directory = os.path.dirname(save_filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f" >> Creato Plot Riassuntivo: {save_filename}")
        plt.close(fig)

# ==========================================
# 3. MAIN RUNNER OTTIMIZZATO
# ==========================================
def run_summary_analysis():
    SEED = 358616
    np.random.seed(SEED)
    
    # Configurazione Simulazione
    n_list = [1000, 10000] # Puoi aggiungere 100000 se hai RAM/Tempo
    K_MAX = 200
    TOL = 1e-4
    
    # Parametri specifici da confrontare
    h_bad = 1e-4  # Caso impreciso
    h_good = 1e-8 # Caso preciso
    
    Problem = BandedTrigonometric
    Method = NewtonMethods.modified_newton_trig
    
    OUTPUT_DIR = "./plots_summary"
    
    print(f"--- STARTING SUMMARY ANALYSIS ---")

    for N in n_list:
        print(f"\nProcessing N = {N}...")
        
        # Generazione Punti
        start_points = []
        start_points.append(("Suggested", np.ones(N)))
        rand_pts = np.random.uniform(0, 2, (3, N)) # Riduciamo a 3 random per pulizia grafico
        for i, pt in enumerate(rand_pts):
            start_points.append((f"Rnd {i+1}", pt))

        # Contenitori per i risultati da confrontare
        data_baseline = {}
        data_full_fixed_bad = {}
        data_full_fixed_good = {}
        data_full_dynamic_good = {}

        # --- LOOP ESECUZIONE ---
        for label, x0 in start_points:
            
            # 1. BASELINE (Exact)
            _, _, gnorm, _, hist = Method(
                x0, Problem.func, Problem.gradient_exact, Problem.hessian_exact,
                alpha0=1.0, kmax=K_MAX, tolgrad=TOL, c1=1e-4, rho=0.5, btmax=50
            )
            if gnorm <= TOL: data_baseline[label] = hist

            # 2. FULL FIXED h=1e-4 (Pessimo)
            _, _, gnorm, _, hist = Method(
                x0, Problem.func, Problem.gradient_fd, Problem.hessian_fd,
                alpha0=1.0, kmax=K_MAX, tolgrad=TOL, c1=1e-4, rho=0.5, btmax=50,
                dynamic=False, h=h_bad
            )
            if gnorm <= TOL: data_full_fixed_bad[label] = hist
            
            # 3. FULL FIXED h=1e-8 (Buono)
            _, _, gnorm, _, hist = Method(
                x0, Problem.func, Problem.gradient_fd, Problem.hessian_fd,
                alpha0=1.0, kmax=K_MAX, tolgrad=TOL, c1=1e-4, rho=0.5, btmax=50,
                dynamic=False, h=h_good
            )
            if gnorm <= TOL: data_full_fixed_good[label] = hist

            # 4. FULL DYNAMIC h=1e-8 (Ottimo)
            _, _, gnorm, _, hist = Method(
                x0, Problem.func, Problem.gradient_fd, Problem.hessian_fd,
                alpha0=1.0, kmax=K_MAX, tolgrad=TOL, c1=1e-4, rho=0.5, btmax=50,
                dynamic=True, h=h_good
            )
            if gnorm <= TOL: data_full_dynamic_good[label] = hist

        # --- GENERAZIONE GRID ---
        # Organizziamo i dati per il plot 2x2
        datasets_to_plot = {
            "a) Baseline (Exact Gradient & Hessian)": data_baseline,
            f"b) Full FD (Fixed $h={h_bad}$)": data_full_fixed_bad,
            f"c) Full FD (Fixed $h={h_good}$)": data_full_fixed_good,
            f"d) Full FD (Dynamic $h={h_good}$)": data_full_dynamic_good
        }
        
        fname = f"{OUTPUT_DIR}/Summary_Rates_N{N}.png"
        plot_summary_grid(N, datasets_to_plot, fname)

if __name__ == "__main__":
    run_summary_analysis()