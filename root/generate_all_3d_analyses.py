import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Import i tuoi moduli
from broyden import BroydenProblem
from banded_trig import BandedTrigonometric
from methods import NewtonMethods

# ==========================================
# 1. FUNZIONE DI PLOT 3D (Multipercorso)
# ==========================================
def plot_3d_surface_and_paths(problem_class, histories, title, x_bounds=(-1.0, 3.0), y_bounds=(-1.0, 3.0), grid_points=100):
    """
    Crea un plot 3D della superficie f(x) e sovrappone TUTTE le traiettorie passate in 'histories'.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # --- 1. Generazione della Superficie ---
    x = np.linspace(x_bounds[0], x_bounds[1], grid_points)
    y = np.linspace(y_bounds[0], y_bounds[1], grid_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Costruiamo il punto 2D
            pt = np.array([X[i, j], Y[i, j]])
            Z[i, j] = problem_class.func(pt)

    # Plot Superficie (Semitrasparente)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, 
                           linewidth=0, antialiased=True, alpha=0.6)

    # --- 2. Ombra (Contour proiettato) ---
    z_min_limit = np.min(Z) - (np.max(Z) - np.min(Z)) * 0.1
    ax.contourf(X, Y, Z, zdir='z', offset=z_min_limit, cmap=cm.viridis, alpha=0.4)

    # --- 3. Plot dei Percorsi ---
    colors = plt.cm.jet(np.linspace(0, 1, len(histories)))
    
    for idx, (label, history) in enumerate(histories):
        # Estrazione coordinate
        try:
            xs = np.array([h['x'][0] for h in history])
            ys = np.array([h['x'][1] for h in history])
            zs = np.array([h['fx'] for h in history])
        except (TypeError, KeyError):
            xs = np.array([h[0][0] for h in history])
            ys = np.array([h[0][1] for h in history])
            zs = np.array([h[1] for h in history])

        # Plot linea
        ax.plot(xs, ys, zs, color=colors[idx], linewidth=2, label=label, zorder=10)
        
        # Plot punti intermedi
        ax.scatter(xs, ys, zs, color=colors[idx], s=15, depthshade=False)
        
        # Start (X) e End (Stella)
        ax.scatter(xs[0], ys[0], zs[0], color='black', marker='x', s=60, linewidth=2.5, label='_nolegend_')
        ax.scatter(xs[-1], ys[-1], zs[-1], color='red', marker='*', s=120, zorder=15, label='_nolegend_')

    # --- 4. Estetica ---
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x)$')
    ax.set_zlim(z_min_limit, np.max(Z))
    
    # Vista iniziale ottimizzata per vedere la "conca"
    ax.view_init(elev=50, azim=-140)
    
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Cost Function Value')
    ax.legend(loc='upper right', fontsize='small')
    
    print(f"Plot generato: {title}")
    print(" -> Chiudi la finestra del grafico per continuare con il prossimo...")
    plt.show()

# ==========================================
# 2. ESECUZIONE DI TUTTE LE COMBINAZIONI
# ==========================================
def run_all_3d_analyses():
    SEED = 358616
    np.random.seed(SEED)
    N = 2
    K_MAX = 200
    TOL = 1e-8
    
    # --- SETUP PROBLEMA BANDED TRIGONOMETRIC ---
    Problem = BroydenProblem
    # Usiamo il metodo diagonale specifico per questo problema
    Method = NewtonMethods.modified_newton_bro 
    
    # Bounds corretti per visualizzare convergenza a 0 partendo da [0, 2]
    bounds = (-2.5, 2.5) 

    # --- Generazione Punti di Partenza ---
    start_points = []
    # Punto suggerito (ones)
    start_points.append( ("Suggested (1.0)", -np.ones(N)) )
    
    # Punti random in [0, 2] come da specifica assignment
    hypercube_random = np.random.uniform(-2, 0, (5, N))
    
    # Riattivato loop punti random
    for i, pt in enumerate(hypercube_random):
        start_points.append( (f"Rnd{i+1}", pt) )

    print("==========================================")
    print(" STARTING FULL 3D VISUALIZATION (BANDED TRIG)")
    print("==========================================")

    # ----------------------------------------
    # 1. BASELINE: EXACT GRADIENT & HESSIAN
    # ----------------------------------------
    hist_exact = []
    for lbl, x0 in start_points:
        _, _, _, _, h = Method(
            x0, Problem.func, Problem.gradient_exact, Problem.hessian_exact,
            alpha0=1.0, kmax=K_MAX, tolgrad=TOL, c1=1e-4, rho=0.5, btmax=50
        )
        hist_exact.append((lbl, h))
        
    plot_3d_surface_and_paths(
        Problem, hist_exact, 
        "3D Banded Trig: Exact Gradient & Hessian", 
        x_bounds=bounds, y_bounds=bounds
    )

    # ----------------------------------------
    # 2. LOOPS SU STRATEGIE E H
    # ----------------------------------------
    strategies = [("Fixed", False), ("Dynamic", True)]
    h_exponents = [4, 8, 12]

    for strat_name, is_dyn in strategies:
        for k_pow in h_exponents:
            h_val = 10**(-k_pow)
            h_str = f"10^{{-{k_pow}}}"
            
            hist_mixed = []
            hist_full = []
            
            for lbl, x0 in start_points:
                # --- A. MIXED FD (Exact Grad + FD Hessian) ---
                # Usiamo hess_diag_fd_from_exact_grad perché è un metodo diagonale
                hess_wrapper = lambda x: Problem.hessian_with_jacobian(x, h=h_val, is_h_dynamic=is_dyn)
                
                _, _, _, _, h_m = Method(
                    x0, Problem.func, Problem.gradient_exact, hess_wrapper,
                    alpha0=1.0, kmax=K_MAX, tolgrad=TOL, c1=1e-4, rho=0.5, btmax=50,
                    dynamic=is_dyn, h=h_val
                )
                hist_mixed.append((lbl, h_m))
                
                # --- B. FULL FD (FD Grad + FD Hessian) ---
                # Gradient: Calcolato internamente con h passato
                # Hessian: Problem.hessian (metodo standard con FD)
                _, _, _, _, h_f = Method(
                    x0, Problem.func, Problem.gradient_fd, Problem.hessian_fd,
                    alpha0=1.0, kmax=K_MAX, tolgrad=TOL, c1=1e-4, rho=0.5, btmax=50,
                    dynamic=is_dyn, h=h_val
                )
                hist_full.append((lbl, h_f))
            
            # Genera Plot Mixed
            plot_3d_surface_and_paths(
                Problem, hist_mixed,
                f"3D Mixed FD ({strat_name}): Exact Grad + FD Hess (h={h_str})",
                x_bounds=bounds, y_bounds=bounds
            )
            
            # Genera Plot Full
            plot_3d_surface_and_paths(
                Problem, hist_full,
                f"3D Full FD ({strat_name}): FD Grad (h={h_str}) + FD Hess",
                x_bounds=bounds, y_bounds=bounds
            )

if __name__ == "__main__":
    run_all_3d_analyses()