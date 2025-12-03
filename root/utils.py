import numpy as np
import matplotlib.pyplot as plt

def backtracking_line_search(f, grad_f, x, p, alpha0=1.0, rho=0.5, c=1e-4, btmax=20):
    k = 0
    satisfyed = False
    fx = f(x)
    grad_fx = grad_f(x)
    
    # Pre-calcoliamo il termine lineare per efficienza
    grad_dot_p = np.dot(grad_fx, p)
    
    while k < btmax and not satisfyed:
        alpha = alpha0 * (rho ** k)
        # Condizione di Armijo
        if f(x + alpha * p) <= fx + c * alpha * grad_dot_p:
            satisfyed = True
            print(k)
        else:
            k += 1
    
    return alpha if satisfyed else alpha0 * (rho ** btmax)

def analyze_convergence(history):
    """
    Calcola l'ordine di convergenza p_k basato sulla norma del gradiente.
    history: lista di tuple (k, f(x), grad_norm)
    """
    errors = np.array([h[2] for h in history])
    errors = errors[errors > 1e-16] # Evita divisioni per zero
    
    rates = []
    print(f"\n{'Iter k':<10} | {'Grad Norm':<15} | {'Stima Ordine p':<15}")
    print("-" * 45)
    
    for k in range(1, len(errors) - 1):
        e_k = errors[k]
        e_k_prev = errors[k-1]
        e_k_next = errors[k+1]
        
        try:
            # p ~ log(e_{k+1}/e_k) / log(e_k/e_{k-1})
            denom = np.log(e_k / e_k_prev)
            if abs(denom) < 1e-12: 
                p_k = 0.0
            else:
                p_k = np.log(e_k_next / e_k) / denom
        except (ZeroDivisionError, ValueError):
            p_k = 0.0
            
        rates.append(p_k)
        print(f"{k:<10} | {e_k:<15.2e} | {p_k:<15.4f}")
        
    return errors, rates

def plot_convergence(history, method_name="Method"):
    iterazioni, values_f, values_g = zip(*history)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color_f = 'tab:blue'
    ax1.set_xlabel('Iterazioni (k)')
    ax1.set_ylabel(r'$F(x_k)$', color=color_f, fontsize=12)
    ax1.plot(iterazioni, values_f, color=color_f, marker='o', label=r'$F(x_k)$')
    ax1.tick_params(axis='y', labelcolor=color_f)
    ax1.set_yscale('log')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2 = ax1.twinx()
    color_g = 'tab:red'
    ax2.set_ylabel(r'$||\nabla F(x_k)||$', color=color_g, fontsize=12)
    ax2.plot(iterazioni, values_g, color=color_g, linestyle='--', marker='x', label=r'$||\nabla F||$')
    ax2.tick_params(axis='y', labelcolor=color_g)
    ax2.set_yscale('log')
    
    plt.title(f'Convergenza: {method_name}')
    fig.tight_layout()
    plt.show()