import numpy as np
import numpy.linalg as npl
import scipy.linalg as spla
import scipy.sparse
from utils import backtracking_line_search
class NewtonMethods:
    
    @staticmethod
    def modified_newton_bro(x0, f, gradf, hessf, alpha0=1.0, kmax=100, tolgrad=1e-5, c1=1e-4, rho=0.5, btmax=20, dynamic=False, h=1e-5, verbose=False):
        '''
            Solves an unconstrained optimization problem using the Modified Newton method, 
            exploiting the banded structure of the Hessian matrix.

            This implementation assumes the Hessian has a specific sparsity pattern 
            (lower bandwidth = 2) and utilizes `scipy.linalg.cholesky_banded` for efficient 
            linear system solving. If the Hessian is not positive definite, a multiple 
            of the identity matrix is added (Cholesky with added diagonal correction).

            :param x0: The starting point for the optimization algorithm (numpy array of shape (n,)).
            :param f: The objective function to minimize. Callable, returns a scalar.
            :param gradf: The gradient of the objective function. Callable, returns a numpy array of shape (n,).
            :param hessf: The Hessian of the objective function. Callable, returns a scipy.sparse matrix. 
                        Note: The code logic assumes a pentadiagonal structure (lower bandwidth of 2).
            :param alpha0: The initial step size for the backtracking line search.
            :param kmax: The maximum number of iterations allowed for the Newton method.
            :param tolgrad: The tolerance for the stopping criterion based on the gradient norm 
                            (stops if ||grad f|| < tolgrad).
            :param c1: The parameter for the Armijo (sufficient decrease) condition (typically in (0, 1)).
            :param rho: The reduction factor for the step size during backtracking (typically in (0, 1)).
            :param btmax: The maximum number of backtracking steps allowed per iteration.
            
            :return: A tuple containing the final iterate, the sequence of iterates, and the number of iterations.
        '''
        xk = x0.copy()
        n = len(x0)
        k = 0
        history = []
        
        # Initial state computation
        grad_xk = gradf(xk, h, is_h_dynamic=dynamic)
        gradfk_norm = np.linalg.norm(grad_xk)
        fx = f(xk)

        history.append({'k': 0, 'x': xk.copy(), 'fx': fx, 'gnorm': gradfk_norm})
        if verbose:
            print(f"--------- START: Modified Newton with BANDED Cholesky. N={n} ---------")
        
        while k < kmax and gradfk_norm > tolgrad:
            H_sparse = hessf(xk)
            
            # 2. Conversion to "Banded" format for scipy.linalg (Lower form)
            # H_sparse.diagonal(-k) returns an array of length N-k
            # For cholesky_banded (lower=True):
            # Row 0: Main diagonal (Len N)
            # Row 1: Lower diagonal 1 (Len N-1 -> Pad 1 at the end)
            # Row 2: Lower diagonal 2 (Len N-2 -> Pad 2 at the end)
            
            main_diag = H_sparse.diagonal(0)
            # NOTE: Padding at the end (append), not at the beginning!
            low1_diag = np.append(H_sparse.diagonal(-1), 0) 
            low2_diag = np.append(H_sparse.diagonal(-2), [0, 0])
            
            # Construct the ab 3xN matrix
            # Use copy() to ensure we don't modify the original main_diag in the tau loop
            ab_original = np.vstack([main_diag, low1_diag, low2_diag])
            
            # --- MODIFIED NEWTON (Tau adjustment) ---
            tau = 0
            beta = 1e-3
            
            while True:
                # Create a working copy of ab
                ab = ab_original.copy()
                
                # Add tau to the main diagonal (Row 0)
                if tau > 0:
                    ab[0, :] += tau
                
                try:
                    # Perform Banded Cholesky
                    # lower=True: uses the provided rows as lower diagonals
                    c = spla.cholesky_banded(ab, lower=True, overwrite_ab=False)
                    
                    # Solve H*p = -g
                    pk = spla.cho_solve_banded((c, True), -grad_xk)
                    
                    # If tau was high, print for debug
                    #if tau > 0:
                    #    print(f"  > Matrix corrected with tau={tau:.4e}")
                    break 

                except np.linalg.LinAlgError:
                    # Matrix not positive definite
                    if tau == 0:
                        tau = beta
                    else:
                        tau = 2 * tau
                    
                    # Safety check to avoid infinite loops
                    if tau > 1e10:
                        pk = -grad_xk # Fallback to gradient
                        #print("  ! Fallback to gradient (Ill-conditioned Hessian)")
                        break

            # 3. Backtracking Line Search
            alpha_k = backtracking_line_search(f, gradf, xk, pk, alpha0, rho, c1, btmax)
            
            xk = xk + alpha_k * pk
            
            # Update gradients
            grad_xk = gradf(xk, h, is_h_dynamic=dynamic)
            gradfk_norm = np.linalg.norm(grad_xk)
            fx = f(xk)
            
            k += 1
            #history.append((k, fx, gradfk_norm))
            history.append({'k': k, 'x': xk.copy(), 'fx': fx, 'gnorm': gradfk_norm})
            #if k % 10 == 0: # Print less frequently for cleanliness
            #    print(f"Iter: {k} | f(x): {fx:.4e} | ||g||: {gradfk_norm:.4e}")
        if verbose:
            print(f"Convergence reached at iteration {k}")
        return xk, fx, gradfk_norm, k, history
    
    @staticmethod
    def modified_newton_trig(x0, f, gradf, hessf, alpha0=1.0, kmax=100, tolgrad=1e-5, c1=1e-4, rho=0.5, btmax=20, dynamic=False, h=1e-5, verbose=False):
        '''
        Solves an unconstrained optimization problem assuming a DIAGONAL Hessian structure.
        Aligned with the logic of modified_newton_bro (iterative tau update).

        Since the Hessian is diagonal, the linear system H*p = -g becomes a simple
        element-wise division: p_i = -g_i / H_ii.
        '''
        xk = x0.copy()
        n = len(x0)
        k = 0
        history = []
        
        # Initial state computation
        grad_xk = gradf(xk, h, dynamic)
        gradfk_norm = np.linalg.norm(grad_xk)
        fx = f(xk)

        history.append({'k': 0, 'x': xk.copy(), 'fx': fx, 'gnorm': gradfk_norm})
        if verbose:
            print(f"--------- START: Modified Newton DIAGONAL (Trigonometric). N={n} ---------")
        
        while k < kmax and gradfk_norm > tolgrad:
            # 1. Sparse Hessian Computation
            # For the trigonometric problem, hessf returns a sparse matrix.
            # We immediately extract the diagonal as a 1D numpy array.
            H_sparse = hessf(xk, h, dynamic)
            diag_H = H_sparse.diagonal() 
            
            # --- MODIFIED NEWTON (Iterative Diagonal Correction) ---
            # We seek the minimum tau such that H + tau*I > 0 (Positive Definite).
            # For a diagonal matrix, the condition is simply: min(diag) > 0.
            
            tau = 0.0
            beta = 1e-3
            
            while True:
                # Apply tau to the diagonal
                denominator = diag_H + tau
                
                # Check for positive definiteness
                # Use a small epsilon threshold (e.g., 1e-14) instead of strict 0 for numerical stability
                min_val = np.min(denominator)
                
                if min_val > 1e-14:
                    # SUCCESS: The matrix is positive definite
                    # Solve diagonal system: p_i = -g_i / (H_ii + tau)
                    pk = -grad_xk / denominator
                    
                    # (Optional) Debug print if correction was applied
                    # if tau > 0:
                    #     print(f"  > Matrix corrected with tau={tau:.4e}")
                    break
                
                else:
                    # FAILURE: Matrix is not positive definite (or ill-conditioned)
                    # Increase tau following standard logic
                    if tau == 0:
                        tau = beta
                    else:
                        tau = 2 * tau # Double tau
                    
                    # SAFEGUARD: If tau explodes, exit to avoid infinite loops
                    if tau > 1e10:
                        pk = -grad_xk # Fallback to pure Gradient Descent
                        #print(f"  ! Fallback to gradient (Ill-conditioned Hessian) at step {k}")
                        break

            # 3. Backtracking Line Search
            alpha_k = backtracking_line_search(f, gradf, xk, pk, alpha0, rho, c1, btmax, dynamic, h)
            
            # Update x
            xk = xk + alpha_k * pk
            
            # Update gradients and function value for next iteration
            grad_xk = gradf(xk, h, dynamic)
            gradfk_norm = np.linalg.norm(grad_xk)
            fx = f(xk)
            
            k += 1
            history.append({'k': k, 'x': xk.copy(), 'fx': fx, 'gnorm': gradfk_norm})
            
            #if k % 10 == 0 and verbose:
                #print(f"Iter: {k} | f(x): {fx:.4e} | ||g||: {gradfk_norm:.4e}")
        if verbose:
                print(f"Convergence reached at iteration {k}")
        return xk, fx, gradfk_norm, k, history

    @staticmethod
    def truncated_newton(x0, f, gradf, hessf, alpha0, kmax, tolgrad, c1, rho, btmax, dynamic=False, h=1e-5, verbose=False):
        xk = x0.copy()
        n = len(x0)
        history = []
        
        # Save initial state
        fx = f(xk)
        gradk = gradf(xk, h, is_h_dynamic=dynamic)
        grad_norm = npl.norm(gradk)
        history.append({'k': 0, 'x': xk.copy(), 'fx': fx, 'gnorm': grad_norm})
        
        if verbose:
            print(f"--------- START: Newton Truncated. N={n} ---------")
            
        for k in range(kmax):
            gradk = gradf(xk, h, is_h_dynamic=dynamic)
            grad_norm = npl.norm(gradk)
            
            # Check convergence
            if grad_norm < tolgrad:
                if verbose:
                    print(f"Convergence reached at iteration {k}")
                return xk, f(xk), grad_norm, k, history
            
            B = hessf(xk)

            # Forcing sequence
            eta_k = min(0.5, np.sqrt(grad_norm))
            
            z = np.zeros(n)          # Partial solution (initially 0)
            r = -gradk - B @ z       # Initial residual (-g since z=0)
            d = r.copy()             # Initial search direction
            
            cg_iter = 0
            max_cg_iter = kmax       # Safety limit for CG iterations
            pk = None

            # --- Inner Loop: Conjugate Gradient ---
            while cg_iter < max_cg_iter:
                Bd = B @ d
                dBd = d.T @ Bd

                # Check for negative curvature
                if dBd > 0:
                    alpha_j = (r.T @ r) / dBd
                    z_next = z + alpha_j * d
                    r_next = r - alpha_j * Bd
                
                    beta_cg = (r_next.T @ r_next) / (r.T @ r)
                    d = r_next + beta_cg * d

                else:
                    # Negative curvature detected
                    if cg_iter == 0:
                        pk = -gradk # Fallback to steepest descent
                    else:
                        pk = z      # Use the direction found so far
                    #print('Break: 1')
                    break
                
                z = z_next
                r = r_next
                cg_iter += 1

                # Truncation (stopping) criterion for CG
                if npl.norm(r_next) <= eta_k * grad_norm:
                    pk = z_next
                    #print('Break: 2')
                    break

            if pk is None: # If the loop terminates due to max_iter
                pk = z

            # Backtracking Line Search
            alpha_k = backtracking_line_search(f, gradf, xk, pk, alpha0, rho, c1, btmax, dynamic, h)

            # Update x
            xk = xk + alpha_k * pk

            # Recalculate values for the next step and history logging
            fx = f(xk)
            gradk = gradf(xk, h, is_h_dynamic=dynamic)
            grad_norm = npl.norm(gradk)
            
            history.append({'k': k+1, 'x': xk.copy(), 'fx': fx, 'gnorm': grad_norm})
            #history.append((k, f(xk), grad_norm))

        return xk, fx, grad_norm, k, history