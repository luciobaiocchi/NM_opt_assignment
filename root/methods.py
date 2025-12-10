import numpy as np
import numpy.linalg as npl
import scipy.linalg as spla
from utils import backtracking_line_search

class NewtonMethods:
    
    @staticmethod
    def modified_newton_banded(x0, f, gradf, hessf, alpha0=1.0, kmax=100, tolgrad=1e-5, c1=1e-4, rho=0.5, btmax=20):
        xk = x0.copy()
        n = len(x0)
        k = 0
        history = []
        
        grad_xk = gradf(xk)
        gradfk_norm = np.linalg.norm(grad_xk)
        fx = f(xk)

        history.append({'k': 0, 'x': xk.copy(), 'fx': fx, 'gnorm': gradfk_norm})


        
        print(f"START: Newton Modificato con Cholesky a BANDA. N={n}")
        
        while k < kmax and gradfk_norm > tolgrad:
            # 1. Calcolo Hessiana Sparsa
            H_sparse = hessf(xk)
            
            # 2. Conversione in formato "Banded" per scipy.linalg (Lower form)
            # H_sparse.diagonal(-k) restituisce array di lunghezza N-k
            # Per cholesky_banded (lower=True):
            # Riga 0: Diag principale (Len N)
            # Riga 1: Diag inferiore 1 (Len N-1 -> Pad 1 alla fine)
            # Riga 2: Diag inferiore 2 (Len N-2 -> Pad 2 alla fine)
            
            main_diag = H_sparse.diagonal(0)
            # NOTA: Padding alla fine (append), non all'inizio!
            low1_diag = np.append(H_sparse.diagonal(-1), 0) 
            low2_diag = np.append(H_sparse.diagonal(-2), [0, 0])
            
            # Costruiamo la matrice ab 3xN
            # Usiamo copy() per assicurarci di non modificare main_diag originale nel loop del tau
            ab_original = np.vstack([main_diag, low1_diag, low2_diag])
            
            # --- NEWTON MODIFICATO (Tau adjustment) ---
            tau = 0
            beta = 1e-3
            
            while True:
                # Creiamo una copia di lavoro di ab
                ab = ab_original.copy()
                
                # Aggiungiamo tau alla diagonale principale (Riga 0)
                if tau > 0:
                    ab[0, :] += tau
                
                try:
                    # Eseguiamo Cholesky a Banda
                    # lower=True: usa le righe fornite come diagonali inferiori
                    c = spla.cholesky_banded(ab, lower=True, overwrite_ab=False)
                    
                    # Risolviamo il sistema H*p = -g
                    pk = spla.cho_solve_banded((c, True), -grad_xk)
                    
                    # Se tau era alto, stampiamo per debug
                    if tau > 0:
                        print(f"  > Matrice corretta con tau={tau:.4e}")
                    break 

                except np.linalg.LinAlgError:
                    # Matrice non definita positiva
                    if tau == 0:
                        tau = beta
                    else:
                        tau = 2 * tau
                    
                    # Safety check per evitare loop infiniti
                    if tau > 1e10:
                        pk = -grad_xk # Fallback al gradiente
                        print("  ! Fallback al gradiente (Hessiana mal condizionata)")
                        break

            # 3. Backtracking
            alpha_k = backtracking_line_search(f, gradf, xk, pk, alpha0, rho, c1, btmax)
            
            xk = xk + alpha_k * pk
            
            # Aggiornamento gradienti
            grad_xk = gradf(xk)
            gradfk_norm = np.linalg.norm(grad_xk)
            fx = f(xk)
            
            k += 1
            #history.append((k, fx, gradfk_norm))
            history.append({'k': k, 'x': xk.copy(), 'fx': fx, 'gnorm': gradfk_norm})
            if k % 10 == 0: # Stampiamo meno spesso per pulizia
                print(f"Iter: {k} | f(x): {fx:.4e} | ||g||: {gradfk_norm:.4e}")

        return xk, fx, gradfk_norm, k, history

    @staticmethod
    def truncated_newton(x0, f, gradf, hessf, alpha0, kmax, tolgrad, c1, rho, btmax):
        xk = x0.copy()
        n = len(x0)
        history = []

         # Salviamo lo stato iniziale
        fx = f(xk)
        gradk = gradf(xk)
        grad_norm = npl.norm(gradk)
        history.append({'k': 0, 'x': xk.copy(), 'fx': fx, 'gnorm': grad_norm})


        for k in range(kmax):
            gradk = gradf(xk)
            grad_norm = npl.norm(gradk)
            if grad_norm < tolgrad:
                print(f"Convergenza raggiunta all'iterazione {k}")
                return xk, f(xk), grad_norm, k, history
            
            B= hessf(xk)

            eta_k = min(0.5, np.sqrt(grad_norm))

            z = np.zeros(n)          # Soluzione parziale (inizialmente 0)
            r = -gradk - B @ z    # Residuo iniziale (-g perché z=0)
            d = r.copy()             # Direzione di ricerca iniziale
            
            cg_iter = 0
            max_cg_iter = kmax     # Limite di sicurezza per CG
            pk = None

            while cg_iter < max_cg_iter:
                Bd = B @ d
                dBd = d.T @ Bd

                if dBd > 0:
                    alpha_j = (r.T @ r) / dBd
                    z_next = z + alpha_j * d
                    r_next = r - alpha_j * Bd
                
                    beta_cg = (r_next.T @ r_next) / (r.T @ r)
                    d = r_next + beta_cg * d

                else:
                    if cg_iter == 0:
                        pk = -gradk
                    else:
                        pk = z
                    #print('Break: 1')
                    break
                
                z = z_next
                r = r_next
                cg_iter += 1

                if npl.norm(r_next) <= eta_k * grad_norm:
                    pk = z_next
                    #print('Break: 2')
                    break

            if pk is None: # Se il loop finisce per max_iter
                pk = z

            alpha_k = backtracking_line_search(f, gradf, xk, pk, alpha0, rho, c1, btmax)

            xk = xk + alpha_k * pk

             # Ricalcolo valori per il prossimo step e per la history
            fx = f(xk)
            gradk = gradf(xk)
            grad_norm = npl.norm(gradk)
            
            history.append({'k': k+1, 'x': xk.copy(), 'fx': fx, 'gnorm': grad_norm})
#            history.append((k, f(xk), grad_norm))

        return xk, fx, grad_norm, k, history