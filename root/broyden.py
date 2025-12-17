import numpy as np
from scipy.sparse import diags
from scipy import sparse

class BroydenProblem:
    @staticmethod
    def func(x):
        """
        Calcola la funzione obiettivo tridiagonale di Broyden.
        
        F(x) = 0.5 * sum(f_k(x)^2)
        
        Dove:
        f_k(x) = (3 - 2*x_k)*x_k - x_{k-1} - 2*x_{k+1} + 1
        con x_0 = x_{n+1} = 0
        """
        x = np.asarray(x)
        
        # Array con zero all'inizio e alla fine
        # x_pad diventerà: [0, x_1, x_2, ..., x_n, 0], 
        # serve per poi fare i calcoli con vettori piuttosto che cilco
        x_pad = np.pad(x, (1, 1), mode='constant', constant_values=0)
        
        # Definiamo i vettori spostati per il calcolo vettorizzato
        x_k = x_pad[1:-1]       # x_k corrisponde agli elementi centrali (l'input originale)
        x_prev = x_pad[0:-2]    # x_{k-1} corrisponde agli elementi spostati a sinistra
        x_next = x_pad[2:]      # x_{k+1} corrisponde agli elementi spostati a destra
        
        # in questo modo:
        # x_prev [0, 1, 2, 3]
        # x_k    [1, 2, 3, 4]
        # x_next [2, 3, 4, 0]
        
        
        # Per non fare cicli calcolo del vettore f_k per tutti i k da 1 a n simultaneamente
        f_vec = (3 - 2 * x_k) * x_k - x_prev - 2 * x_next + 1
        
        # Calcolo finale di F(x)
        F_x = 0.5 * np.sum(f_vec**2)
        
        return F_x

    @staticmethod
    def gradient(x, h=1e-5, is_h_dynamic=True):
        n = len(x)
        xp = np.pad(x, (2, 2), mode='constant', constant_values=0)
        
        def get_term_val(val_k, val_km1, val_kp1):
            return ((3 - 2 * val_k) * val_k - val_km1 - 2 * val_kp1 + 1)

        idx = np.arange(n)
        k = idx + 2
        
        v_k   = xp[k]
        v_km1 = xp[k-1]; v_km2 = xp[k-2]
        v_kp1 = xp[k+1]; v_kp2 = xp[k+2]

        def calc_local_energy_change(perturbation):
            val_k_mod = v_k + perturbation
            
            t_im1 = get_term_val(v_km1, v_km2, val_k_mod)
            t_im1[0] = 0.0 # Bordo sinistro f_{-1}
            
            t_i = get_term_val(val_k_mod, v_km1, v_kp1)
            
            t_ip1 = get_term_val(v_kp1, val_k_mod, v_kp2)
            t_ip1[-1] = 0.0 # Bordo destro f_n
            
            return 0.5 * (t_im1**2 + t_i**2 + t_ip1**2)
        if is_h_dynamic:
            h = h * np.abs(x)
            #print("broyden, gradient with dynamic h")
        #print(f"HHHHhhhhhhHHHhHHHHHH{h}")
        #print(f"XXXXXXXXXXXXXXXXXXXX{x}")
        #print(f"ABSSOEJFONEFJNFJEJFN{np.abs(x)}")
        #print(f"JHSHHSHSHSHSHSHSHSHS{h * np.abs(x)}")
        E_plus  = calc_local_energy_change(h)
        E_minus = calc_local_energy_change(-h)
        
        return (E_plus - E_minus) / (2 * h)

    @staticmethod
    def hessian_sparse(x, h=1e-5):
        n = len(x)
        
        # Pad x per gestire i vicini senza if/else
        xp = np.pad(x, (2, 2), mode='constant', constant_values=0)
        
        # Helper: calcola il termine grezzo
        def get_term_val(val_k, val_km1, val_kp1):
            return ((3 - 2 * val_k) * val_k - val_km1 - 2 * val_kp1 + 1)

        # ==========================================
        # 1. DIAGONALE PRINCIPALE (i, i)
        # ==========================================
        idx = np.arange(n)
        k = idx + 2 # Mappa indici su xp
        
        # Carichiamo i valori dai vicini
        v_k   = xp[k]
        v_km1 = xp[k-1]; v_km2 = xp[k-2]
        v_kp1 = xp[k+1]; v_kp2 = xp[k+2]
        
        # Funzione locale per calcolare l'energia di un punto i
        def calc_E_local(shift_val):
            # Termine i-1 (indice k-1):
            # ATTENZIONE: Per i=0, questo è il termine -1. Deve essere 0.
            t_im1 = get_term_val(v_km1, v_km2, v_k + shift_val)
            t_im1[0] = 0.0 # FIX: Maschera bordo sinistro

            # Termine i (indice k):
            t_i   = get_term_val(v_k + shift_val, v_km1, v_kp1)
            
            # Termine i+1 (indice k+1):
            # ATTENZIONE: Per i=n-1, questo è il termine n. Deve essere 0.
            t_ip1 = get_term_val(v_kp1, v_k + shift_val, v_kp2)
            t_ip1[-1] = 0.0 # FIX: Maschera bordo destro
            
            return 0.5 * (t_im1**2 + t_i**2 + t_ip1**2)

        E_plus   = calc_E_local(h)
        E_minus  = calc_E_local(-h)
        E_center = calc_E_local(0)
        
        main_diag = (E_plus - 2*E_center + E_minus) / (h**2)

        # ==========================================
        # 2. OFF-DIAGONAL 1 (i, i+1)
        # ==========================================
        idx1 = np.arange(n - 1)
        k = idx1 + 2
        
        v_k = xp[k]; v_km1 = xp[k-1]; v_km2 = xp[k-2]
        v_kp1 = xp[k+1]; v_kp2 = xp[k+2]; v_kp3 = xp[k+3]
        
        def calc_E_pair1(p1, p2):
            val_k_mod = v_k + p1
            val_kp1_mod = v_kp1 + p2
            
            # Termine i-1 (idx -1 se i=0) -> FIX: Mask [0]
            t1 = get_term_val(v_km1, v_km2, val_k_mod)
            t1[0] = 0.0 
            
            # Termine i
            t2 = get_term_val(val_k_mod, v_km1, val_kp1_mod)
            
            # Termine i+1
            t3 = get_term_val(val_kp1_mod, val_k_mod, v_kp2)
            
            # Termine i+2 (idx n se i=n-2) -> FIX: Mask [-1]
            t4 = get_term_val(v_kp2, val_kp1_mod, v_kp3)
            t4[-1] = 0.0
            
            return 0.5 * (t1**2 + t2**2 + t3**2 + t4**2)

        val = (calc_E_pair1(h, h) - calc_E_pair1(h, -h) - calc_E_pair1(-h, h) + calc_E_pair1(-h, -h)) / (4 * h**2)
        upper_diag1 = val

        # ==========================================
        # 3. OFF-DIAGONAL 2 (i, i+2)
        # ==========================================
        idx2 = np.arange(n - 2)
        k = idx2 + 2
        
        v_k = xp[k]; v_km1 = xp[k-1]; v_km2 = xp[k-2]
        v_kp1 = xp[k+1]; v_kp2 = xp[k+2]; v_kp3 = xp[k+3]; v_kp4 = xp[k+4]

        def calc_E_pair2(p1, p2):
            val_k_mod = v_k + p1
            val_kp2_mod = v_kp2 + p2
            
            # Termine i-1 (idx -1 se i=0) -> FIX: Mask [0]
            t1 = get_term_val(v_km1, v_km2, val_k_mod)
            t1[0] = 0.0
            
            # Termine i
            t2 = get_term_val(val_k_mod, v_km1, v_kp1)
            
            # Termine i+1
            t3 = get_term_val(v_kp1, val_k_mod, val_kp2_mod)
            
            # Termine i+2
            t4 = get_term_val(val_kp2_mod, v_kp1, v_kp3)
            
            # Termine i+3 (idx n se i=n-3) -> FIX: Mask [-1]
            t5 = get_term_val(v_kp3, val_kp2_mod, v_kp4)
            t5[-1] = 0.0
            
            return 0.5 * (t1**2 + t2**2 + t3**2 + t4**2 + t5**2)

        val = (calc_E_pair2(h, h) - calc_E_pair2(h, -h) - calc_E_pair2(-h, h) + calc_E_pair2(-h, -h)) / (4 * h**2)
        upper_diag2 = val

        return diags(
            [upper_diag2, upper_diag1, main_diag, upper_diag1, upper_diag2], 
            [-2, -1, 0, 1, 2], 
            shape=(n, n)
        )
    
    @staticmethod
    def get_jacobian_sparse(x):
        n = len(x)
        
        # Diagonale principale
        main_diag = 3 - 4 * x
        
        # Diagonali off-set (costanti)
        lower_diag = -1 * np.ones(n - 1)
        upper_diag = -2 * np.ones(n - 1)
        
        # Costruzione della matrice sparsa (formato dia o csr)
        diagonals = [main_diag, lower_diag, upper_diag]
        offsets = [0, -1, 1]
        
        J = sparse.diags(diagonals, offsets, shape=(n, n), format='csr')
        return J
    
    @staticmethod
    def get_residual_vector(x):
        """Restituisce il vettore f(x) invece della norma scalare."""
        x = np.asarray(x)
        x_pad = np.pad(x, (1, 1), mode='constant', constant_values=0)
        
        x_k = x_pad[1:-1]
        x_prev = x_pad[0:-2]
        x_next = x_pad[2:]
        
        # Calcolo vettorizzato del sistema di equazioni
        f_vec = (3 - 2 * x_k) * x_k - x_prev - 2 * x_next + 1
        return f_vec

    @staticmethod
    def exact_gradient(x):
        """
        Calcola il gradiente esatto analitico di F(x).
        Grad F(x) = J^T * f(x)
        """
        f_vec = BroydenProblem.get_residual_vector(x)
        J = BroydenProblem.get_jacobian_sparse(x)
        
        # Moltiplicazione matrice trasposta per vettore
        # J è in formato CSR o DIA, la trasposta è efficiente
        grad = J.T @ f_vec
        return grad
    
    @staticmethod
    def hessian_with_jacobian(x, h_arg=1e-5, ):
        """
        Calcola l'Hessiana approssimata FD supportando h scalare o vettoriale.
        h_arg: può essere float (es. 1e-4) o array numpy di dimensione n (h_i per ogni x_i).
        """
        n = len(x)
        
        # Gestione h scalare vs vettoriale
        if np.isscalar(h_arg):
            h_vec = np.full(n, h_arg)
        else:
            h_vec = np.asarray(h_arg)
            # Safety check: se h_i è 0 (perché x_i=0), imposta un minimo per evitare divisioni per zero
            h_vec[h_vec == 0] = 1e-16

        diff_vectors = []
        
        # Loop Coloring (Stride 5)
        for s in range(5):
            # Crea maschera per gli indici perturbati in questo gruppo
            mask = np.zeros(n)
            mask[s::5] = 1.0
            
            # Calcola il vettore di perturbazione specifico: 
            # dove mask è 1, usiamo il valore corrispondente di h_vec
            p = mask * h_vec 
            
            # Differenza finita centrata
            # Calcoliamo solo la differenza grezza qui (numerator)
            g_plus = BroydenProblem.exact_gradient(x + p)
            g_minus = BroydenProblem.exact_gradient(x - p)
            
            diff = (g_plus - g_minus) 
            diff_vectors.append(diff)
            
        # --- RICOSTRUZIONE E NORMALIZZAZIONE ---
        idx_n = np.arange(n)
        
        # 1. Diagonale Principale (offset 0): H[i, i]
        # Derivata rispetto a x_i -> Dividiamo per 2*h_i
        raw_d0 = np.choose(idx_n % 5, diff_vectors)
        d0 = raw_d0 / (2 * h_vec)
        
        # 2. Diagonale Superiore 1 (offset +1): H[i, i+1]
        # Derivata rispetto a x_{i+1} -> Dividiamo per 2*h_{i+1}
        raw_d1 = np.choose((idx_n[:-1] + 1) % 5, [v[:-1] for v in diff_vectors])
        d1 = raw_d1 / (2 * h_vec[1:]) # Nota lo slicing su h_vec
        
        # 3. Diagonale Inferiore 1 (offset -1): H[i, i-1]
        # Derivata rispetto a x_{i-1} -> Dividiamo per 2*h_{i-1}
        raw_dm1 = np.choose((idx_n[1:] - 1) % 5, [v[1:] for v in diff_vectors])
        dm1 = raw_dm1 / (2 * h_vec[:-1])
        
        # 4. Diagonale Superiore 2 (offset +2): H[i, i+2]
        # Derivata rispetto a x_{i+2}
        raw_d2 = np.choose((idx_n[:-2] + 2) % 5, [v[:-2] for v in diff_vectors])
        d2 = raw_d2 / (2 * h_vec[2:])
        
        # 5. Diagonale Inferiore 2 (offset -2): H[i, i-2]
        # Derivata rispetto a x_{i-2}
        raw_dm2 = np.choose((idx_n[2:] - 2) % 5, [v[2:] for v in diff_vectors])
        dm2 = raw_dm2 / (2 * h_vec[:-2])
        
        return diags(
            [dm2, dm1, d0, d1, d2], 
            [-2, -1, 0, 1, 2], 
            shape=(n, n), 
            format='csr'
        )