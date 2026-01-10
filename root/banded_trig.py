import numpy as np
from scipy.sparse import diags


class BandedTrigonometric:
    @staticmethod
    def func(x):
        '''
        calcola la funzione obiettivo del Banded Tridiagonal Problem
        
        F(x) = sum(i * [(1 - cos(x_i)) + sin(x_{i-1}) - sin(x_{i+1})])
        
        x_0 = x_{n+1} = 0,

        bar(x)_i = 1, i >= 1

        Si puo semplificare algebricamente e avere:

        F(x) = sum(i * (1-cos(x_i))) + 2 * sum(sin(x_k)) - (n-1) * sin(x_n)
        '''
        x = np.asarray(x)
        n = len(x)
        
        # 1. Parte del Coseno: Dipende dall'indice 'i'
        # Formula: sum( i * (1 - cos(x_i)) )
        i_vec = np.arange(1, n + 1)
        cos_part = np.sum(i_vec * (1 - np.cos(x)))
        
        # 2. Parte dei Seni: Semplificata algebricamente
        # Formula: 2 * sum(sin(x_1...x_{n-1})) - (n-1)*sin(x_n)
        # Nota: x[:-1] prende tutti tranne l'ultimo
        sin_sum_inner = 2 * np.sum(np.sin(x[:-1])) 
        sin_last_term = - (n - 1) * np.sin(x[-1])
        
        sin_part = sin_sum_inner + sin_last_term
        
        F_x = cos_part + sin_part

        return F_x
    
    @staticmethod
    def gradient_fd(x, h=1e-5, is_h_dynamic=False):
        """
        Calcola il gradiente usando Differenze Finite Centrate 
        sulla versione SEMPLIFICATA algebricamente della funzione.
        Non richiede padding né accesso ai vicini.
        """
        x = np.asarray(x)
        n = len(x)
        
        # Indici [1, 2, ..., n] per il coefficiente del coseno
        ids = np.arange(1, n + 1)
        
        def get_local_contributions(val):
            """
            Calcola il vettore dei contributi locali per un dato vettore val.
            Restituisce un array di dimensione n.
            """
            # 1. Parte Coseno: k * (1 - cos(x_k))
            cos_part = ids * (1 - np.cos(val))
            
            # 2. Parte Seno: 
            #    2 * sin(x_k) per tutti
            #    -(n-1) * sin(x_n) per l'ultimo
            # 1. Creiamo un array vuoto della stessa forma di val
            sin_part = np.empty_like(val)

            # 2. Riempiamo tutti tranne l'ultimo (da 0 a n-1)
            # Calcola il seno solo per gli elementi necessari
            sin_part[:-1] = 2 * np.sin(val[:-1])

            # 3. Riempiamo l'ultimo elemento specificamente
            sin_part[-1] = -(n - 1) * np.sin(val[-1])
            
            return cos_part + sin_part

        # Calcolo Vettorizzato
        # Calcoliamo i termini perturbati (x+h) e (x-h) per tutto il vettore insieme
        if is_h_dynamic:
            h = h * np.abs(x)
        terms_plus = get_local_contributions(x + h)
        terms_minus = get_local_contributions(x - h)
        
        # Differenza Centrata
        # Gradiente = (f(x+h) - f(x-h)) / 2h
        grad = (terms_plus - terms_minus) / (2 * h)
        
        return grad
    
    @staticmethod
    def hessian_fd(x, h=1e-5, is_h_dynamic=False):
        # THIS IS HARDCODED IN ORDER TO LET THE TESTING BE EASIER
        h=1e-5
        is_h_dynamic=False
        
        x = np.asarray(x)
        n = len(x)
        ids = np.arange(1, n + 1)
        
        # Coefficienti per la parte seno
        alpha = np.full(n, 2.0)
        alpha[-1] = -(n - 1)
        
        def get_local_val(val):
            # Calcola il valore "locale" della funzione per ogni componente
            # F_k = k*(1 - cos(x)) + alpha*sin(x)
            term_cos = ids * (1 - np.cos(val))
            term_sin = alpha * np.sin(val)
            return term_cos + term_sin

        # 1. Calcoliamo f(x), f(x+h), f(x-h)
        val_curr = get_local_val(x)
        val_plus = get_local_val(x + h)
        val_minus = get_local_val(x - h)
                
        # 2. Formula Differenze Finite per derivata seconda
        h_diag = (val_plus - 2 * val_curr + val_minus) / (h ** 2)
        
        return diags(h_diag) #, 0, shape= (n,n))
    
    @staticmethod
    def gradient_exact(x, h=None, is_h_dynamic=None):
        """
        Gradiente ESATTO della funzione semplificata:
        F(x) = sum_{i=1}^n i(1-cos x_i) + 2 sum_{i=1}^{n-1} sin x_i - (n-1) sin x_n

        g_i = i*sin(x_i) + 2*cos(x_i)            per i = 1,...,n-1
        g_n = n*sin(x_n) - (n-1)*cos(x_n)
        """
        x = np.asarray(x)
        n = len(x)
        ids = np.arange(1, n + 1)

        grad = ids * np.sin(x)
        grad[:-1] += 2.0 * np.cos(x[:-1])
        grad[-1]  += -(n - 1) * np.cos(x[-1])
        return grad

    @staticmethod
    def hessian_exact(x, h=None, is_h_dynamic=None):
        """
        Hessiana ESATTA (diagonale) della funzione semplificata:

        H_ii = i*cos(x_i) - 2*sin(x_i)           per i = 1,...,n-1
        H_nn = n*cos(x_n) + (n-1)*sin(x_n)
        """
        x = np.asarray(x)
        n = len(x)
        ids = np.arange(1, n + 1)

        hdiag = ids * np.cos(x)
        hdiag[:-1] += -2.0 * np.sin(x[:-1])
        hdiag[-1]  += (n - 1) * np.sin(x[-1])

        return diags(hdiag, 0, shape=(n, n), format="csr")

    @staticmethod
    def hessian_with_jacobian(x, h=1e-5, is_h_dynamic=False, eps_min=1e-16):
        """
        Versione efficiente se ti basta la diagonale:
        H_ii ≈ ( [∇F(x + h_i e_i)]_i - [∇F(x - h_i e_i)]_i ) / (2 h_i)
        """
        x = np.asarray(x, dtype=float)
        n = x.size
        

        if is_h_dynamic:
            h = h * np.abs(x)
            h = np.maximum(h, eps_min)
        else:
            h = np.full(n, h, dtype=float)

        g_plus  = BandedTrigonometric.gradient_exact(x + h)
        g_minus = BandedTrigonometric.gradient_exact(x - h)

        hdiag = (g_plus - g_minus) / (2.0 * h)   # elementwise = diag(H)
        return diags(hdiag, 0, shape=(n, n), format='csr')