import numpy as np

class BandedTrigonometric:
    
    @staticmethod
    def gradient_exact(x):
        """
        Calcola il gradiente analitico esatto (Ground Truth).
        Formula: dF/dx_k = k*sin(x_k) + 2*cos(x_k) [con correzione ultimo elemento]
        """
        x = np.asarray(x)
        n = len(x)
        k_vec = np.arange(1, n + 1)
        
        # 1. Parte derivata dal Coseno: k * sin(x)
        grad_cos_part = k_vec * np.sin(x)
        
        # 2. Parte derivata dal Seno
        # Coefficienti: 2 per tutti, -(n-1) per l'ultimo
        coeffs = np.full(n, 2.0)
        coeffs[-1] = -(n - 1)
        
        grad_sin_part = coeffs * np.cos(x)
        
        return grad_cos_part + grad_sin_part

    @staticmethod
    def gradient_fd(x, h=1e-5):
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
        terms_plus = get_local_contributions(x + h)
        terms_minus = get_local_contributions(x - h)
        
        # Differenza Centrata
        # Gradiente = (f(x+h) - f(x-h)) / 2h
        grad = (terms_plus - terms_minus) / (2 * h)
        
        return grad

def run_test():
    # 1. Configurazione del Test
    N = 1000     # Dimensione del problema (prova anche 1000)
    h = 1e-5       # Passo per le differenze finite
    np.random.seed(42) # Per riproducibilità
    
    # Generiamo un punto casuale x
    x_test = np.random.randn(N)
    
    print(f"--- TEST GRADIENTE (N={N}, h={h}) ---")
    
    # 2. Calcolo dei Gradienti
    grad_num = BandedTrigonometric.gradient_fd(x_test, h=h)
    grad_ana = BandedTrigonometric.gradient_exact(x_test)
    
    # 3. Analisi dell'Errore
    # Errore assoluto: |Num - Ana|
    abs_error = np.abs(grad_num - grad_ana)
    max_error = np.max(abs_error)
    
    # Errore relativo (norma): ||Num - Ana|| / ||Ana||
    rel_error = np.linalg.norm(grad_num - grad_ana) / np.linalg.norm(grad_ana)

    # 4. Stampa Risultati Dettagliati (solo primi 5 elementi)
    print("\nConfronto primi 5 elementi:")
    print(f"{'Index':<6} | {'Numerico':<12} | {'Esatto':<12} | {'Diff':<12}")
    print("-" * 50)
    for i in range(min(5, N)):
        print(f"{i:<6} | {grad_num[i]:.6f}     | {grad_ana[i]:.6f}     | {abs_error[i]:.2e}")
    
    print("-" * 50)
    print(f"\nErrore Massimo Assoluto: {max_error:.2e}")
    print(f"Errore Relativo (Norma): {rel_error:.2e}")
    
    # 5. Verdetto Finale
    # Consideriamo il test passato se l'errore è sotto una soglia ragionevole (es. 1e-8)
    # Nota: con h=1e-5, ci aspettiamo un errore intorno a 1e-9 o 1e-10.
    if np.allclose(grad_num, grad_ana, rtol=1e-4, atol=1e-6):
        print("\n✅ TEST PASSATO! Il gradiente numerico coincide con quello esatto.")
    else:
        print("\n❌ TEST FALLITO! C'è troppa discrepanza.")

if __name__ == "__main__":
    run_test()