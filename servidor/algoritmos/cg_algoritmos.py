import numpy as np


def reconstruir_cgne(g_vec: np.ndarray, H: np.ndarray, lam: float, max_iter: int, tol: float) -> tuple[np.ndarray, int]:
    print(f"Iniciando algoritmo CGNE (lambda={lam:.2e}, max_iter={max_iter}, tol={tol:.2e})...")

    Ht = H.T
    b = Ht @ g_vec
    x = np.zeros(H.shape[1])

    r = b - (Ht @ (H @ x) + lam * x)
    d = r.copy()

    norma_b = np.linalg.norm(b)
    MIN_ITER = 10 

    num_iteracoes = 0

    for i in range(max_iter):
        num_iteracoes = i + 1

        q = Ht @ (H @ d) + lam * d
        denom = d @ q
        if abs(denom) < 1e-20:
            print(f"CGNE Convergência: Denominador de alpha muito pequeno ({denom:.2e}) na iteração {num_iteracoes}.")
            break

        alpha = (r @ r) / denom
        x += alpha * d
        r_new = b - (Ht @ (H @ x) + lam * x)

        norma_res_new = np.linalg.norm(r_new)
        if norma_res_new / norma_b < tol and i >= MIN_ITER:
            print(f"CGNE Convergência por tolerância relativa ({norma_res_new:.2e} / {norma_b:.2e} < {tol:.2e}) na iteração {num_iteracoes}.")
            break

        beta = (r_new @ r_new) / (r @ r)
        d = r_new + beta * d
        r = r_new

    if num_iteracoes >= max_iter:
        print(f"CGNE Não convergiu em {max_iter} iterações. Norma do resíduo final: {norma_res_new:.2e}.")

    return x, num_iteracoes


def reconstruir_cgnr(g_vec: np.ndarray, H: np.ndarray, lam: float, max_iter: int, tol: float) -> tuple[np.ndarray, int]:
    
    print(f"Iniciando algoritmo CGNR (lambda={lam:.2e}, max_iter={max_iter}, tol={tol:.2e})...")
    
    Ht = H.T
    
    f  = np.zeros(H.shape[1]) # f_0 = 0
    
    r = g_vec - H @ f # r_0 para o sistema original Hf=g
    z = Ht @ r        # z_0 = Ht @ r_0 para o sistema normal
    p = z.copy()      # p_0 = z_0
    
    # Norma inicial do resíduo r (do sistema Hf=g)
    norma_res_inicial_r = np.linalg.norm(r) 
    num_iteracoes = 0

    for i in range(max_iter):
        num_iteracoes = i + 1

        # w = H @ p
        w = H @ p
        
        # Calcular alpha
        # Numerador: z @ z (norma quadrada de z)
        # Denominador: w @ w (norma quadrada de w)
        numerador_alpha = z @ z
        denom_alpha = w @ w
        if abs(denom_alpha) < 1e-20:
            print(f"CGNR Convergência: Denominador de alpha muito pequeno ({denom_alpha:.2e}) na iteração {num_iteracoes}.")
            break

        alpha = numerador_alpha / denom_alpha

        # Atualizar solução f
        f += alpha * p

        # Atualizar resíduo r e z_new
        r_new = r - alpha * w # r_new = r_old - alpha * w
        z_new = Ht @ r_new    # z_new = Ht @ r_new

        # Critério de parada: norma do resíduo r_new (do sistema Hf=g)
        norma_res_new = np.linalg.norm(r_new)
        if norma_res_new < tol:
            print(f"CGNR Convergência por tolerância ({norma_res_new:.2e} < {tol:.2e}) na iteração {num_iteracoes}.")
            break
        
        # Calcular beta
        # Numerador: z_new @ z_new (norma quadrada de z_new)
        # Denominador: z @ z (norma quadrada de z)
        numerador_beta = z_new @ z_new
        denom_beta = z @ z
        if abs(denom_beta) < 1e-20:
            print(f"CGNR Convergência: Denominador de beta muito pequeno ({denom_beta:.2e}) na iteração {num_iteracoes}.")
            break
        
        beta = numerador_beta / denom_beta

        # Atualizar direção de busca p
        p = z_new + beta * p # p_new = z_new + beta * p_old

        # Atualizar r e z para a próxima iteração
        r = r_new
        z = z_new

    if num_iteracoes >= max_iter:
        print(f"CGNR Não convergiu em {max_iter} iterações. Norma do resíduo final: {norma_res_new:.2e}.")
    return f, num_iteracoes