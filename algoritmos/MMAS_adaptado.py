from ACO_adaptado import *
import numpy as np

def TauMax(rho, f_best):
    return 1.0 / (rho * f_best)

def TauMin(tau_max, n, factor=2):
    return tau_max / (factor * n)

def InitializePheromonesMMAS(n, tau_max):
    return np.full((n,n), tau_max)

def UpdatePheromones(tao, rho, Q, best_ant, cerrado, tau_min, tau_max):
    # Evapora, deposita SOLO la mejor hormiga y recorta tao al rango [min, max]
    tao *= (1 - rho)  
    x = best_ant.x
    deposito = Q / best_ant.f
    for l in range(len(x)-1):
        a, b = x[l], x[l+1]
        tao[a][b] += deposito

    if cerrado:
        a, b = x[-1], x[0]
        tao[a][b] += deposito
    
    np.clip(tao, tau_min, tau_max, out=tao)

def AntColonySystem(N, T, alpha, beta, rho, Q, dist_matrix, cerrado, random_state, inicio,
                         usar_mejor_global=True, factor_tau_min=2, mostrar=True):
    """Max-Min Ant System.

    usar_mejor_global : bool
        Si True, deposita feromona la mejor hormiga GLOBAL encontrada hasta
        el momento (más explotador). Si False, deposita la mejor de CADA
        iteración (más explorador, más habitual en MMAS "puro").
    """
    n = dist_matrix.shape[0]

    assert N>=1, "El número de hormigas debe ser mayor o igual que uno."
    assert T>1, "El número de iteraciones debe ser mayor que uno."
    assert rho>0, "La tasa de evaporación debe ser mayor que cero."
    assert rho<1, "La tasa de evaporación debe ser menor que uno."

    np.random.seed(random_state)
    eta = InitializeHeuristic(dist_matrix)

    # --- Estimación inicial de tau_max con una colonia "semilla" ---
    P = [Ant(n) for _ in range(N)]
    tao_inicial = InitializePheromones(n)  # valor pequeño solo para la 1ª construcción
    ConstructAnts(P, alpha, beta, tao_inicial, eta, inicio)
    EvaluateAnts(P, dist_matrix, cerrado)
    best_ant = Ant(n)
    best_ant_iter = min(P)
    Copia(best_ant, best_ant_iter)

    tau_max = TauMax(rho, best_ant.f)
    tau_min = TauMin(tau_max, n, factor=factor_tau_min)
    tao = InitializePheromonesMMAS(n, tau_max)  # ahora sí, inicialización real de MMAS

    best_outputs, best_outputs_global, progression_bar = IniciaProgreso(P, best_ant, T, mostrar)
    for iter in range(1, T+1):
        tau_max = TauMax(rho, best_ant.f)
        tau_min = TauMin(tau_max, n, factor=factor_tau_min)

        hormiga_que_deposita = best_ant if usar_mejor_global else best_ant_iter
        UpdatePheromones(tao, rho, Q, hormiga_que_deposita, cerrado, tau_min, tau_max)

        ConstructAnts(P, alpha, beta, tao, eta, inicio)
        EvaluateAnts(P, dist_matrix, cerrado)
        best_ant_iter = min(P)
        if best_ant_iter < best_ant:
            Copia(best_ant, best_ant_iter)
        ActualizaProgreso(P, best_ant, best_ant_iter, iter, best_outputs, best_outputs_global, progression_bar)

    progression_bar.close()
    return best_outputs, best_outputs_global, best_ant.f, best_ant.x