from algoritmos.ACO_adaptado import *

def UpdatePheromones(P, tao, rho, Q, cerrado, best_ant=None, e=0):
    # "e" es un hiperparámetro y debe ser mayor que 0 para que EAS no funcione exactamente igual que un ACO. 
    # Pruebas: e=0, e=N, e=2N...
    tao *= (1 - rho)  
    for I in P:
        x = I.x
        deposito = Q / I.f
        for l in range(len(x)-1):
            a, b = x[l], x[l+1]
            tao[a][b] += deposito

        if cerrado:
            a, b = x[-1], x[0]
            tao[a][b] += deposito

    # Se añade un peso extra (elite) a la mejor hormiga
    if best_ant is not None and e > 0:
        x = best_ant.x
        deposito_elite = e * Q / best_ant.f
        for l in range(len(x)-1):
            a, b = x[l], x[l+1]
            tao[a][b] = deposito_elite
        
        if cerrado:
            a, b = x[-1], x[0]
            tao[a][b] = deposito_elite

def AntsColonyOptimization(N, T, alpha, beta, rho, Q, dist_matrix, cerrado, random_state, inicio, e=0):

    n = dist_matrix.shape[0]

    assert N>=1, "El número de hormigas debe ser mayor o igual que uno."
    assert T>1, "El número de iteraciones debe ser mayor que uno."
    assert rho>0, "La tasa de evaporación debe ser mayor que cero."
    assert rho<1, "La tasa de evaporación debe ser menor que uno."

    np.random.seed(random_state)

    tao = InitializePheromones(n)
    eta = InitializeHeuristic(dist_matrix)

    P = [Ant(n) for _ in range(N)]
    ConstructAnts(P, alpha, beta, tao, eta, inicio)
    EvaluateAnts(P, dist_matrix, cerrado)
    best_ant = Ant(n)
    Copia(best_ant, min(P))

    best_outputs, best_outputs_global, progression_bar = IniciaProgreso(P, best_ant, T)
    for iter in range(1, T+1):
        UpdatePheromones(P, tao, rho, Q, cerrado, best_ant=best_ant, e=e)
        ConstructAnts(P, alpha, beta, tao, eta, inicio)
        EvaluateAnts(P, dist_matrix, cerrado)
        best_ant_iter = min(P)
        if best_ant_iter < best_ant:
            Copia(best_ant, best_ant_iter)
        ActualizaProgreso(P, best_ant, best_ant_iter, iter, best_outputs, best_outputs_global, progression_bar)
    return best_outputs, best_outputs_global, best_ant.f, best_ant.x