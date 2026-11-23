import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

class Ant:
    def __init__(self, n):
        self.x = np.full(n, -1)
        self.f = np.inf
    def __lt__(self, ant):
        return self.f < ant.f

def LoadDistanceMatrix(file_path):
    df = pd.read_csv(file_path, index_col=0).values.astype(float)
    assert np.isfinite(df).all(), "Hay pares de nodos sin camino (inf): Limpia la matriz antes"
    return df

def f(x, dist_matrix, cerrado):
    # x: array de índices (permutación de 0..n-1)
    coste = sum(dist_matrix[x[l], x[l+1]] for l in range(len(x)-1))
    if cerrado:
        coste += dist_matrix[x[-1], x[0]]
    return coste

def InitializePheromones(n, tao0 = 1E-3):
    return np.full((n, n), tao0)

def InitializeHeuristic(dist_matrix):
    n = dist_matrix.shape[0]
    eta = 1 / (dist_matrix + np.eye(n) * 1e-10)
    np.fill_diagonal(eta, 0.0)
    return eta

def ConstructAnts(P, alpha, beta, tao, eta, inicio):
    n = tao.shape[0]
    for I in P:
        I.x[0] = inicio
        no_visitados = list(range(n))
        no_visitados.remove(inicio)
        for l in range(1, n):
            actual = I.x[l-1]
            pesos = np.array([tao[actual][j]**alpha * eta[actual][j]**beta for j in no_visitados])
            suma = pesos.sum()
            if suma == 0:
                siguiente = np.random.choice(no_visitados)
            else:
                p = pesos / suma
                siguiente = np.random.choice(no_visitados, p=p)
            I.x[l] = siguiente
            no_visitados.remove(siguiente)

def EvaluateAnts(P, dist_matrix, cerrado):
    for I in P:
        I.f = f(I.x, dist_matrix, cerrado)

def Copia(ant1, ant2):
    ant1.x[:] = ant2.x[:]
    ant1.f = ant2.f

def UpdatePheromones(P, tao, rho, Q, cerrado):
    tao *= (1 - rho)  # evaporación
    for I in P:
        x = I.x
        deposito = Q / I.f
        for l in range(len(x)-1):
            a, b = x[l], x[l+1]
            tao[a][b] += deposito

        if cerrado:
            a, b = x[-1], x[0]
            tao[a][b] += deposito

def IniciaProgreso(P, best_global, T):
    best_outputs = [0]*(T+1)
    best_outputs_global = [0]*(T+1)
    progression_bar = tqdm()
    ActualizaProgreso(P, best_global, best_global, 0, best_outputs, best_outputs_global, progression_bar)
    return best_outputs, best_outputs_global, progression_bar

def ActualizaProgreso(P, best_global, best_individual, iter, best_outputs, best_outputs_global, progression_bar):
    progression_bar.set_description("Iteration: %s Best global (m): %.1f Best iter (m): %.1f" % (str(iter), best_global.f, best_individual.f))
    best_outputs[iter] = best_individual.f
    best_outputs_global[iter] = best_global.f
    progression_bar.update(1)

def AntsColonyOptimization(N, T, alpha, beta, rho, Q, dist_matrix, cerrado, random_state, inicio):
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
        UpdatePheromones(P, tao, rho, Q, cerrado)
        ConstructAnts(P, alpha, beta, tao, eta, inicio)
        EvaluateAnts(P, dist_matrix, cerrado)
        best_ant_iter = min(P)
        if best_ant_iter < best_ant:
            Copia(best_ant, best_ant_iter)
        ActualizaProgreso(P, best_ant, best_ant_iter, iter, best_outputs, best_outputs_global, progression_bar)
    return best_outputs, best_outputs_global, best_ant.f, best_ant.x

def VisualizaEvolucion(best_outputs, ylabel):
    fig, ax = plt.subplots()
    ax.set_title("Algorithm evolution")
    plt.plot(best_outputs)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.show()