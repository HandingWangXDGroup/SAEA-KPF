from copy import deepcopy
from Global.SOLUTION import SOLUTION
from operator import attrgetter
import numpy as np
import random
from Global.crossover import cxSimulatedBinaryBounded
from Global.mutation import mutPolynomialBounded
from Global.selection import selTournament, selRandom


def CreatSolutions(PCheby, PopDec):
    N = np.shape(PopDec)[0]
    population = []
    for i in range(N):
        population.append(SOLUTION(PopDec[i, :], PCheby[i]))
    return population


def Selection(fitness, chromos, Toursize=2):
    population = CreatSolutions(-fitness, chromos)
    selected = selTournament(population, len(population), Toursize, fit_attr="obj")
    selected_clone = [deepcopy(ind) for ind in selected]
    selected_clone = disintegrationSolutions(selected_clone)
    return selected_clone


def disintegrationSolutions(population):
    N = len(population)
    Temp = population[0].dec
    for i in range(1, N):
        Temp = np.append(Temp, population[i].dec, axis=0)
    PopDecs = Temp.reshape(N, -1)
    return PopDecs


def TournamentSelection(K, N, PDec, FrontNo, Crowdis):
    class individual:
        def __init__(self, dec, obj, crowdis) -> None:
            self._front = obj
            self._crowdis = crowdis
            self._dec = dec

        def set_dimension(self, dim):
            self.__dim = dim

        def get_dec(self):
            return self._dec

    Pops = []
    for i in range(N):
        Pops.append(individual(dec=PDec[i, :], obj=FrontNo[i], crowdis=-Crowdis[i]))

    # K* tournament selection
    # chosen = Individuals()
    # chosen.set_dimension(len(PDec[0, :]))
    chosen = []
    for i in range(N):
        aspirants = selRandom(Pops, K)
        ind = min(aspirants, key=attrgetter('_front', '_crowdis'))
        # chosen.AndOne(deepcopy(ind))
        chosen.append(deepcopy(ind))

    Temp = np.array([])
    for i in range(N):
        Temp = np.append(Temp, chosen[i].get_dec(), axis=0)
    newPDec = Temp.reshape(N, -1)
    return newPDec


def tournamentSelection(K, N, fitness):
    parents = np.random.randint(len(fitness), size=(K, N))
    best = np.argmin(fitness.reshape(-1)[parents.reshape(-1)].reshape(K, N), axis=0)
    index = parents.reshape(-1)[np.arange(N) + best * N]
    return index


def Crossover(population, low, up, ProC=1, DisC=20):
    low = low.tolist()
    up = up.tolist()
    (N, D) = population.shape
    off = np.zeros((N, D))
    for i in range(0, N, 2):
        if (np.mod(N, 2) != 0) and (i == N-1):
            ind1 = population[i, :]
            ind2 = population[random.randint(0, max(i-1, 0)), :]
        else:
            ind1 = population[i, :]
            ind2 = population[i+1, :]
        child1, child2 = [deepcopy(ind) for ind in (ind1, ind2)]
        child1, child2 = cxSimulatedBinaryBounded(child1, child2, DisC, ProC, low, up)

        if (np.mod(N, 2) != 0) and (i == N-1):
            off[i, :] = child1.copy()
        else:
            off[i, :] = child1.copy()
            off[i+1, :] = child2.copy()
    return off


def Mutation(population, low, up, ProM=1, DisM=20):
    (N, D) = population.shape
    ProM = ProM/D
    low = low.tolist()
    up = up.tolist()
    off = np.zeros((N, D))
    for i in range(N):
        mutant = deepcopy(population[i, :])
        mutant = mutPolynomialBounded(mutant, DisM, low, up, ProM)

        mutant = np.asarray(list(mutant))
        off[i, :] = mutant.copy()
    return off


def OperatorGA(PDecs, proC, disC, proM, disM, x_opt_max, x_opt_min):
    N, D = np.shape(PDecs)
    # simulated binary crossover
    Offsprings = np.zeros((N, D))
    for i in range(0, N, 2):
        miu = np.random.rand(D)
        beta = np.zeros(D)
        index = miu <= 0.5
        beta[index] = (2*miu[index])**(1/(disC+1))
        beta[~index] = (2-2*miu[~index])**(-1/(disC+1))
        beta = beta*(-1)**np.random.randint(2, size=D)
        beta[np.random.rand(D) > proC] = 1
        if i == N-1:
            i -= 1
        Offsprings[i, :] = (PDecs[i, :] + PDecs[i+1, :])/2 + beta*(PDecs[i, :] - PDecs[i+1, :])/2
        Offsprings[i+1, :] = (PDecs[i, :] + PDecs[i+1, :])/2 - beta*(PDecs[i, :] - PDecs[i+1, :])/2
    # polynominal mutation
    if N == 1:
        MaxValue = x_opt_max.reshape(1, -1)
        MinValue = x_opt_min.reshape(1, -1)
    else:
        MaxValue = np.tile(x_opt_max, (N, 1))
        MinValue = np.tile(x_opt_min, (N, 1))
    k = np.random.rand(N, D)
    miu = np.random.rand(N, D)
    Temp = (k <= proM/D) & (miu < 0.5)  # 变异的基因
    Offsprings[Temp] = Offsprings[Temp] + (MaxValue[Temp] - MinValue[Temp]) * \
        ((2*miu[Temp]+(1-2*miu[Temp])*(1-(Offsprings[Temp]-MinValue[Temp])/(MaxValue[Temp] - MinValue[Temp]))**(disM+1))**(1/(disM+1))-1)
    Temp = (k <= proM/D) & (miu >= 0.5)
    Offsprings[Temp] = Offsprings[Temp] + (MaxValue[Temp] - MinValue[Temp]) * \
        (1-(2*(1-miu[Temp])+2*(miu[Temp]-0.5)*(1-(MaxValue[Temp]-Offsprings[Temp])/(MaxValue[Temp] - MinValue[Temp]))**(disM+1))**(1/(disM+1)))
    # Transboundary processing
    # Offsprings = np.where(Offsprings <= MaxValue, Offsprings, MaxValue)
    # Offsprings = np.where(Offsprings >= MinValue, Offsprings, MinValue)
    Offsprings = np.clip(Offsprings, x_opt_min, x_opt_max)
    return Offsprings
