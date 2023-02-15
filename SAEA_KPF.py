# -*- encoding: utf-8 -*-
'''
@File    :   SAEA_KPF.py
@Time    :   2023/01/22 10:04:31
@Author  :   jftang
'''


import numpy as np
from Global.UniformPoint import generatorPoints
from Global.NondominatedSorting import ENS_SS_NDSort
from Global.SOLUTION import SOLUTIONSET, Properties
from PF_estimationUsing_UH import estimationUsingUnitHyperplane
from Knees import CHIM, local_BiDominance, KneePointIdentification_TradeUtility
from Knees import identificationCurvature, acuteCluster, getExtremePoints, AssociationWeights
from Knees import cal_DistributedDensity, calculateDistMatrix
import math
import random
from Global.selfGA import Selection, Crossover, Mutation
from smt.surrogate_models import KRG
from scipy.stats import norm
import time
import warnings
from Global.SOLUTION import err_print
warnings.filterwarnings("ignore")


# According to the sample set, the Pareto front is fitted to identify the direction of the knee point
def scenarioJudgment(Pops, Vsep, alpha, r, gamma, eva, maxeva, N0, W, ts={'t1': 0, 't2': 0, 'r1': 0, 'r2': 0}):
    Objs = Pops.objs()
    t1, t2, cM = 4, 6, 0.85
    acmin = 0.21

    if np.mod(eva-N0, t1) == 0:
        ts['r1'] += 1
        FrontNo, maxF = ENS_SS_NDSort(Objs, np.shape(Objs)[0])
        NDIndex = np.where(FrontNo == 1)[0]
        Dindex = np.where(FrontNo != 1)[0]
        Addpropers = Properties(Pops.solutions[NDIndex], 'add')
        Addpropers += 1     # non-domination attribution +1
        Pops.updateProperties(NDIndex, Addpropers, 'add')
        Addpropers_d = np.ones((len(Dindex), 1)) * -1   # domination attribution -1
        Pops.updateProperties(Dindex, Addpropers_d, 'add')
        # the stochastic selection of reference vector from the sub-region
        vectors = np.array([Vsep[i].vector.tolist() for i in range(len(Vsep))])
        # arg_index = random.randint(0, W.shape[0]-1)
        # Ri, _ = AssociationWeights(W, vectors)
        nums = np.array([Vsep[i].nums for i in range(len(Vsep))])
        min_value = np.min(nums)
        arg_index = np.where(nums == min_value)[0]
        if np.size(arg_index) > 1:
            arg_index = arg_index[random.randint(0, np.size(arg_index)-1)]
            # arg_index = random.randint(0, W.shape[0]-1)
        else:
            arg_index = arg_index[0]
        Ri, Rc = AssociationWeights(W, vectors)
        index = np.where(Ri == arg_index)[0]
        index = index[random.randint(0, np.size(index)-1)]
        one = W[arg_index, :]
        Vsep[int(Ri[arg_index])].nums += 1
        return one, Vsep, np.ones((1, np.shape(Objs)[1])), ts
    else:
        # translate to the first quadrant
        Objs = Objs - np.tile(np.min(Objs, axis=0), (np.shape(Objs)[0], 1))
        # select the non-dominated solutions
        ts['t1'] += 1
        FrontNo, _ = ENS_SS_NDSort(Objs, np.shape(Objs)[0])
        NDIndex = np.where(FrontNo == 1)[0]
        Dindex = np.where(FrontNo != 1)[0]
        NDSet = Objs[NDIndex, :]
        # fit the Pareto front
        if np.size(NDSet) != 0:
            approximatePF, approximateSTD = estimationUsingUnitHyperplane(NDSet, model="kriging")
        else:
            approximatePF, approximateSTD = estimationUsingUnitHyperplane(np.unique(Objs, axis=0), model="kriging")
        approximatePF = np.concatenate((approximatePF, NDSet), axis=0)
        PF = np.unique(approximatePF, axis=0)
        # translate to the first quadrant
        PF = (PF - np.tile(np.min(PF, axis=0), (np.shape(PF)[0], 1))) /\
            np.tile(np.max(PF, axis=0)-np.min(PF, axis=0), (np.shape(PF)[0], 1))
        # Remove origin/pole
        ind0 = np.where(np.sum(PF, axis=1) == 0)[0]
        ind1 = np.where(PF == 0)[0]
        index0 = np.unique(np.concatenate((ind0, ind1)))
        PF = np.delete(PF, index0, axis=0)

        # cooperative knee identification
        vectors = np.array([Vsep[i].vector.tolist() for i in range(len(Vsep))])
        ind1 = CHIM(PF, r)
        ind2 = local_BiDominance(PF, vectors, alpha, gamma, eva, maxeva, np.shape(PF)[0])
        ind3 = KneePointIdentification_TradeUtility(PF, vectors)
        ind4 = identificationCurvature(PF, vectors)
        ind0 = [ind1, ind2, ind3, ind4]
        ind = ind0[0]
        for j in range(1, len(ind0)):
            ind = np.unique(np.concatenate([ind, ind0[j]]))
        ind = ind.astype(int)

        knees = PF[ind, :]
        E = getExtremePoints(PF)
        kd_cluster = acuteCluster(knees, vectors, E, min(0.5, acmin))
        knees = np.array(kd_cluster)

        Addpropers = Properties(Pops.solutions[NDIndex], 'add')
        Addpropers_d = Properties(Pops.solutions[Dindex], 'add')
        zero_idx_nd = np.where(Addpropers == 0)[0]
        zero_idx_d = np.where(Addpropers_d == 0)[0]
        if len(zero_idx_nd) + len(zero_idx_d) == 1:
            Addpropers += 1     # non-domination attribution +1
            Pops.updateProperties(NDIndex, Addpropers, 'add')
            Addpropers_d = np.ones((len(Dindex), 1)) * -1   # domination attribution -1
            Pops.updateProperties(Dindex, Addpropers_d, 'add')
            old_ndsolutions = np.setdiff1d(np.arange(0, len(NDIndex)), zero_idx_nd)
            if np.all(Addpropers[old_ndsolutions, :] > 1) and (eva / maxeva >= cM and np.mod(eva, t2) == 0):
                # The non-dominated solution remains unchanged -- determine the promising knee points
                Nk, M = np.shape(knees)
                denses = cal_DistributedDensity(Objs[NDIndex, :])
                if denses is None:
                    ts['r2'] += 1
                    index = random.randint(0, Nk-1)
                    knee = knees[index, :].reshape(1, -1)
                    knee_direction = knee / (1e-8 + np.sum(knee))
                    Ri, _ = AssociationWeights(knee, vectors)
                    Ri = Ri.astype(int)
                    Vsep[Ri[0]].nums += 1
                else:
                    ts['t2'] += 1
                    knees_direction = knees / np.tile((1e-8 + np.sum(knees, axis=1))[:, np.newaxis], (M,))
                    ns = np.shape(denses)[0]
                    closest, closest_dis = random.randint(0, Nk-1), np.infty
                    for i in range(ns):
                        dis = calculateDistMatrix(denses[i, :].reshape(1, -1), knees_direction)
                        if np.min(dis) < closest_dis:
                            closest = np.argmin(dis)
                            closest_dis = np.min(dis)
                    knee_direction = knees_direction[closest, :]
                    knee = knees[closest, :].reshape(1, -1)
                    Ri, _ = AssociationWeights(knee, vectors)
                    Ri = Ri.astype(int)
                    Vsep[Ri[0]].nums += 1
            else:
                # The non-dominated solution change -- stochastic knee direction
                ts['r2'] += 1
                index = random.randint(0, np.shape(knees)[0]-1)
                knee = knees[index, :].reshape(1, -1)
                knee_direction = knee / (1e-20 + np.sum(knee))

                Ri, _ = AssociationWeights(knee, vectors)
                Ri = Ri.astype(int)
                Vsep[Ri[0]].nums += 1
        else:
            err_print('there is some error happened')
    return knee_direction, Vsep, PF, ts


def _EvolALG(PCheby, PopDec, model_, IFEs, Pro, fit="EI"):
    selected = Selection(PCheby, PopDec)
    offsprings = Crossover(selected, Pro.lower, Pro.upper)
    offsprings = Mutation(offsprings, Pro.lower, Pro.upper)
    Pools = np.append(PopDec, offsprings, axis=0)
    N = Pools.shape[0]
    EI = np.zeros(N)  # acquisition function
    MU = np.zeros(N)
    STD = np.zeros(N)
    Gbest = np.min(PCheby, axis=0)
    E0 = np.infty
    while IFEs > 0:
        for i in range(N):
            if fit == 'EI':
                mu = model_.predict_values(Pools[i, :].reshape(1, -1))
                std = np.sqrt(model_.predict_variances(Pools[i, :].reshape(1, -1)))
                if std == 0:
                    std = 1e-8
                criteria = -((Gbest-mu)*norm.cdf((Gbest-mu)/std) + std*norm.pdf((Gbest-mu)/std))
            elif fit == 'LCB':
                mu = model_.predict_values(Pools[i, :].reshape(1, -1))
                std = np.sqrt(model_.predict_variances(Pools[i, :].reshape(1, -1)))
                if std == 0:
                    std = 1e-8
                k = 3*1e-3
                criteria = mu - k*std
            EI[i] = criteria
            MU[i] = mu
            STD[i] = std
        sort_index = np.argsort(EI, kind="quicksort")
        if EI[sort_index[0]] < E0:
            Best = Pools[sort_index[0], :].copy()
            E0 = EI[sort_index[0]].copy()
        Parents = Pools[sort_index[:math.ceil(N/2)], :]

        selected = Selection(EI[sort_index[:math.ceil(N/2)]], Parents)
        offsprings = Crossover(selected, Pro.lower, Pro.upper)
        offsprings = Mutation(offsprings, Pro.lower, Pro.upper)
        Pools = np.append(Parents, offsprings, axis=0)

        IFEs -= np.shape(Pools)[0]
    PopDec = Best
    return PopDec.reshape(1, -1)


def transferDirection2(lamda, M, a):
    if a < 1e-3:
        return 1/lamda / np.sum(1/lamda)
    tmpVectors = np.zeros((M, M))
    for i in range(M):
        tmpVectors[i, :] = lamda
        tmpVectors[i, i] += a
        tmpVectors[i, :] = tmpVectors[i, :] / (1 + a)
    Lamda = np.concatenate((tmpVectors, lamda.reshape(1, -1)), axis=0)
    Lamda = np.unique(Lamda, axis=0)
    for i in range(np.shape(Lamda)[0]):
        Lamda[i, :] = 1/Lamda[i, :] / np.sum(1/Lamda[i, :])
    return Lamda


def PChebyFitness2(uPopObj, Lamda, a):
    N, _ = np.shape(uPopObj)
    if a < 1e-3:
        PCheby = np.max(uPopObj*np.tile(Lamda, (N, 1)), axis=1).reshape(N,) +\
            0.05*np.sum(uPopObj*np.tile(Lamda, (N, 1)), axis=1).reshape(N,)
        return PCheby
    else:
        PCheby = np.zeros(N)
        Ls = np.shape(Lamda)
        for i in range(N):
            PCheby[i] = np.min(np.max(Lamda*np.tile(uPopObj[i, :], (Ls[0], 1)), axis=1).reshape(Ls[0],) +
                               0.05*np.sum(Lamda*np.tile(uPopObj[i, :], (Ls[0], 1)), axis=1).reshape(Ls[0],), axis=0)
        return PCheby


class splitWeights():
    def __init__(self, vector) -> None:
        self.vector = vector
        self.nums = 0


class alg():
    def __init__(self) -> None:
        self.save = 0
        self.maxFE = 0
        self.pro = None
        self.result = {}
        self.metric = 0
        self.save_data_for_each_generation = 0
        self.filepath = None
        self.tmpFE = 0
        self.alpha = 0  # the parameter of alpha-dominance
        self.H2 = 0     # the number of inner reference vectors
        self.r = 0      # the parameter of neigborhood of each solution in the objective space
        self.W = 0
        self.eta = 0
        self.times = {'t1': 0, 't2': 0, 'r1': 0, 'r2': 0}
        self.ePF = None

    def _ParameterSet(self, **varargin):
        if len(varargin) > 0:
            Values = []
            for key, value in varargin.items():
                Values.append(value)
                if key == "MaxFEs":
                    self.maxFE = value
                    print("Internal GA evaluation: ", self.maxFE)
                elif key == "filepath":
                    self.filepath = value
                elif key == "save":
                    self.save_data_for_each_generation = value
                elif key == "alpha":
                    self.alpha = value
                elif key == "H2":
                    self.H2 = value
                elif key == "W":
                    self.W = value
                elif key == "r":
                    self.r = value
                elif key == 'eta':
                    self.eta = value
                elif key == 'continues':
                    self.continues = value
            return Values

    def Solve(self, Pro):
        # try:
        self.pro = Pro
        self.metric = {'runtime': time.perf_counter()}
        self.pro.FE = 0
        self.main(Pro)
        return self.result, self.metric

    def InitializeVector(self, M):
        W, Nw = generatorPoints(self.H2, M)
        d = self.eta
        W = W*d+(1-d)/M
        W0, _ = generatorPoints(1, M)
        Nw += 1
        W = np.append(W, W0, axis=0)
        Vs = [splitWeights(W[i, :]) for i in range(Nw)]
        return Vs

    def main(self,  Pro):
        # Parameter setting
        IFEs = self.maxFE
        print('Real evaluation:  %d' % Pro.maxFE)
        # Generate random population
        N = 11*Pro.D - 1
        (PopDec, tmp) = generatorPoints(N, Pro.D, method='Latin')
        PopDec = np.tile((Pro.upper - Pro.lower), (N, 1))*PopDec + np.tile(Pro.lower, (N, 1))
        Pops = SOLUTIONSET(Pro)
        Pops.building(Decs=PopDec, AddProper=np.zeros((N, 1)))
        # Generate weight matrix
        W, _ = generatorPoints(100, Pro.M)

        # Partition vectors to split the objective regions into several sub-regions
        Vsep = self.InitializeVector(Pro.M)
        gamma, maxeva = 0.75, self.pro.maxFE
        while self.NotTerminated(Pops):
            PopObj = Pops.objs()
            eva = self.pro.FE
            # According to the sample set, the Pareto front is fitted to identify the direction of knee point
            lamda, Vsep, uPF, self.times = scenarioJudgment(Pops, Vsep, self.alpha, self.r, gamma, eva, maxeva, 11*Pro.D-1, W, self.times)
            self.ePF = uPF
            # error-tolerant
            a = 0.1*np.cos((self.pro.FE/self.pro.maxFE)*np.pi/2)
            print("lamda: ", lamda,  "  |  a: ", a)
            Lamda = transferDirection2(lamda, Pro.M, a)
            # Screening training samples
            # Normalization
            (N, D) = np.shape(Pops.decs())
            uPopObj = (PopObj - np.tile(np.min(PopObj, axis=0), (N, 1)))\
                / np.tile(np.max(PopObj, axis=0)-np.min(PopObj, axis=0), (N, 1))
            PCheby = PChebyFitness2(uPopObj, Lamda, a)
            N_max = (11*D-1+25)
            half = float(4/5)
            if N > N_max:
                index = np.argsort(PCheby, kind='quicksort')  # 返回数组从小到大排序的序号
                # Next = index[:11*D-1+25]
                sequence = [i for i in range(N)]
                Next = np.concatenate((index[:math.ceil(N_max*half)], np.array(random.sample(sequence, N_max - math.ceil(N_max*half)))))
            else:
                Next = np.arange(0, N, step=1)
            PDec = Properties(Pops.solutions[Next], 'dec')
            PCheby = PCheby[Next]
            # Eliminate solutions with duplicate inputs or outputs
            ia1 = np.unique(np.round(PDec*1e6)/1e6, axis=0, return_index=True)[1]
            ia2 = np.unique(np.round(PCheby*1e6)/1e6, return_index=True)[1]
            ia = np.intersect1d(ia1, ia2)
            PDec = PDec[ia, :].reshape(np.size(ia), -1)
            PCheby = PCheby[ia].reshape(np.size(ia), -1)

            # Surrogate Models
            if np.isnan(PCheby).any() or np.isnan(PDec).any():
                nan_idx = np.where(np.isnan(PCheby))[0]
                PCheby = np.delete(PCheby, nan_idx, axis=0)
                PDec = np.delete(PDec, nan_idx, axis=0)
            if np.size(PCheby) == 0:
                continue
            model_ = KRG(theta0=[1e-2], print_training=False, print_prediction=False, print_problem=False,
                         print_solver=False, poly='constant', corr='squar_exp')
            model_.set_training_values(PDec, PCheby)
            model_.train()
            # internal search via surrogate models
            bestDec = _EvolALG(PCheby, PDec, model_, IFEs, Pro, 'EI')
            Pops.AddOnes(Dec=bestDec, AddProper=np.zeros((1, 1)))

    def NotTerminated(self, Pops):
        index = len(self.result) + 1
        # print("result.key:", index)
        self.result.update({index: [Pops.decs(), Pops.objs(), Pops.cons(), Pops.adds()]})
        # print("ALGORITHM.NotTerminated中: ", self.__pro.FE, self.__pro._maxFE)
        nofinish = self.pro.FE < self.pro.maxFE
        if np.mod(100*self.pro.FE/self.pro.maxFE, 1) <= 1:
            print("process: ", self.pro.FE/self.pro.maxFE)
        if self.save_data_for_each_generation > 0 and nofinish and self.filepath is not None:
            if self.tmpFE == 0 or (self.pro.FE - self.tmpFE) >= self.save_data_for_each_generation or \
                   (self.pro.FE - self.tmpFE) >= self.save_data_for_each_generation*self.pro.N:
                self.save = self.pro.FE
                self.metric['runtime'] = time.perf_counter() - self.metric['runtime']
                self.metric['times'] = self.times
                self.Save(self.filepath + "\\" + str(self.save))
                self.tmpFE = self.pro.FE
        if bool(1-nofinish) and self.filepath is not None:
            self.save = self.pro.FE
            self.metric['runtime'] = time.perf_counter() - self.metric['runtime']
            self.metric['times'] = self.times

            if self.save_data_for_each_generation > 0:
                self.Save(self.filepath + "\\" + str(self.save))
            else:
                self.Save(self.filepath)
        return nofinish
