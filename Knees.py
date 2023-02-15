from functools import reduce
import numpy as np
from Global.NondominatedSorting import FastAlphaNDSort, ENS_SS_NDSort
from scipy.linalg import norm
import random
from sklearn.cluster import KMeans, AffinityPropagation
import os
import sys

'''获取当前目录'''
curPath = os.path.abspath(os.path.dirname(__file__))
print(curPath)

# '''获取上一级目录'''
prePath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(prePath)
sys.path.append(os.path.join(prePath, 'Benchmark'))

# from PMOP import PMOP1


# python: Create a structure using a class
class AssistStruct:
    def print_self(self):
        print(self)


class Individual(AssistStruct):
    def __init__(self, nums, liste):
        self.dominateMe = nums  # (int)
        self.iDominate = liste  # list

    def add_dominateMe(self):
        self.dominateMe = self.dominateMe + 1

    def del_dominateMe(self):
        self.dominateMe = self.dominateMe - 1

    def add_iDominate(self, serial):
        self.iDominate.append(serial)

    def del_iDominate(self, serial):
        self.iDominate.remove(serial)

    def check_if_zero(self):
        if self.dominateMe == 0:
            return True
        else:
            return False


class Front(AssistStruct):
    def __init__(self, liste):
        self.f = liste  # list

    def add_f(self, serial):
        self.f.append(serial)

    def del_f(self, serial):
        self.f.remove(serial)


# -----------------------local alpha-dominance relationship
def AssociationWeights(PopObjs, W):
    N, M = np.shape(PopObjs)
    Nid = np.zeros(M)
    W_size = np.shape(W)[0]

    Ri = np.zeros(N, dtype=int)
    Rc = np.zeros(W_size, dtype=int)

    for i in range(N):
        dis = np.zeros(W_size)
        for j in range(W_size):
            d, sums = 0, np.linalg.norm(W[j, :], ord=2)
            for k in range(M):
                d += np.abs((PopObjs[i, k]-Nid[k])*W[j, k]/sums)
            d2 = 0
            for k in range(M):
                d2 += (PopObjs[i, k] - (Nid[k]+d*W[j, k]))**2
            dis[j] = np.sqrt(d2)
        Index = np.where(dis == min(dis))
        index = Index[0][0]
        Ri[i] = index
        Rc[index] += 1
    return Ri, Rc


def local_alpha_dominance(PObjs, vectors, alpha):
    Ri, Rc = AssociationWeights(PObjs, vectors)
    Ri = Ri.astype(int)
    FrontNo, maxF = FastAlphaNDSort(PObjs, Ri, np.shape(PObjs)[0], alpha)
    ind = np.where(FrontNo == 1)[0]
    return ind


# -----------------------NBI
def CHIM(PF, r):
    # duplication removal
    PF = np.where(PF > 1e-12, PF, 1e-12)
    # Normalization
    PF = (PF - np.tile(np.min(PF, axis=0), (np.shape(PF)[0], 1))) / np.tile(np.max(PF, axis=0)-np.min(PF, axis=0), (np.shape(PF)[0], 1))
    Hyperplane = PF / np.tile(np.sum(PF, axis=1), (np.shape(PF)[1], 1)).T
    distance = np.sqrt(np.sum(abs(PF - Hyperplane)**2, axis=1))
    fmax = np.max(PF, axis=0)
    fmin = np.min(PF, axis=0)
    R = (fmax-fmin)*r
    ifKnees = np.zeros(np.shape(PF)[0])
    for j in range(np.shape(PF)[0]):
        Temp = np.delete(PF, j, axis=0)
        tempDis = np.delete(distance, j)
        dis = np.abs(PF[j, :] - Temp)
        index2 = np.cumprod(dis <= R, axis=1)[:, -1]
        ss = np.where(index2 == 1)[0]
        Neighbors = Temp[ss, :]
        Neighbors_metric = tempDis[ss]
        sN = np.shape(Neighbors)
        ne = 0
        for k in range(sN[0]):
            if Neighbors_metric[k] > distance[j]:
                ne += 1
        if ne == 0:
            ifKnees[j] = 1
    ind = np.where(ifKnees == 1)[0]
    return ind


# ------------------------Two localized dominance relationships
def getExtremePoints(Objs, transpose=False):
    N, M = np.shape(Objs)
    E = np.zeros((2, M))
    # tmp1 -- ideal point
    # tmp2 -- nadir point
    for m in range(M):
        tmp1 = np.inf
        tmp2 = -np.inf
        for i in range(N):
            if tmp1 > Objs[i, m]:
                tmp1 = Objs[i, m]
            elif tmp2 < Objs[i, m]:
                tmp2 = Objs[i, m]
        E[0, m] = tmp1
        E[1, m] = tmp2
    if transpose:
        extremes = np.zeros((2, M))
        for i in range(M):
            extremes[i, :] = E[0, :]
            extremes[i, i] = E[1, i]
        return extremes
    return E


def local_BiDominance(PObjs, vectors, alpha, gamma, eva, maxeva, popsize):
    E = getExtremePoints(PObjs)
    Ri, _ = AssociationWeights(PObjs, vectors)
    Ri = Ri.astype(int)
    FrontNo, _ = FastAlphaNDSort(PObjs, Ri, np.shape(PObjs)[0], alpha)
    LnFrontNo = np.zeros(np.size(FrontNo))

    fi_index = np.where(FrontNo == 1)[0]
    LnFrontNo[fi_index], _ = KDSorting(PObjs[fi_index, :], vectors, E, gamma, eva, maxeva, popsize)
    index1 = np.where(FrontNo == 1)[0]
    index2 = np.where(LnFrontNo == 1)[0]
    final_index = reduce(np.intersect1d, [index1, index2])
    return final_index   # Returns the subscript of the solution in the population PObjs


def KDSorting(PopObj, vectors, E, gamma, eva, maxeva, popsize):
    (N, M) = np.shape(PopObj)
    FronNo = np.ones(N)*np.infty
    MaxFNo = 1

    Ri, Rc = AssociationWeights(PopObj, vectors)
    Ri = Ri.astype(int)
    solutions = np.array([Individual(0, []) for i in range(N)])
    Flist = [Front([])]
    for i in range(N):
        for j in range(N):
            iDominatej = KneeDominanceComparator(PopObj[i, :], PopObj[j, :], Ri[i], Ri[j], gamma, E, Rc, eva, maxeva, popsize)
            if iDominatej == -1:
                solutions[i].add_iDominate(j)
            elif iDominatej == 1:
                solutions[i].add_dominateMe()
        if solutions[i].dominateMe == 0:
            FronNo[i] = 1
            Flist[0].add_f(i)
    front = 1
    while Flist[front-1].f:
        Q = []
        for i in Flist[front-1].f:
            if solutions[i].iDominate:
                for j in solutions[i].iDominate:
                    solutions[j].del_dominateMe()
                    if solutions[j].check_if_zero():
                        FronNo[j] = front+1
                        Q.append(j)
        front += 1
        Flist.extend([Front(Q)])
    MaxFNo = front-1
    # Flist[MaxFNo-1].f: non-empty，Flist[MaxFNo].f: empty，there are total MaxFNo numbers of front ranks
    # FrontNo: 1~MaxFNo
    return (FronNo, MaxFNo)


def KneeDominanceComparator(Ii, Ij, wi, wj, gamma, E, Rc, eva, maxeva, popsize):
    M = np.size(Ii)
    # extreme points transfer
    exe = np.zeros((2, M))
    for m in range(M):
        exe[1, m] = E[1, m] * (1 + np.power(np.exp(1), -4))
    # translation to calc the tangent theta
    s1_t = translate(Ii, exe)
    s2_t = translate(Ij, exe)
    # the upper and lower limit angle
    stan1, stan2 = np.zeros(2), np.zeros(2)
    stan1[0] = np.min(s1_t)
    stan1[1] = np.max(s1_t)
    stan2[0] = np.min(s2_t)
    stan2[1] = np.max(s2_t)
    # Comparing solutions among the lower limit angle
    AB, BA = np.zeros(M), np.zeros(M)
    for m in range(M):
        AB[m] = Ij[m] - Ii[m]
        BA[m] = -AB[m]
    # -1 : solution1 better; 1 solution2 better; 0 equal.
    wi = int(wi)
    wj = int(wj)
    if wi == wj:
        # turning gamma
        tmp1 = 0.1*eva/(0.1*maxeva)
        tmp2 = 0.1*Rc[wi]/(0.2*popsize)
        if tmp1 < tmp2 or tmp1 < 0.5:
            gamma = 0.5
        else:
            gamma = tmp1 - tmp2
        if calcAngle(AB, Ii, Ij, exe) < (stan1[1] + s1_t[0]*gamma):
            return -1
        elif calcAngle(BA, Ij, Ii, exe) < (stan2[1] + s2_t[0]*gamma):
            return 1
        else:
            return 0
    else:
        return 0


def translate(solution, exe):
    # translation to calc the tangent theta
    M = np.size(solution)
    s_t = np.zeros(M)
    for m in range(M):
        tmp = abs(exe[1, m]) - solution[m]
        sum = 0
        for k in range(M):
            if k != m:
                sum += (exe[0, k] - solution[k])**2
        s_t[m] = np.arctan(np.sqrt(sum)/tmp)
    return s_t


def calcAngle(AB, s1, s2, exe):
    # Calc the angel between two vectors
    size = np.size(AB)
    lena, lenb, lenc = 0, 0, 0
    for i in range(size):
        lena += AB[i]**2
        lenb += (s1[i] - exe[1, i])**2
        lenc += (s2[i] - exe[1, i])**2
    lena = np.sqrt(lena)  # vector AB
    lenb = np.sqrt(lenb)  # vector idea-A
    lenc = np.sqrt(lenc)  # vector idea-B
    return np.pi - np.arccos((lena**2 + lenb**2 - lenc**2)/(2*lena*lenb))


# --------Posterior Decision-Making Based on Decomposition-Driven Knee Point Identification
def TradeUtility(s1, s2, exe):
    M = np.size(s1)
    G12, D12 = 0, 0
    for m in range(M):
        G12 += min(0, (s1[m] - s2[m])/(exe[1, m] - exe[0, m]))
        D12 += max(0, (s1[m] - s2[m])/(exe[1, m] - exe[0, m]))
    U12 = G12 + D12
    if U12 < 0:
        return -1  # s1 is defined to knee_dominate s2
    elif U12 == 0:
        return 0  # s1 is defined to non_knee_dominate s2
    else:
        return 1  # s1 is defined to be knee_dominated by s2


def KneePointIdentification_TradeUtility(PObjs, vectors):
    N, M = np.shape(PObjs)
    E = getExtremePoints(PObjs)
    # Find the neighborhood of each sub-region
    Neighbors = NeighborSearch(PObjs, vectors)
    ifKnees = np.zeros(N)
    for i in range(N):
        flag = 1
        for j in range(np.size(Neighbors[i])):
            ind = Neighbors[i][j]
            if TradeUtility(PObjs[i, :], PObjs[ind, :], E) > 0:
                flag = 0
                break
        if flag == 1:
            ifKnees[i] = 1
    kneesIndex = np.where(ifKnees == 1)[0]
    kneesIndex_sorted = KneeSorted_accumulative_utility(PObjs, kneesIndex, Neighbors, E)
    return kneesIndex_sorted


def KneeSorted_accumulative_utility(PObjs, kneesIndex, Neighbors, E):
    Nk = np.size(kneesIndex)
    K = np.zeros(Nk)
    for i in range(Nk):
        numOfNeighbors = np.size(Neighbors[kneesIndex[i]])
        for j in range(numOfNeighbors):
            K[i] += TradeUtility(PObjs[kneesIndex[i], :], PObjs[Neighbors[kneesIndex[i]][j], :], E)
    sort_index = np.argsort(-K, kind='quicksort', )  # descending sort of K
    return kneesIndex[sort_index]


def AssociationWeights_acuteAngle(PObjs, W):
    N, M = np.shape(PObjs)
    # Normalize the objective space
    uPObjs = (PObjs - np.tile(np.min(PObjs, axis=0), (N, 1))) / np.tile(np.max(PObjs, axis=0)-np.min(PObjs, axis=0)+1e-8, (N, 1))

    Nid = np.zeros(M)
    W_size = np.shape(W)[0]
    Ri = np.zeros(N)
    Rc = np.zeros(W_size)
    for i in range(N):
        angles = np.zeros(W_size)
        for j in range(W_size):
            norm1 = norm(uPObjs[i, :] - Nid, 2)
            norm2 = norm(W[j, :] - Nid, 2)
            angles[j] = np.arccos(np.dot(uPObjs[i, :]-Nid, W[j, :]-Nid)/(norm1*norm2))
        index = np.where(angles == np.min(angles))[0]
        if len(index) > 0:
            Ri[i] = index[0]
            Rc[index[0]] += 1
        else:
            index = random.randint(0, W_size-1)
            Ri[i] = index
            Rc[index] += 1
    return Ri, Rc


def NeighborSearch(PObjs, W):
    W_size = np.shape(W)[0]
    Neighborhoods = []
    for i in range(W_size):
        sub_neighborhood = np.array([])
        min_angle = np.infty
        for j in range(W_size):
            if j != i:
                if AcuteAngle(W[i, :], W[j, :]) < min_angle:
                    min_angle = AcuteAngle(W[i, :], W[j, :])
                    sub_neighborhood = np.array([])
                    sub_neighborhood = np.append(sub_neighborhood, j)
                elif AcuteAngle(W[i, :], W[j, :]) == min_angle:
                    sub_neighborhood = np.append(sub_neighborhood, j)
        sub_neighborhood = np.append(sub_neighborhood, i)
        sub_neighborhood = sub_neighborhood.astype(int)
        Neighborhoods.append(sub_neighborhood)
    Ri, Rc = AssociationWeights_acuteAngle(PObjs, W)
    N = np.shape(PObjs)[0]
    Neighbors = []
    for i in range(N):
        for j in range(W_size):
            if Ri[i] == j:
                sizeOfNeigborhood = np.size(Neighborhoods[j])
                for k in range(sizeOfNeigborhood):
                    k_index = np.where(Ri == Neighborhoods[j][k])[0]
                    Neighbor_individual = np.setdiff1d(k_index, i)
                Neighbors.append(Neighbor_individual)
    return Neighbors


def AcuteAngle(v, u):
    return np.arccos(np.dot(v, u)/(norm(v, 2)*norm(u, 2)))


# ----------------------Cone-domination.
def creatKneeFrontierOfCone(PObjs, sigma=135/180):
    E = getExtremePoints(PObjs)
    FrontNo, maxF = ENS_SS_NDSort(PObjs, np.shape(PObjs)[0])
    LnFrontNo, maxF_Ln = Cone_domination(PObjs, E, sigma)
    index1 = np.where(FrontNo == 1)[0]  # non-dominated solutions
    index2 = np.where(LnFrontNo == 1)[0]  # non-dominated knee solutions
    final_index = reduce(np.intersect1d, [index1, index2])
    return final_index, LnFrontNo   # Returns the subscript of the solution in the population PObjs


def Cone_domination(PopObj, E, sigma):
    (N, M) = np.shape(PopObj)
    FronNo = np.ones(N)*np.infty
    MaxFNo = 1
    solutions = np.array([Individual(0, []) for i in range(N)])
    Flist = [Front([])]
    for i in range(N):
        for j in range(N):
            iDominatej = Cone_domination_operator(PopObj[i, :], PopObj[j, :], E, sigma)
            if iDominatej == -1:
                solutions[i].add_iDominate(j)
            elif iDominatej == 1:
                solutions[i].add_dominateMe()
        if solutions[i].dominateMe == 0:
            FronNo[i] = 1
            Flist[0].add_f(i)
    front = 1
    while Flist[front-1].f:
        Q = []
        for i in Flist[front-1].f:
            if solutions[i].iDominate:
                for j in solutions[i].iDominate:
                    solutions[j].del_dominateMe()
                    if solutions[j].check_if_zero():
                        FronNo[j] = front+1
                        Q.append(j)
        front += 1
        Flist.extend([Front(Q)])
    MaxFNo = front-1
    # Flist[MaxFNo-1].f: non-empty，Flist[MaxFNo].f: empty，there are total MaxFNo numbers of front ranks
    # FrontNo: 1~MaxFNo
    return (FronNo, MaxFNo)


def Cone_domination_operator(P, Q, E, sigma):
    M = np.size(P)
    angle = np.pi*sigma
    # Normalized
    normP = (P - E[0, :]) / (E[1, :] - E[0, :])
    normQ = (Q - E[0, :]) / (E[1, :] - E[0, :])
    dom_better, dom_worse, dom_equal = 0, 0, 0
    for i in range(M):
        cone1 = normP[i].copy()
        cone2 = normQ[i].copy()
        for j in range(M):
            cone1 += np.tan((angle - np.pi/2)/2)*normP[j]
            cone2 += np.tan((angle - np.pi/2)/2)*normQ[j]
        if cone1 < cone2:
            dom_better += 1
        elif cone1 > cone2:
            dom_worse += 1
        else:
            dom_equal += 1
    if dom_worse == 0 and dom_equal != M:
        return -1  # P dominate Q
    elif dom_better == 0 and dom_equal != M:
        return 1  # P is dominated by Q
    else:
        return 0


#  ---------------------------------identifying Knees according to the curvature
def identificationCurvature(PObjs, vectors, K=None, numsOfNeighbor=np.infty):
    N = np.shape(PObjs)[0]
    E = getExtremePoints(PObjs)
    # normalization
    uPObjs = (PObjs - np.tile(E[0, :], (N, 1))) / np.tile(E[1, :] - E[0, :], (N, 1))
    Ri, _ = AssociationWeights(uPObjs, vectors)
    Ri = Ri.astype(int)
    w_size = np.shape(vectors)[0]

    ifKnees = np.zeros(N)
    for j in range(w_size):
        wj_index = np.where(Ri == j)[0]
        if np.size(wj_index) == 0:
            continue
        curvatures = np.ones(np.size(wj_index))*np.infty
        curvatures = calcCurvature(uPObjs[wj_index, :])
        if K is None:
            soi = np.where(curvatures == np.min(curvatures))[0]  # Solution of convex region corresponding to minimum curvature
            ifKnees[wj_index[soi]] = 1
        elif K == w_size:
            soi = np.where(curvatures == np.min(curvatures))[0]  # Solution of convex region corresponding to minimum curvature
            ifKnees[wj_index[soi]] = 1
        else:
            numsOfSoi = 0
            for i in range(len(wj_index)):
                dis = calculateDistMatrix(uPObjs[wj_index[i], :], uPObjs)
                ind = np.sort(dis)
                nl = int(min(numsOfNeighbor, len(wj_index)+1, len(ind)))
                if nl == 1:
                    ifKnees[wj_index[i]] = 1
                    numsOfSoi += 1
                    continue
                neighbors = np.take(curvatures, ind[1:nl], 0)
                if np.min(neighbors) == curvatures[i]:  # If its curvature is the smallest in the field, it may be a promising solution
                    ifKnees[wj_index[i]] = 1
                    numsOfSoi += 1
                    if numsOfSoi == K:
                        break
    ind = np.where(ifKnees == 1)[0]
    return ind


def calcCurvature(uPObjs):
    uPObjs = np.where(uPObjs > 1e-12, uPObjs, 1e-12)
    N, M = np.shape(uPObjs)
    P = np.ones(N)  # Initial curvature
    lamda = 1 + np.zeros(N)
    E = np.sum(uPObjs**np.tile(P[:, np.newaxis], (M,)), axis=1) - 1
    for epoch in range(5000):
        # gradient descent
        G = np.sum(uPObjs**np.tile(P[:, np.newaxis], (M,))*np.log(uPObjs), axis=1)
        newP = P - lamda*E*G
        newE = np.sum(uPObjs**np.tile(newP[:, np.newaxis], (M,)), axis=1) - 1
        # Update the value of each weight
        update = (newP > 0) & (np.sum(newE**2) < np.sum(E**2))
        P[update] = newP[update]
        E[update] = newE[update]
        lamda[update] = lamda[update]*1.1
        lamda[~update] = lamda[~update]/1.1
    return P


# -------------------------Angle integration based on direction
def acuteCluster(kds, vectors, exe, acute=0.1):
    N, M = np.shape(kds)
    Nid = exe[0, :]
    Ri, Rc = AssociationWeights_acuteAngle(kds, vectors)
    w_size = np.shape(vectors)[0]
    kds_cluster = []
    for w in range(w_size):
        w_index = np.where(Ri == w)[0]
        Tmp = kds[w_index, :]
        while np.shape(Tmp)[0] > 0:
            One = Tmp[0, :]
            acutes = calculateAcuteMatrix(One, Tmp, Nid)
            index = np.where(acutes <= acute)[0]
            kds_cluster.append(np.mean(Tmp[index, :], axis=0))
            Tmp = np.delete(Tmp, index, axis=0)
    return kds_cluster


def calculateAcuteMatrix(A, matrix, Nid):
    N = np.shape(matrix)[0]
    norm1 = norm(A-Nid, 2)
    acutes = np.ones(N)*np.infty
    for i in range(N):
        norm2 = norm(matrix[i, :]-Nid, 2)
        cosh = np.where(np.dot(A-Nid, matrix[i, :]-Nid)/(norm1*norm2) <= 1, np.dot(A-Nid, matrix[i, :]-Nid)/(norm1*norm2), 1)
        acutes[i] = np.arccos(cosh)
    return acutes


# -------------------------knee point identification based on the density of demapping in the extreme point hyperplane
def identificationDensity(PObjs, f):
    # f: size of solution group in the knee region
    # First, normalize all solutions into the first quadrant
    N, M = np.shape(PObjs)
    E = getExtremePoints(PObjs)
    # normalize
    uPObjs = (PObjs - np.tile(E[0, :], (N, 1))) / np.tile(E[1, :] - E[0, :], (N, 1))
    # transformation: Map to the extreme point hyperplane
    maping_p = uPObjs / np.tile(np.sum(uPObjs, axis=1)[:, np.newaxis], (M,))
    # Calculate the density of mapping solutions on the extreme point hyperplane by measuring the proximity of a solution to its neighbors
    K0 = []
    K0_index = []
    index = np.arange(0, N, dtype=np.int16)
    for i in range(N):
        # remove the extreme points
        if np.sum(uPObjs[i, :]) == 1:
            K0_index.append(i)
            K0.append(PObjs[i, :])
    index = np.setdiff1d(index, K0_index)
    k = 3*f
    den = np.zeros(N)
    for i in index:
        tmp = np.setdiff1d(index, i)
        # step1, find the k nearest neighbors of each solution, and calculate the Euclidean distance between a solution and its neighbors
        dis_matrix = calculateDistMatrix(maping_p[i, :].reshape(1, -1), maping_p[tmp, :])
        k_nearst_dis = np.sort(dis_matrix)[:k]
        dis_harmonic = k / np.mean(1/k_nearst_dis)   # harmonic distance
        den[i] = 1 / dis_harmonic
    # Affinity propagation clustering
    # af = KMeans(n_clusters=f, init='k-means++', n_init=10, max_iter=400, tol=1e-4).fit(PObjs)  # 通过聚类找到中心
    af = AffinityPropagation(preference=-3, random_state=3).fit(PObjs)
    idx = af.labels_
    b = 2*f
    solution_set = [K0]
    while len(index) > 0:
        eta = np.argmin(den[index])
        cluster_index = index[np.where(idx[index] == idx[index[eta]])[0]]
        K = uPObjs[cluster_index, :]
        dis_matrix = calculateDistMatrix(uPObjs[index[eta], :].reshape(1, -1), K).ravel()
        b_nearst = np.argsort(dis_matrix)[:min(b, np.shape(K)[0])+1]
        situation = Distinguish(uPObjs[cluster_index[b_nearst], :])
        if situation >= 0:
            if cal_radialCoordinateValues(uPObjs[cluster_index[b_nearst], :], situation, 0.01):
                solution_set.append(PObjs[cluster_index[b_nearst], :])
            index = np.setdiff1d(index, cluster_index)
        else:
            index = np.setdiff1d(index, cluster_index)
    return solution_set


def Distinguish(P):
    exe = getExtremePoints(P)
    N, M = np.shape(P)
    # Normalization
    P_ = (P - np.tile(exe[0, :], (N, 1))) / np.tile(exe[1, :] - exe[0, :], (N, 1))
    index = []
    for i in range(N):
        # remove the extreme points
        if np.sum(P_[i, :]) == 1:
            index.append(i)
    P_ = np.delete(P_, index, axis=0)
    pp_norm = np.sum(P_, axis=1) - 1
    if np.all(pp_norm > 0):
        # above the hyperplane
        return 1
    elif np.all(pp_norm < 0):
        # below the hyperplane
        return 0
    else:
        # on the hyperplane
        return -1


def cal_radialCoordinateValues(P, situation, diff=0.01):
    N, M = np.shape(P)
    r = np.zeros(N)
    if situation == 0:  # below the hyperplane
        for i in range(N):
            r[i] = (M - np.sqrt(M**2 - 4*np.sum(P[i, :]*P[i, :]))) / 2
    else:   # above the hyperplane
        for i in range(N):
            r[i] = np.sum(P[i, :]*P[i, :])
    r_ = np.sum(r[1:]) / (N-1)
    var = np.abs(r[0] - r_)
    if var >= diff:
        return True
    else:
        return False


def calculateDistMatrix(datas, DATAS):
    dist = np.zeros((datas.shape[0], DATAS.shape[0]))  # the distance matrix
    if datas.shape[1] > 1:
        for i in range(datas.shape[0]):
            Temp = np.sum((DATAS - np.dot(np.ones((DATAS.shape[0], 1)), datas[i, :][np.newaxis, :]))**2, axis=1)
            dist[i, :] = np.sqrt(Temp)
    else:  # 1-D data
        for i in range(datas.shape[0]):
            dist[i, :] = np.abs(datas[i] - DATAS)
    return dist


# -----------Calculate the distribution density of the solution
def cal_DistributedDensity(PObjs):
    N, M = np.shape(PObjs)
    E = getExtremePoints(PObjs)
    # normalization
    uPObjs = (PObjs - np.tile(E[0, :], (N, 1))) / np.tile(E[1, :] - E[0, :], (N, 1))
    # transformation: Map to the extreme point hyperplane
    maping_p = uPObjs / np.tile(np.sum(uPObjs, axis=1)[:, np.newaxis], (M,))
    # Calculate the density of mapping solutions on the extreme point hyperplane by measuring the proximity of a solution to its neighbors
    b = max(int(N / 6), 1)
    af = KMeans(n_clusters=b, init='k-means++', n_init=10, max_iter=400, tol=1e-4).fit(maping_p)
    # af = AffinityPropagation(preference=-3, random_state=3).fit(PObjs)
    idx = af.labels_
    dense = np.zeros((b, M))
    iters, i = 1, 0
    for eta in range(af.n_clusters):
        if i > b and iters > 30:
            break

        cluster_index = np.where(idx == eta)[0]
        if len(cluster_index) == 1:
            iters += 1
            continue
        situation = Distinguish(uPObjs[cluster_index, :])
        if situation <= 0:
            dense[i, :] = af.cluster_centers_[eta, :]
            i += 1
        else:
            print('find density region does not below the hyperplane')
        iters += 1
    if iters == 31:
        index = np.where(np.sum(dense, axis=1) > 0)[0]
        if len(index) > 0:
            dense = np.unique(dense[index, :], axis=0)
            return dense.reshape(-1, M)
        else:
            return None
    else:
        index = np.where(np.sum(dense, axis=1) > 0)[0]
        dense = np.unique(dense[index, :], axis=0)
        return dense.reshape(-1, M)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Benchmark.DEBDK import DEB2DK
    pro = DEB2DK()
    pro.Setting(10, 2, 1)
    path = "D:\\File\\VsCode\\expensiveMaOP_knee\\finalVersion\\Benchmark\\Benchmark\\DEB2DK\\10D2M\\K1"
    # pro = PMOP1()
    # pro.Setting(12, 3, 4, 1, -1, 1)
    # path = "D:\\File\\VsCode\\expensiveMaOP_knee\\Benchmark\\PMOP1\\12D3M\\A4"
    # from Benchmark.CKP import CKP
    # pro = CKP()
    # pro.Setting(10, 2, 1)
    # path = "D:\\File\\VsCode\\expensiveMaOP_knee\\finalVersion\\Benchmark\\Benchmark\\CKP\\10D2M\\K1"
    PF_knees = np.load(path+"\\PF_knees.npy")
    PF = pro.GetPF(300)
    # reference vectors
    from Global.UniformPoint import generatorPoints
    W, Nw = generatorPoints(min(pro.M+2, 5), pro.M)
    d = 0.5
    W = W*d+(1-d)/2
    # database = identificationCurvature(PF, W)
    # database = KneePointIdentification_TradeUtility(PF, W)
    # database = CHIM(PF, 0.5)
    database = local_alpha_dominance(PF, W, alpha=0.75)
    print(database)
    if pro.M == 2:
        fig = plt.figure(figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.scatter(PF[:, 0], PF[:, 1], c='b', marker='.', label='$PF$')
        plt.scatter(PF[database, 0], PF[database, 1], c='r', label='identified knees')
        plt.scatter(PF_knees[:, 0], PF_knees[:, 1], c='g', marker='x', label='True knees')
        plt.legend(fontsize=24, loc=0)
        plt.show()
    elif pro.M == 3:
        fig = plt.figure(figsize=(14, 10), dpi=50, facecolor='w', edgecolor='k')
        # ax = plt.axes(projection='3d')
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot(PF[:, 0], PF[:, 1], PF[:, 2], marker='.', alpha=0.5, label='$PF$')
        ax.scatter(PF[database, 0], PF[database, 1], PF[database, 2], marker='p', c='r')
        ax.scatter(PF_knees[:, 0], PF_knees[:, 1], PF_knees[:, 2], marker='s', c='black')
        ax.legend(fontsize=24, loc=0)
        ax.tick_params(labelsize=24)
        ax.set_xlabel("$f_1$", fontsize=28)
        ax.set_ylabel("$f_2$", fontsize=28)
        ax.set_zlabel("$f_2$", fontsize=28)
        plt.show()
