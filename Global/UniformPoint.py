from scipy.special import comb
import numpy as np
from math import gcd
from itertools import combinations


def generatorPoints(N, M, method='NBI'):
    if method == 'NBI':
        (W, N) = NBI(N, M)
    elif method == 'Latin':
        (W, N) = Latin(N, M)
    elif method == 'MUD':
        (W, N) = MixtureUniformDesign(N, M)
    elif method == 'ILD':
        (W, N) = ILD(N, M)
    return (W, N)


def NBI(N, M):
    H1 = 1
    while comb(H1+M, M-1) <= N:  # comb(H1+M, M-1) Combination number (binomial coefficient) "select (M-1) numbers from (H1+M)"
        H1 = H1 + 1
    s = range(1, H1+M)
    W = np.asarray(list(combinations(s, M-1))) - np.tile(np.arange(0, M-1), (int(comb(H1+M-1, M-1)), 1)) - 1

    # W = np.array(W) - np.tile(np.arange(0, M-1), (int(comb(H1+M-1, M-1)), 1)) - 1
    W = (np.append(W, np.zeros((W.shape[0], 1))+H1, axis=1) - np.append(np.zeros((W.shape[0], 1)), W, axis=1))/H1
    if H1 < M:
        H2 = 0
        while comb(H1+M-1, M-1) + comb(H2+M, M-1) <= N:
            H2 += 1
        if H2 > 0:
            W2 = []
            s2 = range(1, H2+M)
            W2 = np.asarray(list(combinations(s2, M-1))) - np.tile(np.arange(0, M-1), (int(comb(H2+M-1, M-1)), 1)) - 1
            # W2 = np.array(W2) - np.tile(np.arange(0, M-1), (int(comb(H2+M-1, M-1)), 1)) - 1
            W2 = (np.append(W2, np.zeros((W2.shape[0], 1))+H2, axis=1) - np.append(np.zeros((W2.shape[0], 1)), W2, axis=1))/H2
            W = np.append(W, W2/2+1/(2*M), axis=0)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i, j] < 1e-6:
                W[i, j] = 1e-6
    N = W.shape[0]
    return (W, N)


def Latin(N, M):
    W = np.random.random((N, M))
    W = np.argsort(W, axis=0, kind='mergesort') + 1
    W = (np.random.random((N, M)) + W - 1)/N
    return (W, N)


def ILD(N, M):
    In = M * np.eye(M)
    W = np.zeros((1, M))
    edgeW = W
    while np.shape(W)[0] < N:
        edgeW = np.tile(edgeW, (M, 1)) + np.repeat(In, np.shape(edgeW)[0], axis=0)
        edgeW = np.unique(edgeW, axis=0)
        ind = np.where(np.min(edgeW, axis=0) == 0)[0]
        edgeW = np.take(edgeW, ind, axis=0)
        W = np.append(W+1, edgeW, axis=0)
    W = W / np.tile(np.sum(W, axis=1)[:, np.newaxis], (np.shape(W)[1],))
    W = np.where(W > 1e6, 1e6, W)
    N = np.shape(W)[0]
    return W, N


def MixtureUniformDesign(N, M):
    X = GoodLatticePoint(N, M-1)**(1/np.tile(np.arange(M-1, 0, -1), (N, 1)))
    X = np.clip(X, -np.infty, 1e6)
    X = np.where(X == 0, 1e-12, X)
    W = np.zeros((N, M))
    W[:, :-1] = (1-X)*np.cumprod(X, axis=1)/X
    W[:, -1] = np.prod(X, axis=1)
    return W, N


def GoodLatticePoint(N, M):
    range_nums = np.arange(1, N+1, 1)
    ind = np.asarray([], dtype=np.int64)
    for i in range(np.size(range_nums)):
        if gcd(range_nums[i], N) == 1:
            ind = np.append(ind, i)
    W1 = range_nums[ind]
    W = np.mod(np.dot(np.arange(1, N+1, 1).reshape(-1, 1), W1.reshape(1, -1)), N)
    W = np.where(W == 0, N, W)
    nCombination = int(comb(np.size(W1), M))
    if nCombination < 1e4:
        Combination = np.asarray(list(combinations(np.arange(1, np.size(W1)+1, 1), M)))
        CD2 = np.zeros((nCombination, 1))
        for i in range(nCombination):
            tmp = Combination[i, :].tolist()
            UT = np.empty((np.shape(W)[0], len(tmp)))
            for j in range(len(tmp)):
                UT[:, j] = W[:, tmp[j]-1]
            CD2[i] = CalCD2(UT)
        minIndex = np.argmin(CD2)
        tmp = Combination[minIndex, :].tolist()
        Data = np.empty((np.shape(W)[0], len(tmp)))
        for j in range(len(tmp)):
            Data[:, j] = W[:, tmp[j]-1]
    else:
        CD2 = np.zeros((N, 1))
        for i in range(N):
            UT = np.mod(np.dot(np.arange(1, N+1, 1).reshape(-1, 1), (i+1)**np.arange(0, M, 1).reshape(1, -1)), N)
            CD2[i] = CalCD2(UT)
        minIndex = np.argmin(CD2)
        Data = np.mod(np.dot(np.arange(1, N+1, 1).reshape(-1, 1), (minIndex+1)**np.arange(0, M, 1).reshape(1, -1)), N)
        Data = np.where(Data == 0, N, Data)
    Data = (Data-1)/(N-1)
    return Data


def CalCD2(UT):
    N, S = np.shape(UT)
    X = (2*UT-1)/(2*N)
    CS1 = np.sum(np.prod(2+np.abs(X-1/2)-(X-1/2)**2, axis=1))
    CS2 = np.zeros((N, 1))
    for i in range(N):
        CS2[i] = np.sum(np.prod((1+1/2*np.abs(np.tile(X[i, :], (N, 1))-1/2)
                        + 1/2*np.abs(X-1/2)
                        - 1/2*np.abs(np.tile(X[i, :], (N, 1))-X)), axis=1))
    CS2 = np.sum(CS2)
    CD2 = (13/12)**S - 2**(1-S)/N*CS1 + 1/(N**2)*CS2
    return CD2


if __name__ == '__main__':
    ILD(4, 3)
