import numpy as np
from sklearn.cluster import KMeans
import random
from Global.PROBLEM import PROBLEM
from Global.UniformPoint import generatorPoints
from Global.selfGA import Selection, Mutation, Crossover


Nc = 2


class DO2DK(PROBLEM):
    def __init__(self) -> None:
        super().__init__()
        self.K = 0
        self.s = 0

    def Setting(self, D, M, K, s):
        self.M = M
        self.D = D
        self.K = K  # control the number of knees
        self.s = s
        self.lower = np.zeros(self.D)
        self.upper = np.ones(self.D)
        self.encoding = "real"

    def CalObj(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], self.M))
        for p in range(PopDec.shape[0]):
            sums = 0
            for i in range(1, self.D):
                sums += PopDec[p, i]
            g = 1+9*sums/(self.D-1)
            r = 5+10*(PopDec[p, 0]-0.5)*(PopDec[p, 0]-0.5)+2**(self.s/2)*np.cos(2*self.K*np.pi*PopDec[p, 0])/self.K
            PopObj[p, 0] = g*r*(np.sin(np.pi*PopDec[p, 0]/2**(self.s+1) + (1+(2**self.s-1)/2**(self.s+2))*np.pi)+1)
            PopObj[p, 1] = g*r*(np.cos(np.pi*PopDec[p, 0]/2+np.pi) + 1)
        self.FE += PopDec.shape[0]
        return PopObj

    def GetPF(self, N=100):
        # print(N)
        PopDec, N = generatorPoints(N, self.D, method='Latin')
        # print(PopDec, N)
        PopDec = np.tile((self.upper - self.lower), (N, 1))*PopDec + np.tile(self.lower, (N, 1))
        R = self.GetOptimum(PopDec)
        return R

    def GetOptimum(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], self.M))
        for p in range(PopDec.shape[0]):
            g = 1
            r = 5+10*(PopDec[p, 0]-0.5)*(PopDec[p, 0]-0.5)+2**(self.s/2)*np.cos(2*self.K*np.pi*PopDec[p, 0])/self.K
            PopObj[p, 0] = g*r*(np.sin(np.pi*PopDec[p, 0]/2**(self.s+1) + (1+(2**self.s-1)/2**(self.s+2))*np.pi)+1)
            PopObj[p, 1] = g*r*(np.cos(np.pi*PopDec[p, 0]/2+np.pi) + 1)
        return PopObj

    def GetKnees_Of_KFunc(self, low=0, up=1):
        N = 100
        lower = np.ones(1)*low
        upper = np.ones(1)*up
        # print(N)
        PopDec, N = generatorPoints(N, 1, method='Latin')
        # print(PopDec, N)
        PopDec = np.tile((up - low), (N, 1))*PopDec + np.tile(low, (N, 1))
        maxG = 100
        PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
        for p in range(PopDec.shape[0]):
            PopObj[p] = 5+10*(PopDec[p, 0]-0.5)*(PopDec[p, 0]-0.5)+2**(self.s/2)*np.cos(2*self.K*np.pi*PopDec[p, 0])/self.K

        import matplotlib.pyplot as plt
        plt.scatter(PopDec, PopObj)
        plt.title('The Knee function')
        plt.show()

        for g in range(maxG):
            offsprings = Crossover(PopDec, lower, upper, ProC=0.9)
            offsprings = Mutation(offsprings, lower, upper, ProM=0.1)
            OffObj = np.zeros(offsprings.shape[0])[:, np.newaxis]
            for i in range(offsprings.shape[0]):
                OffObj[i] = 5+10*(offsprings[i]-0.5)*(offsprings[i]-0.5)+2**(self.s/2)*np.cos(2*self.K*np.pi*offsprings[i])/self.K
            PopDec = Selection(OffObj, offsprings)
            PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
            for i in range(PopDec.shape[0]):
                PopObj[i] = 5+10*(PopDec[i]-0.5)*(PopDec[i]-0.5)+2**(self.s/2)*np.cos(2*self.K*np.pi*PopDec[i])/self.K
        return PopDec, PopObj

    def GetKnees_in_PF(self, kn_decs, nc=Nc):
        nums_cluster = nc
        kmeans = KMeans(n_clusters=nums_cluster, random_state=2).fit(kn_decs)
        kn_dec = np.zeros((nums_cluster, 1))
        for i in range(nums_cluster):
            index = np.where(kmeans.labels_ == i)[0]
            kn_dec[i] = np.mean(kn_decs[index])
        print(kn_dec)
        N, D = np.shape(kn_dec)
        k = self.D - 1  # the number of X2
        NI = N**(self.M-1)
        Decs = np.zeros((NI, self.M-1))
        if self.M == 2:
            Decs = kn_dec
        else:
            from itertools import permutations
            group = np.asarray(list(permutations(range(N), self.M-1)))
            for i in range(NI):
                if i < N:
                    Decs[i, :] = kn_dec[i]*np.ones(self.M-1)
                else:
                    Decs[i, :] = kn_dec[group[i, :]]
        randPoints, NI = generatorPoints(NI, k, method='Latin')
        Temp = np.tile(self.upper[self.M-1:] - self.lower[self.M-1:], (NI, 1))*randPoints +\
            np.tile(self.lower[self.M-1:], (NI, 1))
        Decs = np.concatenate((Decs, Temp), axis=1)
        Objs = self.GetOptimum(Decs)
        return Objs

    def GetKneesAround_in_PF(self, kn_decs, nc=Nc):
        nums_cluster = nc
        kmeans = KMeans(n_clusters=nums_cluster, random_state=2).fit(kn_decs)
        kn_dec = np.zeros((nums_cluster, 1))
        for i in range(nums_cluster):
            index = np.where(kmeans.labels_ == i)[0]
            kn_dec[i] = np.mean(kn_decs[index])

        N, D = np.shape(kn_dec)
        print(kn_dec)
        k = self.D-1  # the number of X2
        NI = N**(self.M-1)
        Decs = np.zeros((NI, 1))
        if self.M == 2:
            Decs = kn_dec
        randPoints, NI = generatorPoints(NI, k, method='Latin')
        Temp = np.tile(self.upper[self.M-1:] - self.lower[self.M-1:], (NI, 1))*randPoints +\
            np.tile(self.lower[self.M-1:], (NI, 1))
        Decs = np.concatenate((Decs, Temp), axis=1)

        # 增加扰动
        N = Decs.shape[0]
        for k in range(N):
            each = 20
            temp = np.ones((each, self.D))
            for i in range(each):
                temp[i, :] = Decs[k, :] + random.gauss(0, 0.01)*np.random.random((1, self.D))
            Decs = np.concatenate((Decs, temp))
        Objs = self.GetOptimum(Decs)
        return Objs
