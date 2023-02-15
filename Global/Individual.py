# -*- encoding: utf-8 -*-
'''
@File    :   SOLUTION.py
@Time    :   2022/06/27 16:24:01
@Author  :   jftang
'''

import numpy as np
from copy import deepcopy


'''format one'''


def getPDec(TempSolutionArray):
    PDec = np.array(TempSolutionArray[0].dec)
    for i in range(1, len(TempSolutionArray)):
        PDec = np.append(PDec, TempSolutionArray[i].dec, axis=0)
    return PDec.reshape(len(TempSolutionArray), -1)


def getPObj(TempSolutionArray):
    PObj = np.array(TempSolutionArray[0].obj)
    for i in range(1, len(TempSolutionArray)):
        PObj = np.append(PObj, TempSolutionArray[i].obj, axis=0)
    return PObj.reshape(len(TempSolutionArray), -1)


def getPCon(TempSolutionArray):
    PCon = np.array(TempSolutionArray[0].con)
    for i in range(1, len(TempSolutionArray)):
        PCon = np.append(PCon, TempSolutionArray[i].con, axis=0)
    return PCon.reshape(len(TempSolutionArray), -1)


def getPAdd(TempSolutionArray):
    PAdd = np.array(TempSolutionArray[0].add)
    for i in range(1, len(TempSolutionArray)):
        PAdd = np.append(PAdd, TempSolutionArray[i].add)
    return PAdd.reshape(len(TempSolutionArray), -1)


def err_print(msg, original_line=None):
    print('ERROR  ' * 3)
    print(msg)
    if original_line:
        print(original_line)
    print('ERROR  ' * 3)
    exit(1)


def Properties(TempSolutionArray, key='dec'):
    try:
        if key == 'dec':
            return getPDec(TempSolutionArray)
        elif key == 'obj':
            return getPObj(TempSolutionArray)
        elif key == 'con':
            return getPCon(TempSolutionArray)
        elif key == 'add':
            return getPAdd(TempSolutionArray)
    except Exception as e:
        # print(e)
        err_print('can not find such property', e)
    finally:
        pass


class SOLUTIONSET:
    # SOLUTIONSET methods:
    #   building	<public>        the constructor, which sets all the
    #                               properties of the solution
    #   decs        <public>      	get the matrix of decision variables of
    #                               multiple solutions
    #   objs        <public>        get the matrix of objective values of
    #                               multiple solutions
    #   cons        <public>        get the matrix of constraint violations of
    #                               multiple solutions
    #   adds        <public>        get the matrix of additional properties of
    #                               multiple solutions
    #   best        <public>        get the feasible and nondominated solutions
    #                               among multiple solutions

    def __init__(self, instance) -> None:
        self.problem = instance
        self.solutions = 0

    def building(self, **varvargin):
        Decs = varvargin['Decs']
        self.problem.N = Decs.shape[0]
        solutions = np.array([SOLUTION() for i in range(self.problem.N)])
        if len(varvargin.keys()) > 1:
            AddProper = varvargin['AddProper']
        Objs = self.problem.CalObj(Decs)
        Cons = self.problem.CalCon(Decs)
        for i in range(self.problem.N):
            solutions[i].dec = np.array(Decs[i, :])
            solutions[i].obj = Objs[i, :]
            solutions[i].con = Cons[i, :]
            if len(varvargin.keys()) > 1:
                solutions[i].add = AddProper[i, :]
        self.solutions = solutions
        return solutions

    def group(self, *varvargin):
        Decs = varvargin[0]
        Objs = varvargin[1]
        Cons = varvargin[2]
        Adds = varvargin[3]
        self.problem.N = Decs.shape[0]
        solutions = np.array([SOLUTION() for i in range(self.problem.N)])
        for i in range(self.problem.N):
            solutions[i].dec = np.array(Decs[i, :])
            solutions[i].obj = Objs[i, :]
            solutions[i].con = Cons[i, :]
            solutions[i].add = Adds[i, :]
        self.solutions = solutions
        return solutions

    def decs(self):
        Ns = len(self.solutions)
        Decs = np.zeros((Ns, self.problem.D))
        for i in range(Ns):
            Decs[i, :] = deepcopy(self.solutions[i].dec)
        return Decs

    def objs(self):
        Ns = len(self.solutions)
        Objs = np.zeros((Ns, self.problem.M))
        for i in range(Ns):
            Objs[i, :] = deepcopy(self.solutions[i].obj)
        return Objs

    def cons(self):
        Ns = len(self.solutions)
        Cons = np.zeros((Ns, 1))
        for i in range(Ns):
            Cons[i] = deepcopy(self.solutions[i].con)
        return Cons

    def adds(self):
        Ns = len(self.solutions)
        a = np.size(self.solutions[0].add)
        AddProper = np.zeros((Ns, a))
        for i in range(Ns):
            AddProper[i, :] = deepcopy(self.solutions[i].add)
        return AddProper

    def updateProperties(self, whos, Properties, key='dec'):
        try:
            if key == 'dec':
                self.updateDecs(whos, Properties)
            elif key == 'obj':
                self.updateObjs(whos, Properties)
            elif key == 'con':
                self.updateCons(whos, Properties)
            elif key == 'add':
                self.updateAdds(whos, Properties)
        except Exception as e:
            # print(e)
            err_print('can not find such property', e)
        finally:
            pass

    def updateAdds(self, whos, AddPropers):
        for i in range(len(whos)):
            self.solutions[whos[i]].add = AddPropers[i, :]

    def updateDecs(self, whos, Decs):
        for i in range(len(whos)):
            self.solutions[whos[i]].dec = Decs[i, :]

    def updateObjs(self, whos, Objs):
        for i in range(len(whos)):
            self.solutions[whos[i]].obj = Objs[i, :]

    def updateCons(self, whos, Cons):
        for i in range(len(whos)):
            self.solutions[whos[i]].con = Cons[i, :]

    def best(self):
        # best - Get the best solutions in a population.
        Feasible = np.any(self.cons() <= 0, axis=1)  # np.any(axis=1) Determines whether any array element in each row is non-zero
        Objs = self.objs()
        Decs, Cons, AddPropers = self.decs(), self.cons(), self.adds()
        # if np.all(Feasible, where=False):
        #     Best = []
        if self.problem.M > 1:
            pass
        else:
            Best = np.argsort(Objs[Feasible, :], kind="quicksort")
            Best = Best[0]
        P = [Decs[Best, :], Objs[Best, :], Cons[Best, :], AddPropers[Best, :]]
        return P

    def AddOnes(self, **varvargin):
        if len(varvargin) == 1:
            Dec = varvargin['Dec']
        elif len(varvargin) == 2:
            Dec = varvargin['Dec']
            AddProper = varvargin['AddProper']
        Obj = self.problem.CalObj(Dec)
        Con = self.problem.CalCon(Dec)
        N = Dec.shape[0]
        new_solution = np.array([SOLUTION() for i in range(N)])
        for i in range(N):
            new_solution[i].dec = np.array(Dec[i, :])
            new_solution[i].obj = Obj[i, :]
            new_solution[i].con = Con[i, :]
            if len(varvargin) > 1:
                new_solution[i].add = AddProper[i, :]
        # print("new_solution.dec.shape", new_solution.dec.shape)
        self.solutions = np.append(self.solutions, new_solution)
        self.problem.N += N
        return self


class SOLUTION:
    # SOLUTION - The class of a solution.
    def __init__(self, *varvargin) -> None:
        self.dec = 0
        self.obj = 0
        self.con = 0
        self.add = 0
        if len(varvargin) > 0:
            self.dec = varvargin[0]
            if len(varvargin) > 1:
                self.obj = varvargin[1]
                if len(varvargin) > 2:
                    self.con = varvargin[2]
                    if len(varvargin) > 3:
                        self.add = varvargin[3]


class individual:
    def __init__(self, obj, dec, con, add) -> None:
        self._obj = obj
        self._dec = dec
        self._con = con
        self._add = add


class Individuals:
    def __init__(self, *varargin) -> None:
        if len(varargin) == 0:
            self.__individuals = []
        elif len(varargin) == 1:
            Decs = varargin[0]
            N = np.shape(Decs)[0]
            self.__individuals = []
            for i in range(N):
                decs = Decs[i, :].copy()
                objective, constraint = getObjectiveConstraint(decs)
                G = np.max(np.concatenate((np.zeros((1, len(constraint))), np.array(constraint).reshape(1, -1)), axis=0), axis=0)
                self.__individuals.append(individual(obj=objective, dec=decs, con=np.sum(G), add=[]))

    def __len__(self):
        return len(self.__individuals)

    def __getitem__(self, position):
        return self.__individuals[position]

    def set_dimension(self, dim):
        self.__dim = dim

    def Decs(self):
        decs = np.zeros((len(self.__individuals), self.__dim))
        for i in range(len(self.__individuals)):
            decs[i, :] = self.__individuals[i]._dec
        return decs

    def Objs(self):
        objs = np.zeros((len(self.__individuals), 1))
        for i in range(len(self.__individuals)):
            objs[i, :] = self.__individuals[i]._obj
        return objs

    def Cons(self):
        cons = np.zeros((len(self.__individuals), 1))
        for i in range(len(self.__individuals)):
            cons[i, :] = self.__individuals[i]._con
        return cons

    def Adds(self):
        if len(self.__individuals[0]._add) == 0:
            return []
        adds = np.zeros((len(self.__individuals), len(self.__individuals[0]._add)))
        for i in range(len(self.__individuals)):
            adds[i, :] = self.__individuals[i]._add
        return adds

    def AndOne(self, ind):
        return self.__individuals.append(ind)


def getObjectiveConstraint(decs):
    pass


if __name__ == '__main__':
    class PROBLEM:
        def __init__(self) -> None:
            self.N = 20
            self.FE = 0
            self.M = 1
            self.D = 2
            self.maxFE = 500

        def CalObj(self, PopDec):
            PopObj = np.zeros((np.size(PopDec, axis=0), self.M))
            return PopObj

        def CalCon(self, PopDec):
            PopCon = np.zeros((np.size(PopDec, axis=0), 1))
            return PopCon

        def CalAdd(self, Adds):
            pass

    Pro = PROBLEM()
    upper, lower = 10, 0
    PopDec = np.random.random((Pro.N, Pro.D))
    PopDec = np.tile((upper - lower), (Pro.N, 1))*PopDec + np.tile(lower, (Pro.N, 1))
    Pops = SOLUTIONSET(Pro)
    Pops.building(Decs=PopDec, AddProper=np.zeros((Pro.N, 1)))
    # Get properties of all individuals
    pObjs = Pops.objs()
    pDecs = Pops.decs()
    pAdds = Pops.adds()
    pCons = Pops.cons()
    # Get properties of some individuals
    index = np.arange(start=0, stop=Pro.N-1, step=1)
    Next = index[:12]
    PDec = Properties(Pops.solutions[Next], 'dec')
    # Modify the properties of the specified individual
    index = [0, 1, 2]
    Pops.updateProperties(index, pDecs[:len(index), :], 'dec')
    # Add a new individual, and the objective evaluation is conducted internally
    Pops.AddOnes(Dec=np.random.random((1, Pro.D)), AddProper=np.zeros((1, 1)))
    # Instantiate new objects
    newPops = SOLUTIONSET(Pro)
    newPops.group(pDecs.copy(), pObjs.copy(), pCons.copy(), pAdds.copy())
