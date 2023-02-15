import numpy as np


def NDSort(*varargin):
    PopObj = varargin[0]
    (N, M) = np.shape(PopObj)
    nSort = N
    nargin = len(varargin)
    if nargin == 2:
        nSort = varargin[1]
    elif nargin == 3:
        PopCon = varargin[1]
        nSort = varargin[2]
        # the index of nonfeasible solutions, np.any(axis=1) Determines whether any array element in each row is non-zero
        Infeasible = np.any(PopCon > 0, axis=1)
        PopObj[Infeasible, :] = np.tile(np.max(PopObj, axis=0), (sum(Infeasible), 1)) + np.tile(np.sum(np.maximum(0, PopCon[Infeasible, :]), axis=1), (1, M))
    (FronNo, MaxFNo) = ENS_SS_NDSort(PopObj, nSort)
    return (FronNo, MaxFNo)


def FastAlphaNDSort(*varargin):
    PopObj = varargin[0]
    (N, M) = np.shape(PopObj)
    nargin = len(varargin)
    if nargin == 3:
        Ri = varargin[1]
        nSort = varargin[2]
        alpha = 0.75
    elif nargin == 4:
        Ri = varargin[1]
        nSort = varargin[2]
        alpha = varargin[3]
    FronNo = np.ones(N)*np.inf
    MaxFNo = 1

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

    solutions = np.array([Individual(0, []) for i in range(nSort)])
    Flist = [Front([])]
    for i in range(nSort):
        for j in range(nSort):
            # iDominatej = pareto_dominance_operator(PopObj[i, :], PopObj[j, :])
            iDominatej = localized_alpha_operator(PopObj[i, :], PopObj[j, :], Ri[i], Ri[j], alpha)
            if iDominatej == 1:
                solutions[i].add_iDominate(j)
            elif iDominatej == -1:
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
    MaxFNo = front-1  # Flist[MaxFNo-1].f: non-empty，Flist[MaxFNo].f: empty，there are total MaxFNo numbers of front ranks
    return (FronNo, MaxFNo)


def ENS_SS_NDSort(PObj, nSort):
    # PopObj Sort in ascending order according to the first target value,
    # and nSort represents the number of individuals to be sorted without control
    (PopObj, ia, ic) = np.unique(PObj, axis=0, return_index=True, return_inverse=True)
    # (C, ia, ic) = unique(A, axis=0) Returns the same data as in A, but does not contain duplicates. C sorted
    # If A is a matrix or array, then C = A(ia) and A(:) = C(ic)。
    # If the 'rows' option is specified, then C=A(ia,:) and A=C(ic,:)。
    # If A is a table or schedule, then C = A(ia,:) 且 A = C(ic,:)
    (Table, bin_edges) = np.histogram(ic, bins=np.arange(max(ic)+2))
    # hist(x,nbins) Divide x orderly into the number of bin specified by the scalar nbins.
    # N = np.size(PopObj, axis=0)
    (N, M) = np.shape(PopObj)
    FrontNo = np.ones(N)*np.inf
    MaxFNo = 0
    while np.sum(Table[FrontNo < np.inf]) < min(nSort, len(ic)):
        MaxFNo += 1
        for i in range(N):
            if FrontNo[i] == np.inf:
                # FrontNo[i] is compared with previous individuals
                Dominated = False
                for j in range(i-1, -1, -1):  # j<i，Solution j precedes solution i
                    # Only compare with the current front individual
                    if FrontNo[j] == MaxFNo:
                        m = 1  # Only compare with the current front individual
                        while (m < M) and (PopObj[i, m] >= PopObj[j, m]):
                            m += 1
                        Dominated = m >= M
                        # if Dominated or M == 2:
                        if Dominated:
                            break  # Dominated==True : i is controlled by the solution j of the current front
                if bool(1-Dominated):  # Domiatetd==False : i is not controlled by the current front individual
                    FrontNo[i] = MaxFNo
    FrontNo = FrontNo[ic]
    return (FrontNo, MaxFNo)


def pareto_dominance_operator(si, sj):
    pass


def alpha_ENS_SS_NDSort(alpha, *varargin):
    PObj = varargin[0]
    (N, M) = np.shape(PObj)
    nargin = len(varargin)
    if nargin == 2:
        nSort = varargin[1]
    elif nargin == 3:
        Ri = varargin[1]
        nSort = varargin[2]
    # PopObj Sort in ascending order according to the first target value,
    # and nSort represents the number of individuals to be sorted without control
    (PopObj, ia, ic) = np.unique(PObj, axis=0, return_index=True, return_inverse=True)
    Ric = Ri[ia]
    # (C, ia, ic) = unique(A, axis=0) Returns the same data as in A, but does not contain duplicates. C sorted
    # If A is a matrix or array, then C = A(ia) and A(:) = C(ic)。
    # If the 'rows' option is specified, then C=A(ia,:) and A=C(ic,:)。
    # If A is a table or schedule, then C = A(ia,:) 且 A = C(ic,:)
    (Table, bin_edges) = np.histogram(ic, bins=np.arange(max(ic)+2))
    # hist(x,nbins) Divide x orderly into the number of bin specified by the scalar nbins.
    # N = np.size(PopObj, axis=0)
    (N, M) = np.shape(PopObj)
    FrontNo = np.ones(N)*np.inf
    MaxFNo = 0
    while np.sum(Table[FrontNo < np.inf]) < min(nSort, len(ic)):
        MaxFNo += 1
        for i in range(N):
            if FrontNo[i] == np.inf:
                # FrontNo[i] is compared with previous individuals
                Dominated = False
                for j in range(i-1, -1, -1):  # j<i，Solution j precedes solution i
                    # Only compare with the current front individual
                    if FrontNo[j] == MaxFNo:
                        jDominatei = localized_alpha_operator(PopObj[j, :], PopObj[i, :], Ric[j], Ric[i], 0.75)
                        if jDominatei == 1:
                            Dominated = True
                        if Dominated:
                            break  # Dominated==True : i is controlled by the solution j of the current front
                if bool(1-Dominated):  # Domiatetd==False : i is not controlled by the current front individual
                    FrontNo[i] = MaxFNo
    FrontNo = FrontNo[ic]
    return (FrontNo, MaxFNo)


def localized_alpha_operator(si, sj, iIndex, jIndex, alpha):
    if iIndex != jIndex:
        iDominatej = 0
    else:
        M = np.size(si)
        xy = np.zeros(M)
        for p in range(M):
            xy[p] = si[p] - sj[p]
            for q in range(M):
                if p != q:
                    xy[p] = xy[p] + alpha * (si[q] - sj[q])
        dominate1 = 0
        dominate2 = 0
        for m in range(M):
            if xy[m] < 0:
                dominate1 += 1
            elif xy[m] > 0:
                dominate2 += 1
            else:
                pass
        if (dominate2 == 0) and (dominate1 > 0):  # i dominates j
            iDominatej = 1
        elif (dominate1 == 0) and (dominate2 > 0):  # j dominates i
            iDominatej = -1
        else:
            iDominatej = 0
    return iDominatej
