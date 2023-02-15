# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/01/22 10:04:40
@Author  :   jftang
'''

import os
import sys
from pathlib import Path
from multiprocessing import Pool
import time
import numpy as np


'''Get the current directory and set the data storage location'''
curPath = os.path.abspath(os.path.dirname(__file__))
BenchmarksPath = curPath + "\\Benchmark\\Benchmark"
print(BenchmarksPath)


def realReferenceData(test, D, M, K, needKnee=True):
    global BenchmarksPath
    sys.path.append(curPath)
    print(test)
    if test.lower() == 'ckp':
        from Benchmark.CKP import CKP

        prob = CKP()
        prob.Setting(D, M, K)

    elif test.lower() == 'deb2dk':
        from Benchmark.DEBDK import DEB2DK

        prob = DEB2DK()
        prob.Setting(D, M, K)

    elif test.lower() == 'deb3dk':
        from Benchmark.DEBDK import DEB3DK

        prob = DEB3DK()
        prob.Setting(D, M, K)

    elif test.lower() == 'do2dk':
        from Benchmark.DO2DK import DO2DK

        prob = DO2DK()
        prob.Setting(D, M, K, 1)
    PF = prob.GetPF(300)
    if needKnee:
        if test.lower() == 'do2dk':
            knees = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"S1"+"\\PF_knees.npy")
            regions = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"S1"+"\\regions.npy")
        else:
            knees = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"\\PF_knees.npy")
            regions = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"\\regions.npy")
        return prob, PF, regions, knees
    else:
        return prob, PF, knees, regions


def parameterSet(test):
    D = int(test[test.find('\\')+1:test.find('D')])
    M = int(test[test.find('D')+1:test.find('M')])
    K = int(test[test.find('M\\')+2:test.find('K')])
    if test[:test.find('\\')] == 'deb2dk':
        from Benchmark.DEBDK import DEB2DK
        pro = DEB2DK()
        if K == 1:
            pro.Setting(D, M, 1)
        elif K == 2:
            pro.Setting(D, M, 2)
        elif K == 3:
            pro.Setting(D, M, 3)
        elif K == 4:
            pro.Setting(D, M, 4)
        elif K == 9:
            pro.Setting(D, M, 9)
        if D == 10:
            maxFEs = pro._ParameterSet(MaxFEs=150)
        elif D == 30:
            maxFEs = pro._ParameterSet(MaxFEs=400)
    elif test[:test.find('\\')] == 'do2dk':
        from Benchmark.DO2DK import DO2DK
        pro = DO2DK()
        if K == 1:
            pro.Setting(D, M, 1, 1)
        elif K == 2:
            pro.Setting(D, M, 2, 1)
        elif K == 3:
            pro.Setting(D, M, 3, 1)
        elif K == 4:
            pro.Setting(D, M, 4, 1)
        if D == 10:
            maxFEs = pro._ParameterSet(MaxFEs=200)
        elif D == 30:
            maxFEs = pro._ParameterSet(MaxFEs=400)
    elif test[:test.find('\\')] == 'ckp':
        from Benchmark.CKP import CKP
        pro = CKP()
        if K == 1:
            pro.Setting(D, M, 1)
        elif K == 2:
            pro.Setting(D, M, 2)
        elif K == 3:
            pro.Setting(D, M, 3)
        elif K == 4:
            pro.Setting(D, M, 4)
        if D == 10:
            maxFEs = pro._ParameterSet(MaxFEs=200)
        elif D == 30:
            maxFEs = pro._ParameterSet(MaxFEs=400)
    elif test[:test.find('\\')] == 'deb3dk':
        from Benchmark.DEBDK import DEB3DK
        pro = DEB3DK()
        if K == 1:
            pro.Setting(D, M, 1)
        elif K == 2:
            pro.Setting(D, M, 2)
        elif K == 3:
            pro.Setting(D, M, 3)
        elif K == 4:
            pro.Setting(D, M, 4)
        if D == 10:
            maxFEs = pro._ParameterSet(MaxFEs=350)
        elif D == 30:
            maxFEs = pro._ParameterSet(MaxFEs=500)
    if M == 2:
        h = 5
        ro = 0.2
    elif M == 3:
        h = 3
        ro = 0.25
    return pro, maxFEs, h, ro


def process(test_path):
    test = test_path[0]
    path = test_path[1]

    my_file = Path(path)
    if my_file.exists():
        pass
    else:
        os.makedirs(path)
    dirs = len(os.listdir(path))
    path = path + "\\" + str(dirs+1) + "_times_" + str(os.getpid())
    my_file2 = Path(path)
    if my_file2.exists():
        pass
    else:
        os.makedirs(path)

    t_start = time.time()
    print("Start execution, the process number is%d" % os.getpid())

    from SAEA_KPF import alg
    pro, maxFEs, h, ro = parameterSet(test)
    print("maxFEs", maxFEs)

    solver = alg()
    _ = solver._ParameterSet(MaxFEs=10000, alpha=0.75, H2=h, r=ro, eta=0.7, filepath=path, save=-1)
    _, _ = solver.Solve(pro)
    t_stop = time.time()
    print("Implementation completed, time consumed %0.2f" % (t_stop-t_start))


if __name__ == "__main__":
    benchmarks = [
        'deb2dk\\10D2M\\1K', 'deb2dk\\10D2M\\2K', 'deb2dk\\10D2M\\3K', 'deb2dk\\10D2M\\4K',
        'do2dk\\10D2M\\1K', 'do2dk\\10D2M\\2K', 'do2dk\\10D2M\\3K', 'do2dk\\10D2M\\4K',
        'ckp\\10D2M\\1K', 'ckp\\10D2M\\2K', 'ckp\\10D2M\\3K', 'ckp\\10D2M\\4K',
        'deb3dk\\10D3M\\1K', 'deb3dk\\10D3M\\2K(knee4)', 'deb3dk\\10D3M\\3K(knee9)'
        ]

    benchmarks2 = [
        'deb2dk\\30D2M\\1K', 'deb2dk\\30D2M\\2K', 'deb2dk\\30D2M\\3K', 'deb2dk\\30D2M\\4K',
        'ckp\\30D2M\\1K', 'ckp\\30D2M\\2K', 'ckp\\30D2M\\3K', 'ckp\\30D2M\\4K',
        'do2dk\\30D2M\\1K', 'do2dk\\30D2M\\2K', 'do2dk\\30D2M\\3K', 'do2dk\\30D2M\\4K',
        'deb3dk\\30D3M\\1K', 'deb3dk\\30D3M\\2K(knee4)', 'deb3dk\\30D3M\\3K(knee9)'
    ]

    extraBenchmark = [
        'deb2dk\\10D2M\\9K'
    ]
    parallel = False
    if parallel:
        # Parallel execution
        for test in benchmarks:
            # the data storage location
            path = "D:\\File\\VsCode\\SurrogateAssisted\\Data\\SAEA_KPF\\" + test
            my_file = Path(path)
            if my_file.exists():
                pass
            else:
                os.makedirs(path)
            dirs = len(os.listdir(path))
            Nc = 20
            if dirs >= Nc:
                continue
            else:
                while dirs < Nc:
                    nums = min(Nc - dirs, 5)
                    print(test+" nums:", nums)
                    with Pool(nums) as p:
                        p.map(process, [[test, path] for i in range(nums)])
                    dirs = len(os.listdir(path))
    else:
        test = benchmarks[0]
        from SAEA_KPF import alg
        pro, maxFEs, h, ro = parameterSet(test)
        _, PF, regions, knees = realReferenceData(test[:test.find('\\')], pro.D, pro.M, pro.K)

        dsolver = alg()
        _ = dsolver._ParameterSet(MaxFEs=10000, alpha=0.75, H2=h, r=ro, eta=0.7, filepath=None, save=-1)
        t_start = time.time()
        dsolver.Solve(pro)
        results, metrics = dsolver.Solve(pro)
        t_stop = time.time()
        print("The algorithm is completed and takes time %0.2f" % (t_stop-t_start))

        from Global.NondominatedSorting import NDSort
        Objs = results.get(len(results))[1]
        t_start = time.time()
        FrontNo, MaxFNo = NDSort(Objs)
        t_stop = time.time()
        print("Implementation completed, time consumed%0.2f" % (t_stop-t_start))
        nonIndex1 = np.where(FrontNo == 1)[0]

        # plot
        import matplotlib.pyplot as plt
        if pro.M == 2:
            fig = plt.figure()
            plt.scatter(PF[:, 0], PF[:, 1], marker='.', c='blue')
            plt.scatter(knees[:, 0], knees[:, 1], marker='x', c='orange')
            plt.scatter(Objs[nonIndex1, 0], Objs[nonIndex1, 1], marker='p', c='red')
            plt.show()
        else:
            fig = plt.figure(figsize=(14, 10), dpi=50, facecolor='w', edgecolor='k')
            # ax = plt.axes(projection='3d')
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.scatter(PF[:, 0], PF[:, 1], PF[:, 2], marker='.', alpha=0.5, label='$PF$')
            ax.scatter(Objs[:, 0], Objs[:, 1], Objs[:, 2], marker='p', c='r')
            ax.scatter(knees[:, 0], knees[:, 1], knees[:, 2], marker='s', c='black')
            ax.legend(fontsize=24, loc=0)
            ax.tick_params(labelsize=24)
            ax.set_xlabel("$f_1$", fontsize=28)
            ax.set_ylabel("$f_2$", fontsize=28)
            ax.set_zlabel("$f_2$", fontsize=28)

            plt.show()