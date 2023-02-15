import numpy as np
from Global.UniformPoint import generatorPoints
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from random import sample
import math


class RBFn(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_neurons=10, gamma=1, w_weights=None, knl='rbf'):
        self.kernel = knl
        self.hidden_neurons = hidden_neurons
        self.gamma = gamma
        self.w_weights = w_weights  # Parameters between output layer and hidden layer

    def fit(self, x_train, y_train):
        x_train = np.c_[-1 * np.ones(x_train.shape[0]), x_train]
        kmeans = KMeans(n_clusters=self.hidden_neurons).fit(x_train)  # Find the center through clustering
        self.centers = kmeans.cluster_centers_
        # Value of radial basis function
        if self.kernel == 'rbf':
            H = rbf_kernel(x_train, self.centers, gamma=self.gamma)
        elif self.kernel == 'cubic':
            H = polynomial_kernel(x_train, self.centers, gamma=1, degree=3, coef0=0)
        H = np.c_[-1 * np.ones(H.shape[0]), H]

        try:
            # numpy.linalg.lstsq(a, b, rcond='warn') Solving equations:a x = b
            self.w_weights = np.linalg.lstsq(H, np.asmatrix(y_train).T, rcond=-1)[0]
        except Exception as e:
            print(e.args)
            self.w_weights = np.linalg.pinv(H) @ y_train.reshape(-1, 1)  # @ Represent matrix-vector multiplication
        return self

    def predict(self, x_test):
        x_test = np.c_[-1 * np.ones(x_test.shape[0]), x_test]
        if self.kernel == 'rbf':
            H = rbf_kernel(x_test, self.centers, gamma=self.gamma)
        elif self.kernel == 'cubic':
            H = polynomial_kernel(x_test, self.centers, gamma=1, degree=3, coef0=0)
        H = np.c_[-1 * np.ones(H.shape[0]), H]
        return np.asmatrix(H) @ np.asmatrix(self.w_weights)

    def score(self, X, y, sample_weight=None):
        # from scipy.stats import pearsonr
        # r, p_value = pearsonr(y.reshape(-1, 1), self.predict(X))
        # return r ** 2
        #  Pearson: The square of the correlation coefficient p is the judgment coefficient R^2
        return r2_score(y.reshape(-1, 1), self.predict(X))


def estimationUsingUnitHyperplane(samples, model="kriging", dge=2):
    samples = np.clip(samples, 1e-6, 1e6)
    N, M = np.shape(samples)
    # L1 unit vector
    unit_samples = samples / np.tile(np.sum(samples, axis=1), (M, 1)).T
    # L1 norm
    normL1 = np.sum(samples, axis=1)
    # duplicate removal
    ia1 = np.unique(np.round(unit_samples*1e6)/1e6, axis=0, return_index=True)[1]
    # ia1 = np.unique(unit_samples, axis=0, return_index=True)[1]
    # Estimation
    (Hyperplane, Nw) = generatorPoints(65*M, M)
    approximatePF = np.zeros((Nw, M))
    approximateSTD = np.zeros(Nw)

    # Surrogate model prediction: various models to be used
    if model == "kriging":
        from smt.surrogate_models import KRG

        print("kriging model")
        gpr = KRG(theta0=[1e-2], print_training=False, print_prediction=False, print_problem=False,
                  print_solver=False, poly='constant', corr='squar_exp')
        gpr.set_training_values(unit_samples[ia1, :M-1], normL1[ia1].reshape(np.size(ia1), 1))
        gpr.train()
        for i in range(Nw):
            mu = gpr.predict_values(Hyperplane[i, :M-1].reshape(1, -1))
            std = np.sqrt(gpr.predict_variances(Hyperplane[i, :M-1].reshape(1, -1)))
            approximatePF[i, :] = mu*Hyperplane[i, :]
            approximateSTD[i] = std
    elif model == "poly":
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline  # Use the list to build a pair in Pipeline, which can be used to link multiple estimators into one

        print("Polynomial model")
        poly = Pipeline([('poly', PolynomialFeatures(degree=dge)), ('linear', LinearRegression(fit_intercept=False))])
        poly.fit(unit_samples[ia1, :M-1], normL1[ia1].reshape(np.size(ia1), 1))
        for i in range(Nw):
            mu = poly.predict(Hyperplane[i, :M-1].reshape(1, -1))
            approximatePF[i, :] = mu*Hyperplane[i, :]
    elif model == "mlp":
        from sklearn.neural_network import MLPRegressor

        print("Multi-layer Perceptrons model")
        regre = MLPRegressor(solver="sgd", activation="logistic", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, verbose=True, max_iter=500)
        regre.fit(unit_samples[ia1, :M-1], normL1[ia1].reshape(np.size(ia1), 1))
        for i in range(Nw):
            mu = regre.predict(Hyperplane[i, :M-1].reshape(1, -1))
            approximatePF[i, :] = mu*Hyperplane[i, :]
    elif model == "rbfns":
        print("Radial Basis Function Networks model")
        rbfn = RBFn(hidden_neurons=min(10, len(ia1)), gamma=0.12)
        rbfn.fit(unit_samples[ia1, :M-1], normL1[ia1].reshape(np.size(ia1), 1))
        for i in range(Nw):
            mu = rbfn.predict(Hyperplane[i, :M-1].reshape(1, -1))
            approximatePF[i, :] = mu*Hyperplane[i, :]
    else:
        print("Mo such model")
    return approximatePF, approximateSTD


def PF_sampling(PF, sampleNums, uniformSpacing=False):
    # Sample
    lists = [i for i in range(np.shape(PF)[0])]
    if bool(1-uniformSpacing):
        sp = sample(lists, k=sampleNums)
    else:
        sp = np.linspace(0, len(lists), num=sampleNums, endpoint=False)
        for i in range(np.size(sp)):
            sp[i] = np.where(np.mod(sp[i], 1) == 0, sp[i], math.floor(sp[i]))
        sp = sp.astype(int)
    return sp


if __name__ == "__main__":
    from Benchmark.DEBDK import DEB2DK
    pro = DEB2DK()
    pro.Setting(5, 2, 2)
    PF = pro.GetPF()
    PF = np.unique(PF, axis=0)
    print("PF:", PF)

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    # plt.figure(figsize=(8, 6))
    fig = plt.figure()
    gs = GridSpec(40, 40)
    ax1 = fig.add_subplot(gs[2:15, 2:19])
    # plt.subplot(221)
    ax1.scatter(PF[:, 0], PF[:, 1])

    sp = PF_sampling(PF, 5, True)
    sp = np.unique(np.concatenate((sp, [0, np.shape(PF)[0]-1])))
    print("Sampling mark sp:", sp)

    ax1.scatter(PF[sp, 0], PF[sp, 1], marker="*", c='r')
    approximatePF, approximateSTD = estimationUsingUnitHyperplane(PF[sp, :], model='rbfns')

    ax2 = fig.add_subplot(gs[2:15, 22:38])
    ax2.scatter(approximatePF[:, 0], approximatePF[:, 1], marker="p", c='g')
    ax2.scatter(PF[sp, 0], PF[sp, 1], marker="*", c='r')

    y = np.sum(PF, axis=1)
    mu = np.sum(approximatePF, axis=1)
    test_y = mu.ravel()  # ravel(): pulls the array dimension into a one-dimensional array
    # uncertainty = approximateSTD
    ax3 = fig.add_subplot(gs[18:38, 5:35])
    # plt.title("l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))
    # ax3.fill_between(approximatePF[:, 0], test_y + uncertainty, test_y - uncertainty, alpha=0.1)
    ax3.scatter(PF[:, 0], y, label="true", linewidths=0.25)
    ax3.scatter(approximatePF[:, 0], test_y, label="predict", c="orange", linewidths=1)
    ax3.scatter(PF[sp, 0], y[sp], label="train", c="red", marker="x")
    ax3.legend()
    plt.show()
