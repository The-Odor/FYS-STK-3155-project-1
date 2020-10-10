from partD import learningD
from sklearn.model_selection import train_test_split
import sklearn.linear_model as linmod
import numpy as np
from matplotlib.pyplot import plot, show, legend, title, ylabel, xlabel, figure, savefig

class learningE(learningD):
    def __init__(self, n, p, noisefactor=None):
        super().__init__(n, p, noisefactor)
        self.imageFilePath  = "../outputs/images/partE/"


    def lasso(self, alpha=None):
        if alpha is None: alpha = 5
        clf = linmod.Lasso(alpha=alpha)
        self.craftX()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.yData)
        clf.fit(X_train, y_train)
        ypredict = clf.predict(X_test)
        return self.R2(y_test, ypredict), self.MSE(y_test, ypredict)

    def biasVarianceAnalysis_lasso_bootstrap(self, p_range, sampleSize=None, sampleN=None):
        p_old = self.p
        p_rangeObject = range(p_range[0], p_range[1])
        if sampleSize is None: sampleSize=0.8
        if isinstance(sampleSize, float): sampleSize = int(sampleSize*len(self.x))
        if sampleN is None: sampleN=5

        bias = np.zeros(len(p_rangeObject))
        variance = bias.copy()
        SElist = bias.copy()
        SEkfold = bias.copy()
        clf = linmod.Lasso(alpha=0.1)

        for i in p_rangeObject:
            self.p = i + p_old
            self.craftX()

            X_train, X_test, y_train, y_test = train_test_split(self.X, self.yData)

            ypredict_models = np.zeros((y_test.shape[0], sampleN))

            for j in range(sampleN):
                X_sample, y_sample = self.sample(X_train, y_train)

                clf.fit(X_sample, y_sample)
                ypredict = clf.predict(X_test)

                ypredict_models[:,j] = ypredict

            variance[i-p_range[0]] = self.variance(ypredict_models)
            bias[i-p_range[0]] = self.bias(y_test, ypredict_models)
            SElist[i-p_range[0]] = np.mean( np.mean((y_test.reshape(-1,1) - ypredict_models)**2, axis=1))
            SEkfold[i-p_range[0]] = np.mean(self.kfold(sampleN)[1])

        self.p = p_old
        filetype = ".png"
        imageFileName = "lassoBiasVariance"

        print("MSE/(bias+variance) =", SElist/(bias+variance))

        plot(p_rangeObject, np.log(bias), label="bias")
        plot(p_rangeObject, np.log(variance), label="variance")
        plot(p_rangeObject, np.log(SElist), label="MSE")
        plot(p_rangeObject, np.log(SEkfold), label="MSE: K-fold")

        legend()
        title("Logarithmic bias-variance plot against complexity")
        xlabel("Polynomial complexity")
        ylabel("log[bias/variance]")
        savefig(self.imageFilePath + imageFileName + filetype)
        show()










if __name__ == "__main__":
    print("""Task as interpreted:
    (x) Implement lasso from scikit learn
    ( ) Discuss the three methods (assume Lasso, Ridge, and Bootstrap (or OLS?)), and which fits the best
    ( ) Bias-variance trade-off analysis using bootstrap and kfold
    """)

    Npoints = 100
    polydegree = 5
    polydegree_range = [0,10]
    sampleN = 18
    alphas = [0.01,0.1,0.3,1,3]

    Lassolearner = learningE(Npoints, polydegree)
    for alph in alphas:
        print("Lasso evaluation [R2, MSE] at alpha = {:4}:".format(alph), Lassolearner.lasso())
    Lassolearner.biasVarianceAnalysis_lasso_bootstrap(polydegree_range, sampleN=sampleN)



