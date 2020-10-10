from partA import learningA
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import plot, show, legend, title, ylabel, xlabel, savefig

class learningB(learningA):
    def __init__(self, n, p, noisefactor=None):
        super().__init__(n, p, noisefactor)
        self.imageFilePath  = "../outputs/images/partB/"

    def sample(self, sourceX, sourceY, Nsamples=None):
        if Nsamples is None: Nsamples = 0.8
        if isinstance(Nsamples, float):
            Nsamples = int(Nsamples*sourceX.shape[0])

        sampleArrayX = np.zeros(sourceX[:Nsamples,:].shape)
        sampleArrayY = np.zeros(Nsamples)
        for i in range(Nsamples):
            ind = np.random.randint(Nsamples)
            sampleArrayX[i] = sourceX[ind]
            sampleArrayY[i] = sourceY[ind]

        return sampleArrayX, sampleArrayY

    def plotOLS_trainvtest(self, reps, p_range):
        # Made to be similar to Fig. 2.11 of Hastie, Tibshirani, and Friedman
        p_old = self.p
        p_rangeObject = range(p_range[0], p_range[1])
        SElist_train, SElist_test = np.zeros((2,p_range[1] - p_range[0]))

        for i in p_rangeObject:
            self.p = i + p_old
            self.craftX()

            for _ in range(reps):
                X_train, X_test, y_train, y_test = train_test_split(self.X, self.yData)
                ypredict_test, beta = self.OLS_core([X_train, X_test, y_train, y_test])
                ypredict_train = X_train @ beta

                SElist_test[i-p_range[0]]  += self.MSE(y_test,  ypredict_test)/reps
                SElist_train[i-p_range[0]] += self.MSE(y_train, ypredict_train)/reps

        self.p = p_old

        imageFileName = "trainTestMSE"
        plot(p_rangeObject, (SElist_test), label="Test")
        plot(p_rangeObject, (SElist_train), label="Train")
        legend()
        title("Logarithmic train v test plot for MSE")
        xlabel("Polynomial degree")
        ylabel("Mean Square Error")
        savefig(self.imageFilePath + imageFileName + ".png")
        show()

    def bootstrap(self, sampleSize=None, sampleN=None):
        if sampleSize is None: sampleSize = 0.8
        if sampleN is None: sampleN = 5

        self.craftX()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.yData)

        if isinstance(sampleSize, float): sampleSize = int(sampleSize*X_test.shape[0])
        
        r2list, SElist = np.zeros((2, sampleN)) 
        for i in range(sampleN):
            X_sample, y_sample = self.sample(X_train, y_train, sampleSize)
            r2list[i], SElist[i] = self.OLS([X_sample, X_test, y_sample, y_test])[:2]

        return r2list, SElist

    def bias(self, y_data, y_model):
        return np.mean((y_data - np.mean(y_model, axis=1))**2)
    
    def variance(self, y_model):
        return np.mean(np.var(y_model, axis=1))

    def biasVarianceAnalysis_bootstrap(self, p_range, sampleSize=None, sampleN=None):
        if sampleSize is None: sampleSize=0.8
        if isinstance(sampleSize, float): sampleSize = int(sampleSize*len(self.x))
        if sampleN is None: sampleN=5

        p_old = self.p
        p_rangeObject = range(p_range[0], p_range[1])

        bias = np.zeros(len(p_rangeObject))
        variance = bias.copy()
        SElist = bias.copy()

        # Imported to compare to my own model, see line 102
        from sklearn.linear_model import LinearRegression

        for i in p_rangeObject:
            self.p = i + p_old
            self.craftX()

            X_train, X_test, y_train, y_test = train_test_split(self.X, self.yData)

            ypredict_models = np.zeros((y_test.shape[0], sampleN))

            for j in range(sampleN):
                X_sample, y_sample = self.sample(X_train, y_train)

                
                ypredict = self.OLS_core([X_sample, X_test, y_sample, y_test])[0]
                # My OLS method was compared to ScikitLearns method, and was found not to be the issue
                # reg = LinearRegression().fit(X_sample, y_sample)
                # ypredict = reg.predict(X_test)
                # print("HERE WE GO",np.mean(test_var/ypredict))
                ypredict_models[:,j] = ypredict

            variance[i-p_range[0]] = self.variance(ypredict_models)
            # bias[i-p_range[0]] = np.mean((y_test - np.mean(ypredict_models, axis=1))**2)
            bias[i-p_range[0]] = self.bias(y_test, ypredict_models)

            SElist[i-p_range[0]] = np.mean( np.mean((y_test.reshape(-1,1) - ypredict_models)**2, axis=1))

        print("\nMSE divided by (bias+variance):", SElist/(bias+variance))

        self.p = p_old

        imageFileName = "biasVariance"
        filetype = ".png"

        plot(p_rangeObject, (bias), label="bias")
        plot(p_rangeObject, (variance), label="variance")
        plot(p_rangeObject, (SElist), label="MSE")

        legend()
        title("Logarithmic bias-variance plot against complexity\nOLS with bootstrap")
        xlabel("Polynomial degree")
        ylabel("log[bias/variance]")
        savefig(self.imageFilePath + imageFileName + filetype)
        show()        






if __name__ == "__main__":
    print("""Task as interpreted:
    (?) "general aim: study bias-variance trade-off by implementing bootstrap resampling"
    (x) Implement OLS with resampling techniques (bootstrap)
    (x) Generate a figure similar to Fig. 2.11, Hastie (showing test&training MSE's)
    (x) perform a bias-variance analysis by: Comparing MSE to complexity (MSE v polydegree), make a graph
    (x) Do some equations (mentioned in the introduction part of the report)
    (x) describe bias and variance as part of those equations; 
    (x) Discuss said bias and variance trade-off as a function of complexity, # of data points, and possibly also bootstrap
    """)



    Npoints = 15
    polydegree = 2
    sampleN = 100
    polydegree_range = [0,10]
    trainvtest_reps  = 18*10
    noisefactor = 0

    Bootstraplearner = learningB(Npoints, polydegree, noisefactor)

    for name, result in zip(["R2", "MSE"], Bootstraplearner.bootstrap(sampleN = sampleN)):
        print("\n" + name + 
            ": ( avg =", sum(result)/len(result), 
            "), ( max = ",  max(result), 
            "), ( min = ", min(result),  ")\n",
            result)

    sampleN *= 1

    Bootstraplearner.plotOLS_trainvtest(trainvtest_reps, polydegree_range)
    Bootstraplearner.biasVarianceAnalysis_bootstrap(polydegree_range, sampleN=sampleN)
