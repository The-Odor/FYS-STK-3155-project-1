from partC import learningC
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib.pyplot import plot, show, legend, title, ylabel, xlabel, figure, savefig


class learningD(learningC):
    def __init__(self, n, p, noisefactor=None):
        super().__init__(n, p, noisefactor)
        self.imageFilePath  = "../outputs/images/partD/"

    def ridge_core(self, dataset):
        X_train, X_test, y_train, _, lamb = dataset
        beta = np.linalg.pinv(X_train.T @ X_train + lamb*np.identity(self.nfeatures)) @ X_train.T @ y_train
        ypredict = X_test @ beta
        return ypredict, beta

    def ridge(self, dataset=None):
        if dataset is None:
            self.craftX()
            dataset = train_test_split(self.X, self.yData)
            ypredict, beta = self.ridge_core((dataset))
        else:
            ypredict, beta = self.ridge_core(list(dataset))
        return self.R2(dataset[3], ypredict), self.MSE(dataset[3], ypredict), beta

    def bootstrapRidge(self, lambSpace, sampleSize=0.8, sampleN=5):
        lambN = len(lambSpace)
        self.craftX()
        if isinstance(sampleSize, float):
            sampleSize = int(sampleSize*len(self.x))
        r2list, SElist = np.zeros((2, lambN)) 

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.yData)
        
        for lamb in range(lambN):
            for _ in range(sampleN):
                X_sample, y_sample = self.sample(X_train, y_train)

                r2, SE = self.ridge([X_sample, X_test, y_sample, y_test, lambSpace[lamb]])[:2]
                r2list[lamb] += r2/sampleN
                SElist[lamb] += SE/sampleN

        return r2list, SElist

    def kfoldRidge(self, lambSpace, k, solver=None):
        self.craftX()
        lambN = len(lambSpace)
        R2avg, MSEavg = np.zeros((2, lambN))
        for lamb in range(lambN):
            dataset = self.kfold_yielder(k)
            for data in dataset:
                R2, MSE = self.ridge(list(data) + [lambSpace[lamb]])[:2]
                R2avg[lamb]  += R2/k
                MSEavg[lamb] += MSE/k
        return R2avg, MSEavg

    def biasVarianceAnalysis_ridge(self, p_range, sampleSize=None, sampleN=None):
        #TODO: CURRENTLY JUST A COPY OF THE ORIGINAL FUNCTION FROM b), STILL NEEDS TO BE ADAPTED FOR RIDGE
        p_old = self.p
        p_rangeObject = range(p_range[0], p_range[1])
        if sampleSize is None: sampleSize=0.8
        if isinstance(sampleSize, float): sampleSize = int(sampleSize*len(self.x))
        if sampleN is None: sampleN=5

        bias = np.zeros(len(p_rangeObject))
        variance = bias.copy()
        SElist = bias.copy()

        for i in p_rangeObject:
            self.p = i + p_old
            self.craftX()

            X_train, X_test, y_train, y_test = train_test_split(self.X, self.yData)

            ypredict_models = np.zeros((y_test.shape[0], sampleN))

            for j in range(sampleN):
                X_sample, y_sample = self.sample(X_train, y_train)

                ypredict = self.OLS_core([X_sample, X_test, y_sample, y_test])[0]

                ypredict_models[:,j] = ypredict

            test_var = np.var(ypredict_models, axis=1)

            print(test_var, len(test_var), max(test_var), np.mean(test_var), sep="\n", end="\n-----------------------------------\n")
            variance[i-p_range[0]] = self.variance(ypredict_models)
            bias[i-p_range[0]] = np.mean((y_test - np.mean(ypredict_models, axis=1))**2)

            SElist[i-p_range[0]] = np.mean( np.mean((y_test.reshape(-1,1) - ypredict_models)**2, axis=1))

        self.p = p_old

        plot(p_rangeObject, np.log(bias), label="bias")
        plot(p_rangeObject, np.log(variance), label="variance")
        plot(p_rangeObject, np.log(SElist), label="MSE")

        legend()
        title("Logarithmic bias-variance plot against complexity")
        xlabel("Polynomial complexity")
        ylabel("log[bias/variance]")
        show()

if __name__ == "__main__":
    print("""Task as interpreted
    (x) Implement ridge method with kfold and bootstrap
        (-) Change bootstrap and kfold to take method as argument
            Too much fixing required
        (x) Create bootstrapRidge & kfoldRidge
    (x) Compare ridge-Bootstrap, ridge-kfold, Bootstrap, and kfold
    ( ) Study bias-variance trade-off of various values of lambda using bootstrap. 
    ( ) Comment on the above
    """)

    Npoints = 100
    polydegree = 5
    lambN = 50
    logLow = -3
    logHigh = 0
    
    ridgelearner = learningD(Npoints, polydegree)
    lambSpace = np.logspace(logLow, logHigh, num=int(lambN/2))
    lambSpace = np.concatenate((np.flip(-lambSpace), [0], lambSpace))

    folds = sampleN = OLSruns = 5

    print(" "*35 + "*** NORMAL OLS: ***")
    for name, result in zip(["OLS, multiple times", "bootstrap", "kfold"], 
                            [ridgelearner.OLSeval(OLSruns),
                             ridgelearner.bootstrap(sampleN=sampleN), 
                             ridgelearner.kfold(folds)]):
        print("Method used: ", name)
        for name, result in zip(["R2", "MSE"], result):
            print("\n" + name + 
                ": ( avg =", sum(result)/len(result), 
                "), ( max = ",  max(result), 
                "), ( min = ", min(result),  ")\n",
                result)
        print("--------------------"*4+"\n")

    print(" "*35 + "*** RIDGE: ***")
    for name, result in zip(["BRidge", "KRidge"], 
                            [ridgelearner.bootstrapRidge(lambSpace, sampleN=sampleN), 
                             ridgelearner.kfoldRidge(lambSpace, folds)]):
        print("Method used: ", name)
        for name, result in zip(["R2", "MSE"], result):
            print("\n" + name + 
                ": ( avg =", sum(result)/len(result), 
                "), ( max = ",  max(result), 
                "), ( min = ", min(result),  ")\n",
                result)
        print("--------------------"*4+"\n")

    for name, result in zip(["Bootstrap-Ridge", "Kfold-Ridge"], 
                            [ridgelearner.bootstrapRidge(lambSpace, sampleN=sampleN), 
                             ridgelearner.kfoldRidge(lambSpace, folds)]):
        # print("Method used: ", name)
        # for i in range(len(lambSpace)):
        #     print("lambda =", lambSpace[i], "R2: ", result[0][i], "MSE: ", result[1][i])
        # print("--------------------"*4)
        figure(1)
        plot(lambSpace, result[0], label=name)
        xlabel("Lambda")
        title("R2 values as a function of lambda\nlambda of logarithmic space between +-10^" + str(logLow) +
              " and +-10^" + str(logHigh) + " \nnumber of lambda points = " + str(len(lambSpace)))
        legend()
        figure(2)
        plot(lambSpace, result[1], label=name)
        title("MSE values as a function of lambda\nlambda of logarithmic space between " + "0" +
              " and " + str(lambSpace[-1]) + " \nnumber of lambda points = " + str(len(lambSpace)))
        xlabel("Lambda")
        legend()


        # show(block=False)
    imageFileNameMSE = "MSERidge"
    imageFileNameR2 = "R2Ridge"
    filetype = ".png"
    figure(1)
    savefig(ridgelearner.imageFilePath + imageFileNameR2 + filetype)
    figure(2)
    savefig(ridgelearner.imageFilePath + imageFileNameMSE + filetype)
    show()

    # ridgelearner.biasVarianceAnalysis_ridge([0,10])