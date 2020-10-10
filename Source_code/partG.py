from partF import learningF
from sklearn.model_selection import train_test_split
import numpy as np
import math as mt
from matplotlib.pyplot import plot, figure, title, imshow, xlabel, ylabel, show, savefig, legend


class learningG(learningF):
    def __init__(self, nFrac, p):
        self.loadTerrainData()
        nMax = self.terrainData.shape[0]
        n = int((nMax)*nFrac)
    
        self.p = p
        self.nfeatures = int(((self.p+1)*(self.p+2))/2)

        self.imageFilePath  = "../outputs/images/partG/"
        
        if nFrac < 0.5:
            # Includes both random and regular data points
            x1 = y1 = np.linspace(0, nMax-1, mt.ceil(n/2), dtype=int)
            x2 = np.random.randint(0, nMax, mt.floor(n/2))
            y2 = np.random.randint(0, nMax, mt.floor(n/2))

            x = np.concatenate((x1, x2))
            y = np.concatenate((y1, y2))

        else:
            x = y = np.linspace(0, nMax-1, n, dtype=int)

        x, y = np.meshgrid(x,y)
        self.x = x.flatten()
        self.y = y.flatten()
        self.yData = self.terrainData[self.x,self.y]

    def plotkfold_trainvtest(self, k, p_range):
        p_old = self.p
        p_rangeObject = range(p_range[0], p_range[1])
        SElist_train, SElist_test = np.zeros((2,p_range[1] - p_range[0]))

        for i in p_rangeObject:
            self.p = i + p_old
            self.craftX()
            dataset = self.kfold_yielder(k)
            for i in range(k):
                data = next(dataset)
                ypredict = self.OLS_core(data)[0]


            # X_train, X_test, y_train, y_test = train_test_split(self.X, self.yData)
            # ypredict_test, beta = self.OLS_core([X_train, X_test, y_train, y_test])
            # ypredict_train = X_train @ beta

            SElist_test[i-p_range[0]]  += self.MSE(y_test,  ypredict_test)/k
            SElist_train[i-p_range[0]] += self.MSE(y_train, ypredict_train)/k

        self.p = p_old

        plot(p_rangeObject, np.log(SElist_test), label="Test")
        plot(p_rangeObject, np.log(SElist_train), label="Train")
        legend()
        title("Logarithmic train v test plot for MSE")
        xlabel("Polynomial complexity")
        ylabel("Mean Square Error")
        show()


if __name__ == "__main__":
    print("""Tasks as interpreted:
    (x) Parametrize terrain data
    ( ) Apply all three models of [see below] to geographical data
        ( ) OLS (k-fold)
        ( ) Ridge (k-fold)
        ( ) Lasso (k-fold)
    ( ) Critically evaluate results and discuss "the applicabilty of these regression methods to the type of data presented here"
    """)
    
    polydegree = 5
    datafrac = 0.1
    OLSruns = 6

    finallearner = learningG(datafrac, polydegree)
    # finallearner.OLSeval(OLSruns)

    for name, result in zip(["R2", "MSE"], finallearner.OLSeval(OLSruns)):
        print("\n" + name + 
            ": ( avg =", sum(result)/len(result), 
            "), ( max = ",  max(result), 
            "), ( min = ", min(result),  ")\n",
            result)

    finallearner.plotOLS_trainvtest(OLSruns, [0,20])
    