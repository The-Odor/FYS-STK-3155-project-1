import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import plot, show, legend
import math as mt


# Uses a two-dimensional polynomial of degree p to model Frankes Function
class learningA:
    def __init__(self, n, p, noisefactor=None):
        if noisefactor is None: noisefactor = 0.1
        x1 = y1 = np.linspace(0, 1, mt.ceil(n/2))
        x2 = np.random.uniform(0, 1, mt.floor(n/2))
        y2 = np.random.uniform(0, 1, mt.floor(n/2))

        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))

        x, y = np.meshgrid(x,y)
        self.x = x.flatten()
        self.y = y.flatten()

        self.ytrue = self.FrankeFunction(self.x, self.y)
        self.yData = self.ytrue + noisefactor*np.random.randn(n**2)*self.ytrue.mean()

        self.n = n
        self.p = p
        self.noisefactor = noisefactor
        self.nfeatures = int(((self.p+1)*(self.p+2))/2)

    def MSE(self, y_data, y_model):
        # Optimal value is 0, with higher values beign worse
        return np.sum((y_data-y_model)**2) / np.size(y_model)

    def R2(self, y_data, y_model):
        # Optimal value is 1, with 0 implying that model performs 
        # exactly as well as predicting using the data average would.
        # Lower values imply that predicting using the average would be better
        top = np.sum((y_data - y_model)**2)
        bot = np.sum((y_data - np.mean(y_data))**2)
        return 1 - top/bot

    def FrankeFunction(self, x,y):
        term1 = 0.75*np.exp(-(9*x-2)**2/4.00 - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-(9*x+1)**2/49.0 - 0.10*(9*y+1))
        term3 = 0.50*np.exp(-(9*x-7)**2/4.00 - 0.25*((9*y-3)**2))
        term4 =-0.20*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

    def generateNewDataset(self, n=None, p=None):
        if n is None: n = self.n
        else: self.n = n
        if p is None: p = self.p
        else: self.p = p

        x1 = y1 = np.linspace(0, 1, mt.ceil(n/2), dtype=int)
        x2 = np.random.uniform(0, 1, mt.floor(n/2))
        y2 = np.random.uniform(0, 1, mt.floor(n/2))

        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))
    
        x, y = np.meshgrid(x,y)
        self.x = x.flatten()
        self.y = y.flatten()

        self.ytrue = self.FrankeFunction(self.x, self.y)
        self.yData = self.ytrue + self.noisefactor*np.random.randn(n**2)*self.ytrue.mean()

    def craftX(self, scaling=True):
        self.nfeatures = int(((self.p+1)*(self.p+2))/2)
        self.X = np.zeros((len(self.x), self.nfeatures))

        ind = 0
        for i in range(self.p+1):
            for j in range(self.p+1-i):
                self.X[:,ind] = self.x**i * self.y**j
                ind += 1

        if scaling:
            self.X[:,1:] -= np.mean(self.X[:,1:], axis=0)


    def OLS_core(self, dataset):
        X_train, X_test, y_train = dataset[:3]
        beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
        ypredict = X_test @ beta
        return ypredict, beta

    # Ordinary Least Square solution 
    def OLS(self, dataset=None):
        if dataset is None:
            self.craftX()
            dataset = train_test_split(self.X, self.yData)
            ypredict, beta = self.OLS_core((dataset))
        else:
            ypredict, beta = self.OLS_core(list(dataset))
        return self.R2(dataset[3], ypredict), self.MSE(dataset[3], ypredict), beta

    # Generates R2- and MSE-values from the OLS function
    def OLSeval(self, reps):
        r2list, SElist = np.zeros((2, reps))
        for i in range(reps):
            r2list[i], SElist[i] = self.OLS()[:2]

        return r2list, SElist
        
    # Calculates confidence interval of the parameters for OLS
    def confidenceIntervalOLS(self, reps=None):
        if reps is None: reps = 300
        betas = np.zeros((reps, self.nfeatures))
        for rep in range(reps):
            betas[rep,:] = self.OLS()[2]

        # Calculating the standard deviation
        betasAvg = np.sum(betas, axis = 0)/reps
        betasDiff = betas - betasAvg
        
        STD = np.sqrt((1/(reps-self.p-1))*np.sum(betasDiff**2, axis = 0))

        magicNumberFor95PercentConfidenceInterval = 1.645
        ShortHand = magicNumberFor95PercentConfidenceInterval

        confidenceInterval = [betasAvg-ShortHand*STD, betasAvg+ShortHand, STD**2]

        # returns lower and upper boundaries of confidence interval with the variance
        return confidenceInterval
  


if __name__ == "__main__":
    print("""Task as interpreted: 
    (x) Generate a dataset as a bipolynomial to a general order
    (x) Implement and execute OLS on bipolynomial and Franke
    (x) Find confidence interval of parameters beta
    (x) Evaluate R2 and mean square error
    (x) Scale and split the data
    """)



    Npoints = 100
    polydegree = 5
    OLSruns = 18

    OLSlearner = learningA(Npoints, polydegree)

    print("Npoints = ", Npoints, ", giving a number of data points = ", Npoints**2, sep="")
    print("Polynomial degree = ", polydegree, ", giving a number of features = ",int(((polydegree+1)*(polydegree+2))/2), sep="")
    print("OLS run", OLSruns, "times")

    for name, result in zip(["R2", "MSE"], OLSlearner.OLSeval(OLSruns)):
        print("\n" + name + 
            ": ( avg =", sum(result)/len(result), 
            "), ( max = ",  max(result), 
            "), ( min = ", min(result),  ")\n",
            result)

    lower, upper, STD = OLSlearner.confidenceIntervalOLS()
    print("""\n95 Confidence Interval for beta: 
    Lower: {}\n
    Upper: {}\n
    STD  : {}""".format(lower,upper,STD))


    # Changing the data generation method
    x = np.linspace(0,1,OLSlearner.n)
    y = np.linspace(0,1,OLSlearner.n)
    x, y = np.meshgrid(x,y)
    OLSlearner.x = x.flatten()
    OLSlearner.y = y.flatten()
    OLSlearner.ytrue = OLSlearner.FrankeFunction(OLSlearner.x, OLSlearner.y)
    OLSlearner.yData = OLSlearner.ytrue + OLSlearner.noisefactor*np.random.randn(OLSlearner.n**2)*OLSlearner.ytrue.mean()


    print("\n\nOLS with data determined just by linspace")
    for name, result in zip(["R2", "MSE"], OLSlearner.OLSeval(OLSruns)):
        print("\n" + name + 
            ": ( avg =", sum(result)/len(result), 
            "), ( max = ",  max(result), 
            "), ( min = ", min(result),  ")\n",
            result)

