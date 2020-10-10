from partB import learningB
from sklearn.model_selection import train_test_split
import numpy as np

class learningC(learningB):

    # I stole the shit out of this from StackOverflow
    # https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def kfold_splitter(self, k):
        X_dimx, X_dimy = self.X.shape
        X, y = self.unison_shuffled_copies(self.X, self.yData)
        splitsX = np.zeros((k, int(X_dimx/k), X_dimy))
        splitsY = np.zeros((k, int(X_dimx/k)))
        frac = np.floor(X_dimx/k)

        for i in range(k):
            index_0 = int(i*frac)
            index_1 = int((i+1)*frac)
            try:
                splitsX[i,:,:] = X[index_0:index_1,:]
                splitsY[i,:]   = y[index_0:index_1]
            except ValueError:
                raise Exception("kfold function does not split correctly")

        return splitsX, splitsY

    def kfold_yielder(self, k):
        splitsX, splitsY = self.kfold_splitter(k)

        pointsPerSplit = splitsX.shape[1]

        for i in range(k):
            X_test  = np.zeros((pointsPerSplit, self.nfeatures))
            Y_test  = np.zeros(pointsPerSplit)
            X_train = np.zeros(((k-1) * pointsPerSplit, self.nfeatures))
            Y_train = np.zeros((k-1) * pointsPerSplit)

            X_test[:,:] = splitsX[i,:,:]
            Y_test[:]   = splitsY[i,:]

            flattenedX = splitsX.reshape(-1, splitsX.shape[-1])
            flattenedY = splitsY.reshape(-1)
            
            X_train[:i*pointsPerSplit,:] = flattenedX[:i*pointsPerSplit,:]
            X_train[i*pointsPerSplit:,:] = flattenedX[(i+1)*pointsPerSplit:,:]

            Y_train[:i*pointsPerSplit] = flattenedY[:i*pointsPerSplit]
            Y_train[i*pointsPerSplit:] = flattenedY[(i+1)*pointsPerSplit:]

            yield X_train, X_test, Y_train, Y_test 

    def kfold(self, k, solver=None):
        self.craftX()
        dataset = self.kfold_yielder(k)
        r2list, SElist = np.zeros((2, k)) 
        for i in range(k):
            data = next(dataset)
            R2, MSE = self.OLS(data)[:2]
            r2list[i]  = R2
            SElist[i] = MSE
        return r2list, SElist







if __name__ == "__main__":
    print("""Task as interpreted:
    (X) Implement k-fold cross-validation, evaluate MSE from this
    (X) Compare MSE from k-fold and bootstrap
    (X) try 5-10 folds 
    """)
    
    Npoints = 100
    polydegree = 5
    # folds = 10
    for folds in [5,10]:
        sampleN = folds
        kfoldlearner = learningC(Npoints, polydegree)
        kfoldResults = np.asarray(kfoldlearner.kfold(folds)).sum(axis=1)/folds
        print("\nFolds/number of samples: ", folds)
        print("k-fold          [R2, MSE] :", kfoldResults)

        bootstrapResults = np.asarray(kfoldlearner.bootstrap(sampleN = sampleN)).sum(axis=1)/sampleN
        print("Bootstrap       [R2, MSE] :", bootstrapResults)  

        print("kfold/bootstrap [R2, MSE] :", kfoldResults/bootstrapResults)
