import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, title, imshow, xlabel, ylabel, show, savefig
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from partE import learningE

class learningF(learningE):
    def __init__(self, n, p, noisefactor=None):
        super().__init__(n, p, noisefactor)
        self.imageFilePath  = "geodata/"


    def loadTerrainData(self, filename=None):
        if filename is None: filename = "geodata/n45_e069_1arc_v3.tif"
        self.terrainData = imread(filename)

if __name__ == "__main__":
    print("""Task as interpreted:
    (x) download data and manage to import it
    """)

    Npoints = 2
    polydegree = 0
    # Datapoints moot

    terrainlearner = learningF(Npoints, polydegree)
    terrainlearner.loadTerrainData()

    # Show the terrain
    imageFileName = "geodata_area"
    filetype = ".png"
    figure()
    title("Terrain data")
    imshow(terrainlearner.terrainData, cmap="gray")
    xlabel("X")
    ylabel("Y")
    savefig(terrainlearner.imageFilePath + imageFileName + filetype)
    show()
