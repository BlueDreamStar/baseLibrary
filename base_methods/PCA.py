# -*- coding: utf-8 -*-
# @Author: dreamBoy
# @Date:   2019-04-27 22:27:30
# @Email:  wpf2106@gmail.com
# @Desc:   Welcome to my world!
# @Motto:  Brave & Naive!
# @Last Modified by:   BlueDreamStar
# @Last Modified time: 2019-06-06 11:55:38
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import getopt
import sys
import time
import os
reload(sys)
sys.setdefaultencoding('utf-8')

# colors = ['navy', 'turquoise', 'darkorange']
colors = ['Maroon', 'Brown', 'Olive', 'Teal', 'Navy', 'Black', 'Red', 'Orange', 'Yellow', \
        'Lime', 'Green', 'Cyan', 'Blue', 'Purple', 'Magenta', 'Grey', 'Pink', '#ffd8b1', \
        'Beige', '#aaffc3', 'Lavender', 'Turquoise']
# colors=["#F5F5F5","#DCDCDC","#A9A9A9","#808080","#696969"]
# colors=["#F2F2F2","#BDBDBD","#848484","#424242","#000000"]
# colors=["#FAFAFA","#D8D8D8","#A4A4A4","#585858","#000000"]
DataDir='data/PCA_test'
DataDir_test='data/PCA_testing'
SaveDir='outcome/'

### 
ClusterNumber = 22
ClusterFlag = False
Dimension = 3

if not os.path.exists(SaveDir):
    os.makedirs(SaveDir)

curTime = time.strftime("%Y%m%d-%H%M%s", time.localtime())
# print(curTime)

class PCAParams():
    def __init__(self):
        #### init default values
        self.dimension = Dimension
        self.clusterFlag = ClusterFlag
        self.clusterNumber = ClusterNumber

    def setDimension(self, dimension):
        self.dimension = dimension
        # print(self.dimension)

    def setClusterFlag(self, clusterFlag):
        self.clusterFlag = clusterFlag
        # print(self.clusterFlag)
        # 
    def setClusterNumber(self, clusterNumber):
        self.clusterNumber = clusterNumber
        # print(self.clusterFlag)

def usage():
    print("""
        -h help 
        -c add cluster
        -n cluster numbers (Max support 22 kinds of colors)
        -d dimension
        """)

def loadData(file_dir):
    data = np.loadtxt(file_dir,dtype=np.float64,delimiter=",")
    # userIDs = data[:,0]
    # latentValues = data[:,1:]
    # dataParts = data.strip().split('\t')
    # userIDs = dataParts[0]
    # latentValues = dataParts[1].split(',')
    # return userIDs, latentValues
    return data

def plotFigure( outcomes, kmeans_label, clusterNumber, typeStr):
    fig = plt.figure()
    for color, i in zip(colors, np.arange(clusterNumber)):
        plt.scatter(outcomes[kmeans_label == i, 0], outcomes[kmeans_label == i, 1], alpha=.8, color=color,label="cluster_%s" %i)
        # print(outcomes[kmeans_label == i].shape)
    # plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.legend(bbox_to_anchor=(1.05, 1), shadow=False, loc=2, borderaxespad=0.)
    plt.title('Analysis of dimension reduced: D=2 Cluster=%s' %clusterNumber)
    #### save figure
    fig.savefig('%s2D_Cluster%s%s.png' %(SaveDir,clusterNumber,typeStr), bbox_inches='tight', format='png', dpi=1000)
    # fig.savefig('22.pdf', bbox_inches='tight', format='pdf', dpi=1000)
    # (bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # loc='center left', bbox_to_anchor=(1.0, .5)
    plt.close('all')
    # plt.show()

def plot3DFigure( outcomes, kmeans_label, clusterNumber):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for color, i in zip(colors, np.arange(clusterNumber)):
        ax.scatter(outcomes[kmeans_label == i, 0], outcomes[kmeans_label == i, 1], outcomes[kmeans_label == i, 2], c=color, marker= 'o')
    plt.title('analysis of dimension reduced: D=2 Cluster=%s' %clusterNumber)
    #### add legend
    tempColors = colors[:clusterNumber]
    custom_lines = [plt.Line2D([],[], ls="", marker='.', \
                mec='k', mfc=c, mew=.1, ms=20) for c in tempColors]
    labelTups = np.arange(clusterNumber)
    ax.legend(custom_lines, ["cluster_%s" %i for i in labelTups], \
          loc='center left', bbox_to_anchor=(1.0, .5))
    #### save figure
    fig.savefig('%s3D_Cluster%s.png' %(SaveDir,clusterNumber), bbox_inches='tight', format='png', dpi=1000)
    # fig.savefig('223D.pdf', bbox_inches='tight', format='pdf', dpi=1000)
    plt.close('all')
    # plt.show()


######################
### lda function
######################

if __name__ == '__main__':

    pcaParams = PCAParams()
    # print(pcaParams.dimension)
    try:
        options, args = getopt.getopt(sys.argv[1:], "hc:n:d:", ["help", "cluster=", "number=", "dimension="])
    except getopt.GetoptError:
        print('hi Sam getopt error!Please input -h or --help')
        sys.exit(1) 

    for name, value in options:
        if name in ("-h", "--help"):
            usage()
        elif name in("-c","--cluster"):
            # print("cluster %s" % (value))
            pcaParams.setClusterFlag(value)
        elif name in("-n","--number"):
            # print("cluster %s" % (value))
            pcaParams.setClusterNumber(value)
        elif name in("-d","--dimension"):
            # print("dimension %s", %(value))
            pcaParams.setDimension(value)
    # for item in args:
        # print(item)

    ##### load data #######
    latentValues = loadData(DataDir)
    latentValues_test = loadData(DataDir_test)
    dataLength = len(latentValues)

    ##### PCA
    # pca = PCA(pcaParams.dimension)
    # pca.fit(latentValues)
    # outcomes = pca.transform(latentValues)
    # 
    ##### if need cluster
    # if(pcaParams.clusterFlag):
    #     kmeans = KMeans(n_clusters=pcaParams.clusterNumber, random_state=0).fit(outcomes)
    #     kmeans_label = kmeans.labels_
        # print("kmeans_label",kmeans_label)
    pca2D = PCA(2)
    pca2D.fit(latentValues)

    pca3D = PCA(3)
    pca3D.fit(latentValues)
    
    outcomes2D = pca2D.transform(latentValues)
    outcomes2D_test = pca2D.transform(latentValues_test)

    outcomes3D = pca3D.transform(latentValues)
    outcomes3D_test = pca3D.transform(latentValues_test)

    kmeans_label = np.zeros( dataLength, int)
    plotFigure( outcomes2D, kmeans_label, 1, "train")
    plot3DFigure( outcomes3D, kmeans_label, 1)

    for clusterNums in range(20, pcaParams.clusterNumber + 1):
        kmeans2D = KMeans(n_clusters=clusterNums, random_state=0).fit(outcomes2D)
        kmeans2D_test_label = kmeans2D.predict(outcomes2D_test)
        kmeans3D = KMeans(n_clusters=clusterNums, random_state=0).fit(outcomes3D)
        kmeans3D_test_label = kmeans3D.predict(outcomes3D_test)
        np.savetxt("kmeans2D_test_label%s" %clusterNums, kmeans2D_test_label, fmt='%d', delimiter=',')
        kmeans_label2D = kmeans2D.labels_
        kmeans_label3D = kmeans3D.labels_
        plotFigure( outcomes2D, kmeans_label2D, clusterNums, "train")
        plotFigure( outcomes2D_test, kmeans2D_test_label, clusterNums, "test")
        plot3DFigure( outcomes3D, kmeans_label3D, clusterNums)   