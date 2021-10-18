import numpy as np
import matplotlib.pyplot as plt
from simulation import *
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


# plot for true and fitted value
def plotfitted(debt,xi,X,Beta,name=0):
    y1=debt[:-1]
    y2=fitted(xi,X,Beta)
    T=y1.shape[0]
    x = np.linspace(0, T-1, T)

    plt.plot(x, y1,label="true")
    plt.plot(x, y2,label="fitted")

    plt.title('Comparison')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    if name!=0:
        plt.savefig(name)
    plt.show()
    
def plotfordebt(xi,debt,name):
    plotdata=np.diag(debt[:-1]).dot(np.transpose([np.argmin(xi,axis=1),np.argmax(xi,axis=1)]))

    names = range(plotdata.shape[0])
    names = [str(x) for x in list(names)]

    x1 = np.array(range(len(names)))
    z = np.argmax(xi,axis=1)
    y = np.sum(plotdata,axis=1)
    
    
    cmap = ListedColormap(['r', 'b'])
    norm = BoundaryNorm([-1, 0.5, 1.5], cmap.N)

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be numlines x points per line x 2 (x and y)
    points = np.array([x1, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the line collection object, setting the colormapping parameters.
    # Have to set the actual values used for colormapping separately.
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(z)
    lc.set_linewidth(1)

    fig1 = plt.figure()
    ax=plt.gca()
    ax.add_collection(lc)

    
    plt.xlim(x1.min()-1,x1.max()+1)
    plt.ylim(y.min()-1, y.max()+1)
    #ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y'))#设置时间标签显示格式
    #ax.xaxis.set_major_locator(mdate.YearLocator())
    plt.xticks(rotation=45)#旋转45度显示
    daterange=[str(x) for x in range(2003,2020)]
    plt.xticks(range(0,203,12),daterange)
    plt.xlabel("Year")
    plt.ylabel("Rate")

    plt.savefig(name)
    plt.show()
