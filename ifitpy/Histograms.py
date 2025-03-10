from scipy import stats
import numpy as np


class Histogram(object):

    def __init__(self, bins, xmin=None, xmax=None):
        self.xmin = xmin
        self.xmax = xmax
        self.bins = bins 

        self.profy = None
        self.profx = None
        self.profyerr = None
        self.profbincount = None

    def fill(self,x):
        bins = self.bins
        ra = [self.xmin, self.xmax]
        if ra[0] == None or ra[1] == None: ra = None
        yrr = stats.binned_statistic(range(len(x)), x, 'std', bins=bins, range=ra).statistic
        count = stats.binned_statistic(range(len(x)), x, 'count', bins=bins, range=ra).statistic
        y, edge = np.histogram(x, density=False,bins=bins)
        x = (edge[:-1]+edge[1:])*0.5 

        self.profy = y
        self.profx = x
        self.profyerr = yrr
        self.profbincount = count

    def getSigmas(self):
        return self.profyerr
    def getBins(self):
        return self.profx
    def getMeans(self):
        return self.profy
    def getCount(self):
        return self.profbincount

class Profile1D(object):


    def __init__(self, bins, xmin=None, xmax=None):
        
        self.xmin = xmin
        self.xmax = xmax
        self.bins = bins 

        self.profy = None
        self.profx = None
        self.profyerr = None
        self.profbincount = None

    def fill(self,x,y):
        bins = self.bins
        ra = [self.xmin, self.xmax]
        if ra[0] == None or ra[1] == None: ra = None
        yrr = stats.binned_statistic(x, y, 'std', bins=bins, range=ra).statistic
        y, edge, _ = stats.binned_statistic(x, y, 'mean', bins=bins, range=ra)
        self.profx = (edge[:-1]+edge[1:])*0.5  
        self.profy = y
        self.profyerr = yrr
        self.profbincount = stats.binned_statistic(x, y, 'count', bins=bins, range=ra).statistic

    def getSigmas(self):
        return self.profyerr
    def getBins(self):
        return self.profx
    def getMeans(self):
        return self.profy
    def getCount(self):
        return self.profbincount



class Profile2D(object):

    def __init__(self, binsx, binsy, xmin=None, xmax=None, ymin=None, ymax=None):
        
        self.xmin = xmin
        self.xmax = xmax
        self.binsx = binsx 

        self.ymin = ymin
        self.ymax = ymax
        self.binsy = binsy 

        self.profy = None
        self.profxb = None
        self.profyb = None
        self.profyerr = None
        self.profbincount = None

    def fill(self,x,y):
        bins = [self.binsx, self.binsy]
        ra = [[self.xmin, self.xmax], [self.ymin, self.ymax]]

        if (np.array(ra)==None).any(): ra = None
        sigma = stats.binned_statistic_2d(x, y, np.arange(len(x)), 'std', bins=bins, range=ra).statistic
        count = stats.binned_statistic_2d(x, y, np.arange(len(x)), "count", bins=bins, range=ra).statistic
        res = stats.binned_statistic_2d(x, y, np.arange(len(x)), "mean", bins=bins, range=ra)
        xh, yh = (res.x_edge[:-1]+res.x_edge[1:])*0.5, (res.y_edge[:-1]+res.y_edge[1:])*0.5   

        xh, yh = np.meshgrid(xh, yh, indexing='ij')
        xh, yh = xh.ravel(), yh.ravel()

        self.profxb = xh 
        self.profyb = yh 
        self.profy = res.statistic.ravel()
        self.profyerr = sigma.ravel()
        self.profbincount = count.ravel()

    def getSigmas(self):
        return self.profyerr
    def getBinsX(self):
        return self.profxb
    def getBinsY(self):
        return self.profyb
    def getMeans(self):
        return self.profy
    def getCount(self):
        return self.profbincount