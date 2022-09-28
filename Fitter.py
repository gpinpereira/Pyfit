import numpy as np
from scipy.optimize import curve_fit
import string

class Helper(object):
    def __init__(self) -> None:
        pass
    @staticmethod
    def make_histogram(x,y,bins=400, noise_factor=0.0):
        zh, edges = np.histogramdd(np.column_stack((x,y)), bins=bins, density=True)
        indexes = np.where(zh == zh.max())
        guess_x0, guess_y0, guess_amp = edges[0][indexes[0]][0], edges[1][indexes[1]][0], zh.max()
        zh = zh.T
        zh_R = zh.ravel()
        xh, yh = np.meshgrid(edges[0][:-1], edges[1][:-1])
        xh_R, yh_R = xh.ravel(), yh.ravel()
        filt_index = np.where(zh_R <= (zh_R.max()-zh_R.min())*noise_factor) #0.18
        zh_R[filt_index[0]] = 0.0
        return xh_R, yh_R, zh_R, guess_x0, guess_y0, guess_amp

class Result(object):
    def __init__(self,dict, identity=""):
        #self.identity = identity
        self.vars = []
        for varname in dict:
            setattr(self, varname, dict[varname])
            self.vars.append(dict[varname])
    def __str__(self):
        #out = "identity: "+str(self.__dict__["identity"])
        out= ", ".join([key + ": "+ str(self.__dict__[key]) for key in self.__dict__.keys()])
        return out
    def r(self):
        return self.vars

class Functions(object):

    def __init__(self) -> None:
        pass

    @staticmethod
    def gaussian_2d(xy, x0, y0, sigma_x, sigma_y, amp, theta):
        x = xy[0]
        y = xy[1]
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = amp*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0)  + c*((y-y0)**2)))
        return g
    
    @staticmethod
    def gaussian(x, amp, mean, sigma):
        t = (x-mean)/sigma
        return amp * np.exp(-t*t*0.5)

    @staticmethod
    def expo(x,p0,p1):
        return np.exp(p0+p1*x)

    @staticmethod
    def lin(x,m,b):
        return x*m+b


class Fitter(object):

    def __init__(self, fittype, *args):
        
        self.func = None
        self.params = None
        self.err = None
        self.par = None
        self.fittype = fittype
        if fittype == "linear":
            self.func = Functions.lin
        elif fittype == "gaussian":
            self.func = Functions.gaussian
        elif fittype == "gaussian2d":
            self.func = Functions.gaussian_2d
        elif fittype == "expo":
            self.func = Functions.expo
        elif fittype == "poly":
            self.func = Functions.lin
        self.ident = ",".join(self.func.__code__.co_varnames[:self.func.__code__.co_argcount])
        self.func_out = self.func

    def fit(self, x, y, p0=None):
        x, y = np.array(x), np.array(y)

        if self.fittype == "gaussian":
            self.fitGauss(x,y,p0)
        
        elif self.fittype == "gaussian2d":
            self.fitGauss2D(x,y,p0)

        elif self.fittype == "linear":
            self.fitLin(x,y,p0)

        elif self.fittype == "expo":
            self.fitExpo(x,y,p0) 

        elif self.fittype == "poly":
            self.fitPoly(x,y,p0) 

    def getParams(self):
        return self.params

    def getErrors(self):
        return self.err
    
    def function(self):
        return self.func_out

    def fitGauss(self, x, y, p0=None):
        def ngaussianfit(x, *params): #amp_l, mean_l, sigma_l
            y = np.zeros_like(x)
            for i in range(0, len(params), 3):
                y += self.func(x, params[i], params[i+1], params[i+2])
            return y

        if not hasattr(p0, '__len__'): #is not a list, tuple etc... means that we're fitting n gaussians. Where n=p0
            if p0==None: p0=1
            step = int(len(x)//p0)
            nps = p0
            p0 = []
            for s in range(0,nps):
                xg = x[s*step:s*step+step]
                yg = y[s*step:s*step+step]
                p0 += [np.max(yg),np.average(xg),np.std(xg)]
            print("Estimating p0 :", p0)
        par, cov = curve_fit(ngaussianfit, x, y, p0=p0,maxfev =50000)
        vars = []
        for i  in range(len(cov)):
            vars.append(cov[i][i])

        pars_dict = {}
        if len(par)<=3:
            pars_dict = {"amp":par[0],"mean":par[1], "sigma":par[2]} 
        else:
            for i in range(0, len(p0),3):
                pars_dict["amp_"+str(int(i/3))] = par[i] 
                pars_dict["mean_"+str(int(i/3))] = par[i+1] 
                pars_dict["sigma_"+str(int(i/3))] = par[i+2] 

        self.params = Result(pars_dict)
        self.err = vars
        self.par = par
        self.func_out = ngaussianfit

    def fitGauss2D(self, x, y, p0=None):
        ah_R, bh_R, zh_R, guess_x0, guess_y0, guess_amp = Helper.make_histogram(x,y,bins=400)
        def ngaussian2d(xy,*params): #x0, y0, sigma_x, sigma_y, amp, theta
            z = np.zeros_like(xy[0])
            for i in range(0, len(params), 6):
                z += self.func(xy, *params[i:i+6])
            return z

        if not hasattr(p0, '__len__'): #is not a list, tuple etc... means that we're fitting n gaussians. Where n=p0
            #print(x.shape)
            if p0==None: p0=1
            step = int(len(x)//p0)
            p0 = []
            for s in range(0,len(x), step):
                xg = x[s:s+step]
                yg = y[s:s+step]
                p0 += [np.average(xg),np.average(yg),np.std(xg),np.std(yg), guess_amp, 0]

        par, cov = curve_fit(ngaussian2d, xdata=(ah_R,bh_R), ydata=zh_R, p0=p0, maxfev = 2000, xtol=1e-10)
        vars = []
        for i  in range(len(cov)):
            vars.append(cov[i][i])

        pars_dict = {}
        for i in range(0, len(p0),6):
            pars_dict["x0_"+str(int(i/6))] = par[i] 
            pars_dict["y0_"+str(int(i/6))] = par[i+1] 
            pars_dict["sigma_x_"+str(int(i/6))] = par[i+2] 
            pars_dict["sigma_y_"+str(int(i/6))] = par[i+3] 
            pars_dict["amp_"+str(int(i/6))] = par[i+4] 
            pars_dict["theta_"+str(int(i/6))] = par[i+5] 

        self.params = Result(pars_dict)
        self.err = vars
        self.par = par
        self.func_out = ngaussian2d

    def fitLin(self,x,y,p0=None):
        
        if p0 == None:
            y0, y1 = np.min(y), np.max(y)
            x0, x1 = x[y==y0], x[y==y1]
            m0 = (y1-y0)/(x1-x0)
            b0 = y0-m0*x0
            p0 = (m0,b0)
        par, cov = curve_fit(self.func, x, y, p0=p0, maxfev = 2000, xtol=1e-10)
        vars = []
        for i  in range(len(cov)):
            vars.append(cov[i][i])

        pars_dict = {"m":par[0],"b":par[1]} 
        self.params = Result(pars_dict)
        self.err = vars
        self.par = par

    def fitExpo(self, x, y, p0=None):

        def makeExpoParam(x1,y1,x2,y2):
            p1 = (np.log(y1)-np.log(y2))/(x1-x2)
            p0 = np.log(y2)-p1*x2
            return p0, p1
        if p0 == None:
            t = (y>0)
            y0, y1 = np.min(np.array(y)[t]), np.max(np.array(y)[t])
            x0, x1 = x[y==y0], x[y==y1]
            p2,p1 = makeExpoParam(x0,y0,x1,y1)
            p0 = [p2, p1]
        par, cov = curve_fit(self.func, x, y, p0=p0, maxfev = 2000, xtol=1e-10)

        vars = []
        for i  in range(len(cov)):
            vars.append(cov[i][i])

        pars_dict = {"p0":par[0],"p1":par[1]} 
        self.params = Result(pars_dict)
        self.err = vars
        self.par = par

    def fitPoly(self, x, y,p0=None):
        
        def multipoly(x, *params):
            z = np.zeros_like(x)
            for i in range(0, len(params)):
                z += params[i]*x**((len(params)-1)-i)
            return z 

        if not hasattr(p0, '__len__'): #is not a list, tuple etc... means that we're fitting n gaussians. Where n=p0
            if p0==None: p0=1

            print("p0 ", p0)
            step = int(len(x)//p0)
            nps = p0
            print(step)
            p0 = []
            for s in range(nps):

                xg = x[s*step:s*step+step]
                yg = y[s*step:s*step+step]
                m = (np.max(yg)-np.min(yg))/(xg[yg==np.max(yg)][0]-xg[yg==np.min(yg)][0])
                p0 += [m]

            print("Estimate p0, ", p0)
        par, cov = curve_fit(multipoly, x, y, p0=p0, maxfev = 2000, xtol=1e-10)

        pnames = string.ascii_lowercase

        vars = []
        for i  in range(len(cov)):
            vars.append(cov[i][i])

        pars_dict = {}
        for i in range(0, len(p0)):
            pars_dict[pnames[i]] = par[i] 
        
        self.params = Result(pars_dict)
        self.err = vars
        self.par = par
        self.func_out = multipoly

    def evaluate(self, xx,yy=None):
        
        try:
            xx = np.array(xx)
            return self.func_out(xx,*self.par)
        except Exception:
            xx, yy = np.array(xx),np.array(yy) 
            return self.func_out(xx, yy,*self.par)
        except TypeError:
            print("Invalid input for function")

    def __str__(self):
        return self.ident

