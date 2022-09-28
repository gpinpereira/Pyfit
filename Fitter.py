import numpy as np
from scipy.optimize import curve_fit

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
        self.ident = ",".join(self.func.__code__.co_varnames[:self.func.__code__.co_argcount])
        self.func_out = self.func

    def fit(self, x, y, p0=None):
        if self.fittype == "gaussian":
            self.fitGauss(x,y,p0)
        
        elif self.fittype == "gaussian2d":
            self.fitGauss2D(x,y,p0)

        elif self.fittype == "linear":
            self.fitLin(x,y)

        elif self.fittype == "expo":
            self.fitExpo(x,y) 

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
        par, cov = curve_fit(ngaussianfit, x, y, p0=p0,maxfev =20000)
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
            z = np.zeros_like(xy)
            for i in range(0, len(params), 6):
                z += self.func(x, params[i], params[i+1], params[i+2])
            return z

        par, cov = curve_fit(ngaussian2d, xdata=(ah_R,bh_R),ydata=zh_R, p0=p0, maxfev = 2000, xtol=1e-10)
        vars = []
        for i  in range(len(cov)):
            vars.append(cov[i][i])

        pars_dict = {"x0":par[0],"y0":par[1], "sigma_x":par[2], "sigma_y":par[3], "amp":par[4], "theta":par[5]} 

        self.params = Result(pars_dict)
        self.err = vars
        self.par = par
        self.func_out = ngaussian2d

    def fitLin(self,x,y):
        y0, y1 = np.min(y), np.max(y)
        x0, x1 = x[y==y0], x[y==y1]
        m0 = (y1-y0)/(x1-x0)
        b0 = y0-m0*x0

        par, cov = curve_fit(self.func, x, y, p0=(m0,b0), maxfev = 2000, xtol=1e-10)
        vars = []
        for i  in range(len(cov)):
            vars.append(cov[i][i])

        pars_dict = {"m":par[0],"b":par[1]} 
        self.params = Result(pars_dict)
        self.err = vars
        self.par = par

    def fitExpo(self, x, y):

        def makeExpoParam(x1,y1,x2,y2):
            p1 = (np.log(y1)-np.log(y2))/(x1-x2)
            p0 = np.log(y2)-p1*x2
            return p0, p1

        t = (y>0)
        y0, y1 = np.min(np.array(y)[t]), np.max(np.array(y)[t])
        x0, x1 = x[y==y0], x[y==y1]
        p0,p1 = makeExpoParam(x0,y0,x1,y1)
        print(p0, p1)

        #H, xedges,yedges = np.histogram2d(y,x,bins=(200,200))
        #print(H)
        #print(xedges.ravel())
        #print(yedges)

        print(self.func)
        par, cov = curve_fit(self.func, x, y, p0=(p0, p1), maxfev = 2000, xtol=1e-10)
        #ah_R, bh_R, zh_R, guess_x0, guess_y0, guess_amp = Helper.make_histogram(x,y,bins=200, noise_factor=0.4)

        #print(bh_R)
        #print(ah_R)
        #par, cov = curve_fit(self.func, ah_R, bh_R, sigma=zh_R, p0=(p0, p1), maxfev = 20000, xtol=1e-10)
        vars = []
        for i  in range(len(cov)):
            vars.append(cov[i][i])

        pars_dict = {"p0":par[0],"p1":par[1]} 
        self.params = Result(pars_dict)
        self.err = vars
        self.par = par

    def evaluate(self, xx,yy=None):
        try:
            return self.func_out(xx,*self.par)
        except Exception:
            return self.func_out(xx, yy,*self.par)
        except TypeError:
            print("Invalid input for function")

    def __str__(self):
        return self.ident
