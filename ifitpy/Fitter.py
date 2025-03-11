import numpy as np
from scipy.optimize import curve_fit
import string
from scipy import stats
from iminuit import Minuit
from iminuit.cost import LeastSquares
from . import Histograms as ht
from scipy.interpolate import UnivariateSpline


class Utils(object):
    def __init__(self):
        pass
    
    @staticmethod
    def makeExpoParam(x1,y1,x2,y2):
        p1 = (np.log(y1)-np.log(y2))/(x1-x2)
        p0 = np.log(y2)-p1*x2
        return p0, p1

    # Find roots where first derivative is zero, and second derivative is negative.
    # Get n roots of the highest yy values 
    # Used to estimate the initial means of the gaussian fit
    @staticmethod
    def get_xx_of_n_max(xx, yy, n):
        

        w=5 #moving average with a window of length
        ty=np.convolve(yy, np.ones(w), 'same') / w

        #Make a spline and calculate its derivative roots (position of maximums and minimus)
        y_spl = UnivariateSpline(xx,ty,s=0,k=4)
        y_spl_1d = y_spl.derivative(n=1)
        roots =  y_spl_1d.roots()

        #Make a second-derivative roots (position of maximums)
        y_spl_2d = y_spl.derivative(n=2)
        value_2d_at_root = y_spl_2d(roots)
        potencial_means = roots[(value_2d_at_root<0)]

        yy_roots_values = y_spl(potencial_means)
        ind = np.argpartition(yy_roots_values, -n)[-n:]
        return potencial_means[ind]


#Result class to generate named-values that can get used after a fit 
#outputs: 
#    vars: [np.float64(3346), np.float64(49), np.float64(10)], amp: 3346, mean: 49, sigma: 10
class Result(object):
    def __init__(self, dict, identity=""):
        self.vars = []
        for varname in dict:
            setattr(self, varname, dict[varname])
            self.vars.append(dict[varname])
            
    def __str__(self):
        out= ", ".join([key + ": "+ str(self.__dict__[key]) for key in self.__dict__.keys()])
        return out


#Prepares pre-defined functions for fitting
class Functions(object):

    def __init__(self):
        pass

    @staticmethod
    def gaussian_2d(xy, x0, y0, sigma_x, sigma_y, amp, theta):
        #print(x0, y0, sigma_x, sigma_y, amp, theta)
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
    def expo(x, p0, p1):
        return np.exp(p0+p1*x)

    @staticmethod
    def lin(x, m, b):
        return x*m+b


class Fitter(object):

    def __init__(self, fittype, *args):
        
        self.func = None # based function for fitting
        self.params = None # Results after fitting
        self.err = None # errors of each parameter found in the fit
        self.par = None # list with paramater. Mainly used for internal evaluation
        self.p0 = None # list with paramater. Mainly used for internal evaluation
        self.chi2 = None # chi2 of the fit
        self.dof = None # degrees of the fit
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
        self.ident = ",".join(self.func.__code__.co_varnames[:self.func.__code__.co_argcount]) # to print the arguments of the fittin function (x,amp,mean,sigma in case of Gaussian)
        self.func_out = self.func # actuall fitting function, which sometimes can be a sum of self.func
        self.binned = False # wether to bin data or not

        #information for the profile histogram
        self.profx = None
        self.profy = None
        self.profyrr = None

        #to wether use minuit or not
        self.minuitactive=False

    def disableMinuit(self):
        self.minuitactive =False
    def enableMinuit(self):
        self.minuitactive = True

    # Makes histograms from raw data 
    def fitBinned(self, x, y=np.array([]), p0=None, bins=100, n=1):
        x, y = np.array(x), np.array(y)
        prof = None

        if y.shape[0] == 0:
            prof = ht.Histogram(bins)
            prof.fill(x)
        else:
            prof = ht.Profile1D(bins)
            prof.fill(x,y)

        self.profx, self.profy, self.profyrr = prof.getBins(), prof.getMeans(), prof.getSigmas()
        self.fit(self.profx, self.profy, self.profyrr,p0,n)

    def fit(self, x, y=None, yerr=None, p0=None,n=1):
        x, y = np.array(x), np.array(y)

        if self.fittype == "gaussian":
            self.fitGauss(x,y,yerr,p0,n)
        
        elif self.fittype == "gaussian2d":
            self.fitGauss2D(x,y,p0,n)

        elif self.fittype == "linear":
            self.fitLin(x,y,yerr,p0)

        elif self.fittype == "expo":
            self.fitExpo(x,y,yerr,p0) 

        elif self.fittype == "poly":
            self.fitPoly(x,y,p0) 

    #x, y and yerr are ordered
    def fitGauss(self, x, y=None, yerr=None, p0=None,n=1):

        #prepare a function that is represented by the sum of n gaussians
        def ngaussianfit(x, *params): #amp_l, mean_l, sigma_l
            y = np.zeros_like(x)
            for i in range(0, len(params), 3):
                y += self.func(x, params[i], params[i+1], params[i+2])
            return y 

        # in case of using n paramters to define the number of gaussians 
        # estimate initial paramters 
        if not hasattr(p0, '__len__'): #is not a list, tuple etc... means that we're fitting n gaussians. Where n=p0
            if p0==None: p0=1

            xt = np.array_split(x, n) 
            yt = np.array_split(y, n) 

            # calculated means based on splines. 
            # The output doe snto exists necessarly in x
            means = np.sort(Utils.get_xx_of_n_max(x,y,n))

            p0 = []
            for s in range(0,n):
                shifted = x-means[s]
                shifted_mod = np.abs(shifted) # nearest value of zero is the position of the mean
                true_mean = x[shifted_mod==shifted_mod.min()][0]
                estimated_amp = y[shifted_mod==shifted_mod.min()][0]

                selL = (shifted<0) & (y<estimated_amp*0.5)
                selR = (shifted>0) & (y<estimated_amp*0.5)
                xl =  (x[selR].min() - x[selL].max())/2.3548 # calculate std from fwhm
                #print("mean: ", true_mean, means[s])
                
                xg = xt[s]
                yg = yt[s]
                #print("amp: ", estimated_amp, np.max(yg))
                #print("sigma: ", xl, np.std(xg))
                p0 += [estimated_amp , true_mean, xl]

        #prepare parameters boundaries 
        bounds_hi = []
        bounds_lo = []
        for i in range(0,n):
            bounds_lo += [0,-np.inf,0]
            bounds_hi += [np.inf,np.inf,np.inf]
        par, cov = self.fitter(ngaussianfit, x, y, yerr, p0=p0,bounds=(bounds_lo, bounds_hi))

        #prepare named values
        pars_dict = {}
        if len(par)<=3:
            pars_dict = {"amp":par[0],"mean":par[1], "sigma":par[2]} 
        else:
            for i in range(0, len(p0),3):
                pars_dict["amp_"+str(int(i/3))] = par[i] 
                pars_dict["mean_"+str(int(i/3))] = par[i+1] 
                pars_dict["sigma_"+str(int(i/3))] = par[i+2] 

        self.params = Result(pars_dict)
        self.func_out = ngaussianfit

    def fitGauss2D(self, x, y, p0=None, n=1):

        pf2d = ht.Profile2D(binsx=400, binsy=400)
        pf2d.fill(x, y)
        ah_R, bh_R, zh_R, sigs = pf2d.getBinsX(), pf2d.getBinsY(), pf2d.getCount(), pf2d.getSigmas()
        
        def ngaussian2d(xy,*params): #x0, y0, sigma_x, sigma_y, amp, theta

            z = np.zeros_like(xy[0])
            #print(z)
            for i in range(0, len(params), 6):
                g1 = self.func(xy, *params[i:i+6])
                z += g1
            return z

        if not hasattr(p0, '__len__'): #is not a list, tuple etc... means that we're fitting n gaussians. Where n=p0
            p0 = []

            xt = np.array_split(np.linspace(x.min(),x.max(), n*10), n)
            yt = np.array_split(np.linspace(y.min(),y.max(), n*10), n)

            for s in range(0,n):
                tcut = (x>xt[s].min())&(x<xt[s].max())&(y>yt[s].min())&(y<yt[s].max())
                xg = x[tcut]
                yg = y[tcut]
                p0 += [np.average(xg),np.average(yg),np.std(xg),np.std(yg), zh_R.max()/n, -0.001]

        par, cov = self.fitter(func=ngaussian2d,x=(ah_R,bh_R),y=zh_R, yerr=sigs,p0=p0)

        pars_dict = {}
        for i in range(0, len(p0),6):
            pars_dict["x0_"+str(int(i/6))] = par[i] 
            pars_dict["y0_"+str(int(i/6))] = par[i+1] 
            pars_dict["sigma_x_"+str(int(i/6))] = par[i+2] 
            pars_dict["sigma_y_"+str(int(i/6))] = par[i+3] 
            pars_dict["amp_"+str(int(i/6))] = par[i+4] 
            pars_dict["theta_"+str(int(i/6))] = par[i+5] 

        self.params = Result(pars_dict)
        self.func_out = ngaussian2d

    def fitLin(self,x,y,yerr=None,p0=None):
        
        if p0 == None:
            y0, y1 = np.min(y), np.max(y)
            x0, x1 = x[y==y0], x[y==y1]
            m0 = (y1-y0)/(x1-x0)
            b0 = y0-m0*x0
            p0 = (m0,b0)

        par, cov = self.fitter(self.func,x,y,yerr,p0=p0)

        pars_dict = {"m":par[0],"b":par[1]} 
        self.params = Result(pars_dict)

    def fitExpo(self, x, y, yerr=None, p0=None):

        if p0 == None:
            t = (y>0)
            y0, y1 = np.min(np.array(y)[t]), np.max(np.array(y)[t])
            x0, x1 = x[y==y0], x[y==y1]
            p2,p1 = Utils.makeExpoParam(x0,y0,x1,y1)
            p0 = [p2, p1]

        par, cov = self.fitter(self.func,x,y,yerr,p0=p0)

        pars_dict = {"p0":par[0],"p1":par[1]} 
        self.params = Result(pars_dict)

    def fitPoly(self, x, y,p0=None):
        
        def multipoly(x, *params):
            z = np.zeros_like(x)
            for i in range(0, len(params)):
                z += params[i]*x**((len(params)-1)-i)
            return z 

        if not hasattr(p0, '__len__'): #is not a list, tuple etc... means that we're fitting n gaussians. Where n=p0
            if p0==None: p0=1

            step = int(len(x)//p0)
            nps = p0
            p0 = []
            for s in range(nps):
                xg = x[s*step:s*step+step]
                yg = y[s*step:s*step+step]
                m = (np.max(yg)-np.min(yg))/(xg[yg==np.max(yg)][0]-xg[yg==np.min(yg)][0])
                p0 += [m]
        
        par, cov = self.fitter(multipoly,x,y,p0=p0)

        pnames = string.ascii_lowercase
        pars_dict = {}
        for i in range(0, len(p0)):
            pars_dict[pnames[i]] = par[i] 
        
        self.params = Result(pars_dict)
        self.func_out = multipoly

    def fitter(self,func,x,y,yerr=np.array([]),p0=None, bounds=None):
        
        self.p0 = p0
        if isinstance(yerr, np.ndarray): 
            if((yerr==None).all()):
                yerr = np.array([1e-9] * y.shape[0])
        elif yerr == None:
            yerr = np.array([1e-9] * y.shape[0])
        
        if bounds == None or self.fittype == "gaussian2d":
            if (self.fittype == "gaussian2d"):
                yerr=None

            par, cov = curve_fit(func, x, y, sigma=yerr, p0=p0, maxfev = 10000, xtol=1e-8)
        else:
            par, cov = curve_fit(func, x, y, sigma=yerr, p0=p0, maxfev = 10000, xtol=1e-8, bounds=bounds)
        
        if self.minuitactive:
            ls = LeastSquares(x=x, y=y, yerror=yerr, model=func)
            m = Minuit(ls, *par)
            m.migrad()  # finds minimum of least_squares function
            m.hesse()   # accurately computes uncertainties
            names, par, cov = m.parameters, m.values, m.covariance 
        vars = []
        for i  in range(len(par)):
            #print("c ", cov[i][i])
            if cov is None:continue
            vars.append(cov[i][i]**0.5)

        self.err = vars
        self.par = par
        self.chi2 = 0 #m.fval
        self.dof = 10 #len(x) - m.nfit
        return par, cov


    def getParams(self):
        return self.params

    def getInitParams(self):
        return self.p0

    def getErrors(self):
        return self.err
    
    def function(self):
        return self.func_out
    
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

