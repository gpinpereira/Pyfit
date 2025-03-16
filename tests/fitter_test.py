import unittest
import numpy as np
import ifitpy
from ifitpy import Fitter
print(ifitpy.__file__)


PLACES = 0
class TestFitter(unittest.TestCase):

    def setUp(self):
        """Set up test data."""

        #Linear trend with gaussian noise
        self.m, self.b = 2,20
        self.x_linear = np.arange(10,100,0.01)
        self.y_linear = self.x_linear*self.m+self.b + np.random.normal(0, 2, size = self.x_linear.shape)

        self.g1_mu, self.g1_sig = 50,10
        self.gauss_sample = np.random.normal(self.g1_mu, self.g1_sig, size = 100000)

        self.expo_p0, self.expo_p1 = 10, -0.01
        self.expo_x = np.arange(0,500,0.1)
        self.expo_y = np.exp(self.expo_p0+self.expo_p1*self.expo_x) +np.random.normal(50, 1000, size = self.expo_x.shape)

        self.ngauss_means = [10, 80, 300, 190, 150]
        self.ngauss_stds = [15, 10, 10, 50, 5]
        sizes = [100000, 100000, 100000, 100000, 80000]

        #join various gaussia distributions
        self.ngauss_sample = np.concatenate([np.random.normal(m, s, size) for m, s, size in zip(self.ngauss_means, self.ngauss_stds, sizes)])

    def test_linear_fit(self):
        """Test fitting a linear function."""
        f = Fitter("linear")
        f.fit(self.x_linear, self.y_linear)
        p = f.getParams()
        
        self.assertAlmostEqual(p.m, self.m, places=PLACES)  # Check slope
        self.assertAlmostEqual(p.b, self.b, places=PLACES)  # Check intercept

    def test_binned_linear_fit(self):
        """Test fitting a linear function."""
        f = Fitter("linear")
        f.fitBinned(self.x_linear, self.y_linear)
        p = f.getParams()
        
        self.assertAlmostEqual(p.m, self.m, places=PLACES)  # Check slope
        self.assertAlmostEqual(p.b, self.b, places=PLACES)  # Check intercept

    def test_exponential_fit(self):
        """Test fitting an exponential function."""
        f = Fitter("expo")
        f.fit(self.expo_x, self.expo_y)
        
        p = f.getParams()

        # No direct parameter checks, but ensure fit succeeds
        self.assertAlmostEqual(p.p0, self.expo_p0+10, places=PLACES)  # Check slope
        self.assertAlmostEqual(p.p1, self.expo_p1, places=2)  # Check intercept


    def test_gaussian_fit(self):
        """Test fitting a Gaussian function."""

        hist, bins = np.histogram(self.gauss_sample, bins=100)
        binscenter = (bins[:-1]+bins[1:])*0.5

        f = Fitter("gaussian")
        f.fit(binscenter, hist, n=1)  # Initial guess
        
        p = f.getParams()

        self.assertAlmostEqual(p.mean, self.g1_mu, places=PLACES)  # Peak center
        self.assertAlmostEqual(p.sigma, self.g1_sig, places=PLACES)  # Width

    def test_binned_gaussian_fit(self):
        """Test fitting a Gaussian function."""

        hist, bins = np.histogram(self.gauss_sample, bins=100)
        binscenter = (bins[:-1]+bins[1:])*0.5

        f = Fitter("gaussian")
        f.fitBinned(self.gauss_sample, n=1, bins=100)  # Initial guess
        
        p = f.getParams()

        self.assertAlmostEqual(p.mean, self.g1_mu, places=PLACES)  # Peak center
        self.assertAlmostEqual(p.sigma, self.g1_sig, places=PLACES)  # Widt

    def test_ngaussian_fit(self):
        """Test fitting a Gaussian function."""

        hist, bins = np.histogram(self.ngauss_sample, bins=100)
        binscenter = (bins[:-1]+bins[1:])*0.5
        ngauss = len(self.ngauss_means)
        f = Fitter("gaussian")
        f.fit(binscenter, hist, n=ngauss)  # Initial guess
        
        p = f.getParams()

        # get means by spliting vars with each row representing the paramters of a gaussian 
        cmeans = np.array(np.array_split(p.vars, ngauss))[:,1]

        np.testing.assert_almost_equal(np.sort(cmeans), np.sort(self.ngauss_means), decimal=0)
        #self.assertAlmostEqual(p.mean, self.g1_mu, places=PLACES)  # Peak center
        #self.assertAlmostEqual(p.sigma, self.g1_sig, places=PLACES)  # Width



if __name__ == "__main__":
    unittest.main()
