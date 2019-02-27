# Written by Mike Smith michaeltsmith.org.uk

from __future__ import division
import numpy as np
from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
import math

class Multidimensional_Integral_Limits(Kern): #todo do I need to inherit from Stationary
    """
    Fariba...
    Integral kernel, can include limits on each integral value. This kernel allows an n-dimensional
    histogram or binned data to be modelled. The outputs are the counts in each bin. The inputs
    are the start and end points of each bin: Pairs of inputs act as the limits on each bin. So
    inputs 4 and 5 provide the start and end values of each bin in the 3rd dimension.
    The kernel's predictions are the latent function which might have generated those binned results.    
    """

    def __init__(self, input_dim, variances=None, lengthscale=None, ARD=False, active_dims=None, name='integral'):
        super(Multidimensional_Integral_Limits, self).__init__(input_dim, active_dims, name)

        if lengthscale is None:
            lengthscale = np.ones(1)
        else:
            lengthscale = np.asarray(lengthscale)

        self.lengthscale = Param('lengthscale', lengthscale, Logexp()) #Logexp - transforms to allow positive only values...
        self.variances = Param('variances', variances, Logexp()) #and here.
        self.link_parameters(self.variances, self.lengthscale) #this just takes a list of parameters we need to optimise.

    #useful little function to help calculate the covariances.
    def g(self,z):
        return 1.0 * z * np.sqrt(math.pi) * math.erf(z) + np.exp(-(z**2))

    def k_ff(self,t,tprime,s,sprime,l):
        "This is k_ff"
        """Covariance between observed values.

        s and t are one domain of the integral (i.e. the integral between s and t)
        sprime and tprime are another domain of the integral (i.e. the integral between sprime and tprime)

        We're interested in how correlated these two integrals are.

        Note: We've not multiplied by the variance, this is done in K."""
        return (l**2) * (self.g((t-sprime) / (np.sqrt(2)*l)) + self.g((tprime-s) / (np.sqrt(2)*l)) - self.g((t - tprime) / (np.sqrt(2)*l)) - self.g((s-sprime) / (np.sqrt(2)*l)))

    def calc_K_ff_no_variance(self,X, X2):
        """Calculates K_xx without the variance term"""
        K_ff = np.ones([X.shape[0], X2.shape[0]]) #ones now as a product occurs over each dimension
        for i,x in enumerate(X):
            for j,x2 in enumerate(X2):
                for il,l in enumerate(self.lengthscale):
                    idx = il*2 #each pair of input dimensions describe the limits on one actual dimension in the data
                    K_ff[i,j] *= self.k_xx(x[idx], x2[idx], x[idx+1], x2[idx+1], l)
        return K_ff

    def K(self, X, X2=None):
        """Note: We have a latent function and an output function. We want to be able to find:
          - the covariance between values of the output function
          - the covariance between values of the latent function
          - the "cross covariance" between values of the output function and the latent function
        This method is used by GPy to either get the covariance between the outputs (K_xx) or
        is used to get the cross covariance (between the latent function and the outputs (K_xf).
        We take advantage of the places where this function is used:
         - if X2 is none, then we know that the items being compared (to get the covariance for)
         are going to be both from the OUTPUT FUNCTION.
         - if X2 is not none, then we know that the items being compared are from two different
         sets (the OUTPUT FUNCTION and the LATENT FUNCTION).

        If we want the covariance between values of the LATENT FUNCTION, we take advantage of
        the fact that we only need that when we do prediction, and this only calls Kdiag (not K).
        So the covariance between LATENT FUNCTIONS is available from Kdiag.
        """
        if X2 is None: 
            X2 = X.copy()
        K_ff = self.calc_K_xx_wo_variance(X, X2)
        return K_ff * self.variances[0]

    def Kdiag(self, X):
        # TODO: Not sure if it is correct or not!!!!
        K_ff = np.ones(X.shape[0])
        for i,x in enumerate(X):
            for il,l in enumerate(self.lengthscale):
                idx = il*2
                K_ff[i] *= self.k_ff(x[idx], x[idx], x[idx+1], x[idx+1], l)
        return K_ff * self.variances[0]

    def h(self, z):
        return 0.5 * z * np.sqrt(math.pi) * math.erf(z) + np.exp(-(z**2))

    def dk_dl(self, t, tprime, s, sprime, l): #derivative of the kernel wrt lengthscale
        return 2 * l * ( self.h((t-sprime) / (np.sqrt(2)*l)) - self.h((t - tprime) / (np.sqrt(2)*l)) + self.h((tprime-s) / (np.sqrt(2)*l)) - self.h((s-sprime) / (np.sqrt(2)*l)))

    def update_gradients_full(self, dL_dK, X, X2=None):
        # K should be NxN
        if X2 is None:  #we're finding dK_xx/dTheta
            X2 = X.copy()
        dKff_dl_term = np.zeros([X.shape[0], X2.shape[0], self.lengthscale.shape[0]])
        k_term = np.zeros([X.shape[0], X2.shape[0], self.lengthscale.shape[0]])
        dKff_dl = np.zeros([X.shape[0], X2.shape[0], self.lengthscale.shape[0]])
        dKff_dv = np.zeros([X.shape[0], X2.shape[0]])
        for il,l in enumerate(self.lengthscale):
            idx = il*2
            for i,x in enumerate(X):
                for j,x2 in enumerate(X2):
                    dKff_dl_term[i, j, il] = self.dkff_dl(x[idx], x2[idx], x[idx+1], x2[idx+1], l)
                    k_term[i, j, il] = self.k_ff(x[idx], x2[idx], x[idx+1], x2[idx+1], l)
        for il,l in enumerate(self.lengthscale):
            dKff_dl = self.variances[0] * dKff_dl_term[:, :, il]
            for jl, l in enumerate(self.lengthscale):
                if jl != il:
                    dKff_dl *= k_term[:, :, jl]
                                          # It was dK_dl * dL_dK before!!!
            self.lengthscale.gradient[il] = np.sum(dL_dK * dKff_dl)
        dK_dv = self.calc_K_ff_no_variance(X) #the gradient wrt the variance is k_xx.
        # self.variances.gradient = np.sum(dK_dv * dL_dK)
        self.variances.gradient = np.sum(dL_dK * dK_dv)
