# Written by Mike Smith michaeltsmith.org.uk

from __future__ import division
import numpy as np
from GPy.kern import Kern
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
import math

class Mix_Integral(Kern):
    """
      
    """

    def __init__(self, input_dim, variances=None, lengthscale=None, ARD=False, active_dims=None, name='mix_integral'):
        super(Mix_Integral, self).__init__(input_dim, active_dims, name)

        if lengthscale is None:
            lengthscale = np.ones(1)
        else:
            lengthscale = np.asarray(lengthscale)
            
        assert len(lengthscale)==(input_dim-1)/2

        self.lengthscale = Param('lengthscale', lengthscale, Logexp()) #Logexp - transforms to allow positive only values...
        self.variances = Param('variances', variances, Logexp()) #and here.
        self.link_parameters(self.variances, self.lengthscale) #this just takes a list of parameters we need to optimise.

    def h(self, z):
        return 0.5 * z * np.sqrt(math.pi) * math.erf(z) + np.exp(-(z**2))

    def d(self, z):
        return 0.5 * np.sqrt(math.pi) * math.erf(z) - z * np.exp(-(z**2))

    def dk_dl(self, t_type, tprime_type, t, tprime, s, sprime, l): #derivative of the kernel wrt lengthscale
        #t and tprime are the two start locations
        #s and sprime are the two end locations
        #if t_type is 0 then t and s should be in the equation
        #if tprime_type is 0 then tprime and sprime should be in the equation.
        
        if (t_type==0) and (tprime_type==0): #both integrals
            return l * ( self.h((t-sprime)/l) - self.h((t - tprime)/l) + self.h((tprime-s)/l) - self.h((s-sprime)/l))
        if (t_type==0) and (tprime_type==1): #integral vs latent 
            return self.d((t-tprime)/l) + self.d((tprime-s)/l)
        if (t_type==1) and (tprime_type==0): #integral vs latent 
            return self.d((tprime-t)/l) + self.d((t-sprime)/l)
            #swap: t<->tprime (t-s)->(tprime-sprime)
        if (t_type==1) and (tprime_type==1): #both latent observations            
            return 2*(t-tprime)**2/(l**3) * np.exp(-((t-tprime)/l)**2)
        assert False, "Invalid choice of latent/integral parameter (set the last column of X to 0s and 1s to select this)"
            

    def update_gradients_full(self, dL_dK, X, X2=None): 
        if X2 is None:  #we're finding dK_xx/dTheta
            dK_dl_term = np.zeros([X.shape[0],X.shape[0],self.lengthscale.shape[0]])
            k_term = np.zeros([X.shape[0],X.shape[0],self.lengthscale.shape[0]])
            dK_dl = np.zeros([X.shape[0],X.shape[0],self.lengthscale.shape[0]])
            dK_dv = np.zeros([X.shape[0],X.shape[0]])
            for il,l in enumerate(self.lengthscale):
                idx = il*2
                for i,x in enumerate(X):
                    for j,x2 in enumerate(X):
                        dK_dl_term[i,j,il] = self.dk_dl(x[-1],x2[-1],x[idx],x2[idx],x[idx+1],x2[idx+1],l)
                        k_term[i,j,il] = self.k(x,x2,idx,l)
            for il,l in enumerate(self.lengthscale):
                dK_dl = self.variances[0] * dK_dl_term[:,:,il]
                for jl, l in enumerate(self.lengthscale): ##@FARIBA Why do I have to comment this out??
                    if jl!=il:
                        dK_dl *= k_term[:,:,jl]
                self.lengthscale.gradient[il] = np.sum(dK_dl * dL_dK)
            dK_dv = self.calc_K_xx_wo_variance(X,X) #the gradient wrt the variance is k_xx.
            self.variances.gradient = np.sum(dK_dv * dL_dK)
        else:     #we're finding dK_xf/Dtheta
            raise NotImplementedError("Currently this function only handles finding the gradient of a single vector of inputs (X) not a pair of vectors (X and X2)")


    #useful little function to help calculate the covariances.
    def g(self,z):
        return 1.0 * z * np.sqrt(math.pi) * math.erf(z) + np.exp(-(z**2))

    def k_xx(self,t,tprime,s,sprime,lengthscale):
        """Covariance between observed values.

        s and t are one domain of the integral (i.e. the integral between s and t)
        sprime and tprime are another domain of the integral (i.e. the integral between sprime and tprime)

        We're interested in how correlated these two integrals are.

        Note: We've not multiplied by the variance, this is done in K."""
     #######   l = lengthscale * np.sqrt(2)###TO REINSTATE
        l = lengthscale
        return 0.5 * (l**2) * ( self.g((t-sprime)/l) + self.g((tprime-s)/l) - self.g((t - tprime)/l) - self.g((s-sprime)/l))

    def k_ff(self,t,tprime,lengthscale):
        """Doesn't need s or sprime as we're looking at the 'derivatives', so no domains over which to integrate are required"""        
     #######   l = lengthscale * np.sqrt(2)###TO REINSTATE
        l = lengthscale
        return np.exp(-((t-tprime)**2)/(l**2)) #rbf

    def k_xf(self,t,tprime,s,lengthscale):
        """Covariance between the gradient (latent value) and the actual (observed) value.

        Note that sprime isn't actually used in this expression, presumably because the 'primes' are the gradient (latent) values which don't
        involve an integration, and thus there is no domain over which they're integrated, just a single value that we want."""
     #######   l = lengthscale * np.sqrt(2)###TO REINSTATE
        l = lengthscale
        return 0.5 * np.sqrt(math.pi) * l * (math.erf((t-tprime)/l) + math.erf((tprime-s)/l))

    def k(self,x,x2,idx,l):
        """Helper function to compute covariance in one dimension (idx) between a pair of points.
        The last element in x and x2 specify if these are integrals (0) or latent values (1).
        l = that dimension's lengthscale
        """
        
        if (x[-1]==0) and (x2[-1]==0):
            return self.k_xx(x[idx],x2[idx],x[idx+1],x2[idx+1],l)
        if (x[-1] == 0) and (x2[-1] == 1):
            return self.k_xf(x[idx],x2[idx],x[idx+1],l)
        if (x[-1] == 1) and (x2[-1] == 0):
            return self.k_xf(x2[idx],x[idx],x2[idx+1],l)                        
        if (x[-1]==1) and (x2[-1]==1):
            return self.k_ff(x[idx],x2[idx],l)
        assert False, "Invalid choice of latent/integral parameter (set the last column of X to 0s and 1s to select this)"

    def calc_K_xx_wo_variance(self,X,X2):
        """Calculates K_xx without the variance term"""
        K_xx = np.ones([X.shape[0],X2.shape[0]]) #ones now as a product occurs over each dimension
        for i,x in enumerate(X):
            for j,x2 in enumerate(X2):
                for il,l in enumerate(self.lengthscale):
                    idx = il*2 #each pair of input dimensions describe the limits on one actual dimension in the data
                    K_xx[i,j] *= self.k(x,x2,idx,l)
        return K_xx

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        K_xx = self.calc_K_xx_wo_variance(X,X2)
        return K_xx * self.variances[0]

    def Kdiag(self, X):
        return np.diag(self.K(X))