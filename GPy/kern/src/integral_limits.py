# Written by Mike Smith michaeltsmith.org.uk

from __future__ import division
import math
import numpy as np
from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
from ...util.linalg import tdot
from ... import util

class Integral_Limits(Kern):
# class Integral_Limits(Stationary):
    """
    Integral kernel. This kernel allows 1d histogram or binned data to be modelled.
    The outputs are the counts in each bin. The inputs (on two dimensions) are the start and end points of each bin.
    The kernel's predictions are the latent function which might have generated those binned results.
    """

    def __init__(self, input_dim, variances=None, lengthscale=None, ARD=False, active_dims=None, name='integral'):
        """
        """
        super(Integral_Limits, self).__init__(input_dim, active_dims, name)

        if lengthscale is None:
            lengthscale = np.ones(1)
        else:
            lengthscale = np.asarray(lengthscale)

        self.lengthscale = Param('lengthscale', lengthscale, Logexp()) #Logexp - transforms to allow positive only values...
        self.variances = Param('variances', variances, Logexp()) #and here.
        self.link_parameters(self.variances, self.lengthscale) #this just takes a list of parameters we need to optimise.
        self.ARD = ARD


   #useful little function to help calculate the covariances.
    def g(self,z):
        return 1.0 * z * np.sqrt(math.pi) * math.erf(z) + np.exp(-(z**2))

    def k_xx(self,t,tprime,s,sprime,l):
        "This is k_ff"
        """Covariance between observed values.

        s and t are one domain of the integral (i.e. the integral between s and t)
        sprime and tprime are another domain of the integral (i.e. the integral between sprime and tprime)

        We're interested in how correlated these two integrals are.

        Note: We've not multiplied by the variance, this is done in K."""
        return 0.5 * (l**2) * ( self.g((t-sprime)/l) + self.g((tprime-s)/l) - self.g((t - tprime)/l) - self.g((s-sprime)/l))

    def k_ff(self,t,tprime,s,sprime,l):
        "This is k_ff"
        """Covariance between observed values.

        s and t are one domain of the integral (i.e. the integral between s and t)
        sprime and tprime are another domain of the integral (i.e. the integral between sprime and tprime)

        We're interested in how correlated these two integrals are.

        Note: We've not multiplied by the variance, this is done in K."""
        return (l**2) * (self.g((t-sprime) / (np.sqrt(2)*l)) + self.g((tprime-s) / (np.sqrt(2)*l)) - self.g((t - tprime) / (np.sqrt(2)*l)) - self.g((s-sprime) / (np.sqrt(2)*l)))

    def k_fu_scaled(self,t,tprime,s,l):
        "This is scaled version of k_fu"
        """Covariance between the gradient (latent value) and the actual (observed) value.

        Note that sprime isn't actually used in this expression, presumably because the 'primes' are the gradient (latent) values which don't
        involve an integration, and thus there is no domain over which they're integrated, just a single value that we want."""
        return 0.5 * np.sqrt(math.pi) * l * (math.erf((t-tprime) / l) + math.erf((tprime-s) / l))
    
    def k_fu(self,t,tprime,s,l):
        "This is k_fu which is unscaled"
        """Covariance between the gradient (latent value) and the actual (observed) value.

        Note that sprime isn't actually used in this expression, presumably because the 'primes' are the gradient (latent) values which don't
        involve an integration, and thus there is no domain over which they're integrated, just a single value that we want."""
        return 0.5 * np.sqrt(2 * math.pi) * l * (math.erf((t-tprime) / (np.sqrt(2)*l)) + math.erf((tprime-s) / (np.sqrt(2)*l)))

    def k_uu_scaled(self,t,tprime,l):
        "This is k_uu"
        """Doesn't need s or sprime as we're looking at the 'derivatives', so no domains over which to integrate are required"""
        # print("shape of k_uu:", np.exp(-((t-tprime)**2)/(l**2)).shape)
        # return np.exp(-((t-tprime)**2) / 2* (l**2)) #original rbf
        return np.exp(-((t-tprime)**2) / (l**2)) #scaled rbf

    def k_uu(self,t,tprime,l):
        "This is k_uu"
        """Doesn't need s or sprime as we're looking at the 'derivatives', so no domains over which to integrate are required"""
        return np.exp(-((t-tprime)**2) / 2 * (l**2)) #original rbf
    
    def Kuu_scaled_by_frb(self, X, X2=None):
        """
        This results exactly like Kuu, the difference is in Kuu we use RBF kernel directly and deviding
        it by exp(-0.5 * C). C is -((x - x')**2)/ l**2.
        This is calculating RBF kernel but withought 1/2 part.
        It is been tested against RBF kernel and when muliplying by
        np.exp(-0.5 * -((x - x')**2)/ l**2 ) the result is exactly the same as RBF kernel.
        """
        # print ("Z.shape k_uu in fact, should fix this!!!:", X.shape)
        if X2 is None:
            X2 = X.copy()
        K_uu = np.zeros((X.shape[0], X2.shape[0]))
        for i,x in enumerate(X):
            for j, x2 in enumerate(X2):
                x_mid = (x[0] + x[1]) / 2
                x_mid_2 = (x2[0] + x2[1]) / 2
                K_uu[i,j] = self.k_uu_scaled(x_mid, x_mid_2, self.lengthscale[0])
        return K_uu * self.variances[0]

    def Kuu_scaled_rbf(self, X, X2=None):
        """
        Kernel function applied on inputs X and X2.
        In the stationary case there is an inner function depending on the
        distances from X to X2, called r.

        K(X, X2) = K_of_r((X-X2)**2)
        """

        # adjust by e(-0.5 * c)
        if X2 is None:
            X2_copy = X.copy()
        else:
            X2_copy = X2.copy()

        rbf_scale_k = np.empty((X.shape[0], X2_copy.shape[0]))
        for i,x in enumerate(X):
            for j, x2 in enumerate(X2_copy):
                rbf_scale_k[i,j] = np.exp(-0.5 * -((x - x2)**2) / (self.lengthscale[0] ** 2))
        r = self._scaled_dist(X, X2)
        return self.K_of_r(r) / rbf_scale_k
    
    def Kuu(self, X, X2=None):
        """
        This function using original RBF formulation, When giving Zs, if input dimension is 
        two dimensional mid points should be given. For example if
        Z = 1.0 * np.array([[2,1], [3,2], [4,3], [5,4], [6,5], [7,6]])
        it should be chnaged to:
        if Z_rbf = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])[:,None] 
        """
        r = self._scaled_dist(X, X2)
        return self.K_of_r(r)


    def K_scaled(self, X, X2=None):
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
        # print("X.shape", X.shape)
        # print("this is Kxx, why it comes here!!!!!")
        if X2 is None:
            "This is actually k_ff"
            K_xx = np.zeros([X.shape[0],X.shape[0]])
            for i,x in enumerate(X):
                for j,x2 in enumerate(X):
                    K_xx[i,j] = self.k_xx(x[0],x2[0],x[1],x2[1],self.lengthscale[0])
            # print ('K_xx.shape')
            return K_xx * self.variances[0]
        else:
            "This is k_fu"
            K_fu = np.zeros([X.shape[0],X2.shape[0]])
            for i,x in enumerate(X):
                for j,x2 in enumerate(X2):
                    K_fu[i,j] = self.k_fu_scaled(x[0],x2[0],x[1],self.lengthscale[0]) #x2[1] unused, see k_xf docstring for explanation.
            return K_fu * self.variances[0]

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
        # print("X.shape", X.shape)
        # print("this is Kxx, why it comes here!!!!!")
        if X2 is None:
            "This is actually k_ff"
            K_ff = np.zeros([X.shape[0],X.shape[0]])
            for i,x in enumerate(X):
                for j,x2 in enumerate(X):
                    K_ff[i,j] = self.k_ff(x[0],x2[0],x[1],x2[1],self.lengthscale[0])
            # print ('K_xx.shape')
            return K_ff * self.variances[0]
        else:
            "This is k_fu"
            K_fu = np.zeros([X.shape[0], X2.shape[0]])
            for i,x in enumerate(X):
                for j,x2 in enumerate(X2):
                    K_fu[i,j] = self.k_fu(x[0],x2[0],x[1], self.lengthscale[0]) #x2[1] unused, see k_xf docstring for explanation.
            return K_fu * self.variances[0]

    def Kdiag(self, X):
        K_ff = np.zeros((X.shape[0]))
        for i, x in enumerate(X):
            K_ff[i] = self.k_ff(x[0], x[0], x[1], x[1], self.lengthscale[0])
        return K_ff * self.variances[0]

    def Kdiag_Kuu(self, X):
        return self.variance[0]*np.ones(X.shape[0])
    
    # def Kdiag_Kfu(self, X, X2):
    """
    Do we need this?
    Also is the size of X1 and X2 the same? Because we need the diagnol and it should be square matrix.
    """
    #     K_fu = np.zeros([X.shape[0], X2.shape[0]])
    #     for i, x in enumerate(X):
    #         for j, x2 in enumerate(X2):
    #             K_fu[i,j] = self.k_fu(x[0], x2[0], x[1], self.lengthscale[0]) #x2[1] unused, see k_xf docstring for explanation.
    #     return K_fu * self.variances[0]

    """
    Derivatives!
    """

    def h_scaled(self, z):
        return 0.5 * z * np.sqrt(math.pi) * math.erf(z) + np.exp(-(z**2))

    def dkff_dl_scaled(self, t, tprime, s, sprime, l): #derivative of the kernel wrt lengthscale
        return l * (self.h((t-sprime)/l) - self.h((t - tprime)/l) + self.h((tprime-s)/l) - self.h((s-sprime)/l))

    def h(self, z):
        "Should be fixed!!!"
        return 0.5 * z * np.sqrt(math.pi) * math.erf(z) + 2 * np.exp(-(z**2))

    def dkff_dl(self, t, tprime, s, sprime, l): #derivative of the kernel wrt lengthscale
        "Should be written!!!"
        return l * (self.h((t-sprime)/l) - self.h((t - tprime)/l) + self.h((tprime-s)/l) - self.h((s-sprime)/l))

    def dkuu_dl(self, t, tprime, l):
        return (2/l) * (((t - tprime) ** 2) / l ** 2) * np.exp(-((((t - tprime) ** 2) / l ** 2)))
        # TODO: Not sure how to update this inside update_gradients_full
# ///////////////////////////////////////start Fariba//////////////////////////////////////////////
    def hp(self, z):
        return (np.sqrt(math.pi) / 2) * math.erf(z) - (z * np.exp(-(z**2)))

    def dkfu_dl_scaled(self, t, tprime, s, l): #derivative of the kfu wrt lengthscale
        return (self.hp((t - tprime) / l) + self.hp((tprime - s) / l))

    def dkfu_dl(self, t, tprime, s, l): #derivative of the kfu wrt lengthscale
        return (np.sqrt(2) * (self.hp((t - tprime) / np.sqrt(2)*l) + self.hp((tprime - s) / np.sqrt(2)*l)))
# ///////////////////////////end Fariba/////////////////////////////////////////////////////////

# ///////////////////////////////////////start Fariba//////////////////////////////////////////////
    def update_gradients_full(self, dL_dK, X, X2=None):
        # K should be NxN
        if X2 is None:  #we're finding dK_xx/dTheta
            dKff_dl = np.zeros([X.shape[0],X.shape[0]])
            dKff_dv = np.zeros([X.shape[0],X.shape[0]])
            for i,x in enumerate(X):
                for j,x2 in enumerate(X):
                    dKff_dl[i,j] = self.variances[0]*self.dkff_dl(x[0],x2[0],x[1],x2[1],self.lengthscale[0])
                    dKff_dv[i,j] = self.k_ff(x[0],x2[0],x[1],x2[1],self.lengthscale[0])  #the gradient wrt the variance is k_xx.
            self.lengthscale.gradient = np.sum(dKff_dl * dL_dK)
            # print ("dKff_dl shape:", dKff_dl.shape)
            # return dKff_dl
            self.variances.gradient = np.sum(dKff_dv * dL_dK)
        else:     #we're finding dK_xf/Dtheta
            # import pdb; pdb.set_trace()
            # raise NotImplementedError("Currently this function only handles finding the gradient of a single vector of inputs (X) not a pair of vectors (X and X2)")
            # Does this make sense???? Ask Mauricio!
            # K should be NxM
            dKfu_dl = np.zeros([X.shape[0],X2.shape[0]])
            dKfu_dv = np.zeros([X.shape[0],X2.shape[0]])
            for i,x in enumerate(X):
                for j,x2 in enumerate(X2):
                    dKfu_dl[i,j] = self.variances[0] * self.dkfu_dl(x[0], x2[0], x[1], self.lengthscale[0])
                    dKfu_dv[i,j] = self.k_fu_scaled(x[0], x2[0], x[1], self.lengthscale[0])  #the gradient wrt the variance is k_xx.
            self.lengthscale.gradient = np.sum(dL_dK * dKfu_dl)
            self.variances.gradient = np.sum(dL_dK * dKfu_dv)
# ///////////////////////////end Fariba/////////////////////////////////////////////////////////

 # --------------------------------------------------------------------------
    def K_of_r(self, r):
        return self.variances * np.exp(-0.5 * r**2)

    def dK_dr(self, r):
        return -r * self.K_of_r(r)

    def dK_dr_via_X(self, X, X2):
        """
        compute the derivative of K wrt X going through X
        """
        #a convenience function, so we can cache dK_dr
        return self.dK_dr(self._scaled_dist(X, X2))

    def _unscaled_dist(self, X, X2=None):
        """
        Compute the Euclidean distance between each row of X and X2, or between
        each pair of rows of X if X2 is None.
        """
        #X, = self._slice_X(X)
        if X2 is None:
            Xsq = np.sum(np.square(X),1)
            r2 = -2.*tdot(X) + (Xsq[:,None] + Xsq[None,:])
            util.diag.view(r2)[:,]= 0. # force diagnoal to be zero: sometime numerically a little negative
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)
        else:
            #X2, = self._slice_X(X2)
            X1sq = np.sum(np.square(X),1)
            X2sq = np.sum(np.square(X2),1)
            r2 = -2.*np.dot(X, X2.T) + (X1sq[:,None] + X2sq[None,:])
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)

    def _scaled_dist(self, X, X2=None):
        """
        Efficiently compute the scaled distance, r.

        ..math::
            r = \sqrt( \sum_{q=1}^Q (x_q - x'q)^2/l_q^2 )

        Note that if thre is only one lengthscale, l comes outside the sum. In
        this case we compute the unscaled distance first (in a separate
        function for caching) and divide by lengthscale afterwards

        """
        if self.ARD:
            if X2 is not None:
                X2 = X2 / self.lengthscale
            return self._unscaled_dist(X/self.lengthscale, X2)
        else:
            return self._unscaled_dist(X, X2)/self.lengthscale


# -----------------------------------------------------------------
    """
    Not used yet!
    """
    def _inv_dist(self, X, X2=None):
        """
        Compute the elementwise inverse of the distance matrix, expecpt on the
        diagonal, where we return zero (the distance on the diagonal is zero).
        This term appears in derviatives.
        """
        dist = self._scaled_dist(X, X2).copy()
        return 1./np.where(dist != 0., dist, np.inf)

    def _gradients_X_pure(self, dL_dK, X, X2=None):
        invdist = self._inv_dist(X, X2)
        dL_dr = self.dK_dr_via_X(X, X2) * dL_dK
        tmp = invdist*dL_dr
        if X2 is None:
            tmp = tmp + tmp.T
            X2 = X

        #The high-memory numpy way:
        #d =  X[:, None, :] - X2[None, :, :]
        #grad = np.sum(tmp[:,:,None]*d,1)/self.lengthscale**2

        #the lower memory way with a loop
        grad = np.empty(X.shape, dtype=np.float64)
        for q in range(self.input_dim):
            np.sum(tmp*(X[:,q][:,None]-X2[:,q][None,:]), axis=1, out=grad[:,q])
        return grad/self.lengthscale**2

    def gradients_X(self, dL_dK, X, X2):
        """
        .. math::

            \\frac{\partial L}{\partial X} = \\frac{\partial L}{\partial K}\\frac{\partial K}{\partial X}
        """
        # raise NotImplementedError
        return self._gradients_X_pure(dL_dK, X, X2)

        # return X


# ///////////////////////////////////////start Fariba//////////////////////////////////////////////
    # def frb(self, l):
    #     from functools import partial
    #     from GPy.models import GradientChecker
    #     f = partial(self.K)
    #     df = partial(self.update_gradients_full)
    #     grad = GradientChecker(f, df, l, 'l')
    #     grad.checkgrad(verbose=1)
# ///////////////////////////end Fariba/////////////////////////////////////////////////////////
