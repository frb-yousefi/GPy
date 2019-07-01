# Written by Mike Smith and modified and extended by Fariba.

from __future__ import division
import numpy as np
from GPy.kern import Kern
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
import math
from numba import jit

# If the kernel is stationary, inherit from Stationary in
# GPy.kern.src.stationary.py If the kernel is non-stationary,
# inherit from Kern in GPy.kern.src.kern.py
class Mix_Integral_extend(Kern):
    """
    Integral kernel. This kernel allows 1d histogram or binned data to be modelled.
    The outputs are the counts in each bin. The inputs (on two dimensions) are the start and end points of each bin.
    The kernel's predictions are the latent function which might have generated those binned results.
    This code is the normalised version of the multidim_integral_kernel.py and it normalises kff and kfu.
    For the kff the normalisation is done by dividing the kff to the bounds. for the derivative of dkff/dl the same is done in dk/dl
    however, for the dk/dvar because the normalisation is done in calc_K_wo_variance so we didn't do anything extra for that part.
    """
 
    def __init__(self, input_dim, variance=None, lengthscale=None, ARD=False, active_dims=None, name='Mix_Integral_extend'):
        super(Mix_Integral_extend, self).__init__(input_dim, active_dims, name)
        if lengthscale is None:
            lengthscale = np.ones(1)
        else:
            lengthscale = np.asarray(lengthscale)
        assert len(lengthscale)==(input_dim-1)/2

        self.lengthscale = Param('lengthscale', lengthscale, Logexp()) #Logexp - transforms to allow positive only values...
        self.variance = Param('variance', variance, Logexp()) #and here.
        self.link_parameters(self.variance, self.lengthscale) #this just takes a list of parameters we need to optimise.
        self.ARD = ARD

    #useful little function to help calculate the covariances.
    def g(self, z):
        # print("z", z)
        return 1.0 * z * np.sqrt(math.pi) * math.erf(z) + np.exp(-(z**2))

    def k_ff(self, t, tprime, s, sprime, lengthscale):
        """Covariance between observed values.

        s and t are one domain of the integral (i.e. the integral between s and t)
        sprime and tprime are another domain of the integral (i.e. the integral between sprime and tprime)

        We're interested in how correlated these two integrals are.

        Note: We've not multiplied by the variance, this is done in K."""
        #######   l = lengthscale * np.sqrt(2)###TO REINSTATE
        l = lengthscale
        area_small = (np.absolute(s - t) * np.absolute(sprime - tprime))
        # print("The result of kff:", (0.5 * (l ** 2) * ( self.g((t - sprime) / l) + self.g((tprime - s) / l) - self.g((t - tprime) / l) - self.g((s - sprime) / l))))
        # print("the result after division", (0.5 * (l ** 2) * ( self.g((t - sprime) / l) + self.g((tprime - s) / l) - self.g((t - tprime) / l) - self.g((s - sprime) / l)))  / area_small)
        return (0.5 * (l ** 2) * ( self.g((t - sprime) / l) + self.g((tprime - s) / l) - self.g((t - tprime) / l) - self.g((s - sprime) / l)))  / area_small
    
    # def area_kff(self, t, tprime, s, sprime):
    #     area_small = 0
    #     area_small += (np.absolute(s - t) * np.absolute(sprime - tprime))
    #     return area_small

    def calc_K_wo_variance(self, X, X2):
        """Calculates K without the variance term, it can be Kff, Kfu or Kuu based on the last dimension of the input"""
        # K_ = np.ones([X.shape[0], X2.shape[0]]) #ones now as a product occurs over each dimension
        # for i, x in enumerate(X):
        #     for j, x2 in enumerate(X2):
        #         for il,l in enumerate(self.lengthscale):
        #             idx = il * 2 #each pair of input dimensions describe the limits on one actual dimension in the data
        #             K_[i,j] *= self.k(x, x2, idx, l)
        # return K_
        return frb_calc_K_wo_variance(X, X2, np.array(self.lengthscale))

    def k_uu(self, t, tprime, lengthscale):
        """Doesn't need s or sprime as we're looking at the 'derivatives', so no domains over which to integrate are required"""        
        #######   l = lengthscale * np.sqrt(2)###TO REINSTATE
        l = lengthscale
        return np.exp(-((t - tprime) ** 2) / (l ** 2)) # scaled rbf
        # return np.exp(-0.5 * ((t - tprime) ** 2) / (l ** 2)) #rbf

    def k_fu(self, t, tprime, s, lengthscale):
        """Covariance between the gradient (latent value) and the actual (observed) value.

        Note that sprime isn't actually used in this expression, presumably because the 'primes' are the gradient (latent) values which don't
        involve an integration, and thus there is no domain over which they're integrated, just a single value that we want."""
        #######   l = lengthscale * np.sqrt(2)###TO REINSTATE
        l = lengthscale
        return (0.5 * np.sqrt(math.pi) * l * (math.erf((t - tprime) / l) + math.erf((tprime - s) / l))) / (np.absolute(s-t))

    def k(self, x, x2, idx, l):
        """Helper function to compute covariance in one dimension (idx) between a pair of points.
        The last element in x and x2 specify if these are integrals (0) or latent values (1).
        l = that dimension's lengthscale
        """
        x_type = int(x[-1])
        x2_type = int(x2[-1])

        if (x_type == 0) and (x2_type == 0):
            return self.k_ff(x[idx], x2[idx], x[idx+1], x2[idx+1], l)
        if (x_type == 0) and (x2_type == 1):
            return self.k_fu(x[idx], x2[idx], x[idx+1], l)
        if (x_type == 1) and (x2_type == 0):
            return self.k_fu(x2[idx], x[idx], x2[idx+1], l)                        
        if (x_type == 1) and (x2_type == 1):
            return self.k_uu(x[idx], x2[idx], l)
        
        raise Exception("Invalid choice of latent/integral parameter (set the last column of X to 0s and 1s to select this)")

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        K = self.calc_K_wo_variance(X, X2)
        return K * self.variance[0]

    def Kdiag(self, X):
        return np.diag(self.K(X))

    """
    Maybe we could make Kdiag much more faster, because now every single time it should calculate K and get the diag!!
    # TODO
    """
    # def Kdiag_Kuu(self, X):
    #     return self.variance[0]*np.ones(X.shape[0])

    """
    Derivatives!
    """
    def h(self, z):
        return 0.5 * z * np.sqrt(math.pi) * math.erf(z) + np.exp(-(z**2))

    def hp(self, z):
        return 0.5 * np.sqrt(math.pi) * math.erf(z) - z * np.exp(-(z**2))

    def dk_dl(self, t_type, tprime_type, t, tprime, s, sprime, l): #derivative of the kernel wrt lengthscale
        #t and tprime are the two start locations
        #s and sprime are the two end locations
        #if t_type is 0 then t and s should be in the equation
        #if tprime_type is 0 then tprime and sprime should be in the equation.
        t_type = int(t_type)
        tprime_type = int(tprime_type)
        
        if (t_type == 0) and (tprime_type == 0): #both integrals
            return l * (self.h((t - sprime) / l) - self.h((t - tprime) / l) + self.h((tprime - s) / l) - self.h((s - sprime) / l)) / (np.absolute(s-t) * np.absolute(sprime-tprime))
        if (t_type == 0) and (tprime_type == 1): #integral vs latent 
            return (self.hp((t - tprime) / l) + self.hp((tprime - s) / l)) / (np.absolute(s-t))
        if (t_type == 1) and (tprime_type == 0): #integral vs latent 
            return (self.hp((tprime - t) / l) + self.hp((t - sprime) / l)) / (np.absolute(sprime-tprime))
            #swap: t<->tprime (t-s)->(tprime-sprime)
        if (t_type == 1) and (tprime_type == 1): #both latent observations 
            return 2 * (t - tprime) ** 2 / (l ** 3) * np.exp(-((t - tprime) / l) ** 2)
        
        raise Exception("Invalid choice of latent/integral parameter (set the last column of X to 0s and 1s to select this)")

    def update_gradients_full(self, dL_dK, X, X2=None): 
        if X2 is None:  # we're finding dK_xx/dTheta
            X2 = X
        dK_dl_term = np.zeros([X.shape[0], X2.shape[0], self.lengthscale.shape[0]])
        k_term = np.zeros([X.shape[0], X2.shape[0], self.lengthscale.shape[0]])
        for il, l in enumerate(self.lengthscale):
            idx = il * 2
            for i, x in enumerate(X):
                for j, x2 in enumerate(X2):
                    dK_dl_term[i, j, il] = self.dk_dl(x[-1], x2[-1], x[idx], x2[idx], x[idx+1], x2[idx+1], l)
                    k_term[i, j, il] = self.k(x, x2, idx, l)
        for il,l in enumerate(self.lengthscale):
            dK_dl = self.variance[0] * dK_dl_term[:,:,il]  # Adding constant for the area

            for jl, l in enumerate(self.lengthscale):
                if jl != il:
                    dK_dl *= k_term[:,:,jl]
            self.lengthscale.gradient[il] = np.sum(dL_dK * dK_dl)
        dK_dv = self.calc_K_wo_variance(X,X2) #the gradient wrt the variance is k.
        self.variance.gradient = np.sum(dL_dK * dK_dv)
    
    def update_gradients_diag(self, dL_dKdiag, X):
        """
        Given the derivative of the objective with respect to the diagonal of
        the covariance matrix, compute the derivative wrt the parameters of
        this kernel and store in the <parameter>.gradient field.

        See also update_gradients_full
        """
        dL_dK_full = np.eye(X.shape[0], X.shape[0]) * dL_dKdiag
        self.update_gradients_full(dL_dK_full, X)

    def dk_dz(self, input_type_1, input_type_2, t, tprime, s, sprime, lengthscale):
        # here t stands for z and tprime stands for z' (latent inputs)
        input_type_1 = int(input_type_1)
        input_type_2 = int(input_type_2)
        if (input_type_1 == 0) and (input_type_2 == 1):
            return self.dkfu_dz(t, tprime, s, lengthscale) 
        # TODO: Write the gradient for the below combination--- check if it is correct!!!
        if (input_type_1 == 1) and (input_type_2 == 0):
            return self.dkfu_dz(tprime, t, sprime, lengthscale) 
        if (input_type_1 == 1) and (input_type_2 == 1):
            return self.dkuu_dz(t, tprime, s, sprime, lengthscale) 
        if (input_type_1 == 0) and (input_type_2 == 0):
            #  It is derivative of kff/dz which is zero
            return 0 
        
        raise Exception('KFU should have X and Z and KUU should have Z and Z!')
    
    def dkuu_dz(self, t, tprime, s, sprime, lengthscale):
        l = lengthscale
        return -2 * (t - tprime) / (l ** 2) * np.exp(-(t - tprime) ** 2 / l ** 2)
        # return -(t - tprime) / l * np.exp(-0.5 * (t - tprime) ** 2 / l ** 2)

    def dkfu_dz(self, t, tprime, s, lengthscale):
        l = lengthscale
        return -np.exp(-(t - tprime) ** 2 / l ** 2) + np.exp(-(tprime - s) ** 2 / l ** 2) 
        # Should we change above to this?
        # return -np.exp(-(t - tprime) ** 2 / l ** 2) + np.exp(-(tprime - s) ** 2 / l ** 2) / (np.absolute(s-t))

    
    '''
     ****** when calculating dK_dz, based on how many lengthscale we are having we will have 
     different K for every dimension. So for example if l has 3 dimensions then we have 
     K = K1*K2*K3 now for the dK/dz2 we will have K1*dK2/dz2*K3 
    '''
    def gradients_X(self, dL_dK, X, X2=None):
        #     """
        #     .. math::
        #         \\frac{\partial L}{\partial X} = \\frac{\partial L}{\partial K}\\frac{\partial K}{\partial X}
        #     """
        if X2 is None:
            X2 = X.copy()
        dK_dz_term = np.zeros((X.shape[0], X2.shape[0], self.lengthscale.shape[0]))
        k_term = np.zeros((X.shape[0], X2.shape[0], self.lengthscale.shape[0]))
        for il, l in enumerate(self.lengthscale):
            idx = il * 2
            for i, x in enumerate(X):
                for j, x2 in enumerate(X2):
                    dK_dz_term[i, j, il] = self.dk_dz(input_type_1=x[-1], input_type_2=x2[-1], t=x[idx], tprime=x2[idx], s=x[idx+1], sprime=x2[idx+1], lengthscale=l)
                    assert int(x[-1]) == 1  # we assume the first term (X) is always Z, X2 could be either X or Z
                    k_term[i, j, il] = self.k(x, x2, idx, l)
        # print ('dK_dz_term: \n', dK_dz_term)
        # The result of the derivative should be
        inducing_inputs_gradient = np.ones((X.shape[0], self.lengthscale.shape[0]))
        for il,l in enumerate(self.lengthscale):
            dK_dz = self.variance[0] * dK_dz_term[:,:,il]
            # print('dK_dz:\n', dK_dz)
            for jl, l in enumerate(self.lengthscale): 
                if jl != il:
                    dK_dz *= k_term[:,:,jl]
                    # print('dK_dz inside \n', dK_dz)
            tmp = dL_dK * dK_dz
            inducing_inputs_gradient[:, il][:, None] = np.sum(tmp, axis=1)[:,None]
        return inducing_inputs_gradient
        
    def gradients_X_orig(self, dL_dK, X, X2=None):
        """
        .. math::
            \\frac{\partial L}{\partial X} = \\frac{\partial L}{\partial K}\\frac{\partial K}{\partial X}
        """
        # We don't use this because it expects X2 not to be None!
        if X2 is None:
            dK_dz_term = np.zeros((X.shape[0], X.shape[0], self.lengthscale.shape[0]))
            k_term = np.zeros((X.shape[0], X.shape[0], self.lengthscale.shape[0]))
            for il, l in enumerate(self.lengthscale):
                idx = il * 2
                for i, x in enumerate(X):
                    for j, x2 in enumerate(X):
                        dK_dz_term[i, j, il] = self.dk_dz(x[-1], x2[-1], x[idx], x2[idx], x[idx+1], x2[idx+1], l)
                        k_term[i, j, il] = self.k(x, x2, idx, l)
            # The result of the derivative should be 
            inducing_inputs_gradient = np.ones((X.shape[0], self.lengthscale.shape[0]))
            for il,l in enumerate(self.lengthscale):
                dK_dz = self.variance[0] * dK_dz_term[:,:,il]
            #     # It doesn't work without these three lines but I don't know what is that!!!
                for jl, l in enumerate(self.lengthscale): ##@FARIBA Why do I have to comment this out??
                    if jl != il:
                        dK_dz *= k_term[:,:,jl]
                tmp = dL_dK * dK_dz
                inducing_inputs_gradient[:, il][:, None] = np.sum(tmp, axis=1)[:,None]
                # inducing_inputs_gradient[il] = np.sum(dL_dK * dK_dz)
            return inducing_inputs_gradient
        else:
            raise NotImplementedError('Currently this function only handles finding the gradient of a single vector of inputs (X) not a pair of vectors (X and X2)")')

# ------------------------------------------------------------------------------------------------------------------------------
# MAKING CODE FASTER USING NUMBA
# ------------------------------------------------------------------------------------------------------------------------------

@jit(nopython=True)
def frb_calc_K_wo_variance(X, X2, lengthscale):
    """
    Calculates K without the variance term, it can be Kff, Kfu or Kuu based on the last dimension of the input
    """
    K_ = np.ones((X.shape[0], X2.shape[0])) #ones now as a product occurs over each dimension
    for i in range(X.shape[0]):
        x = X[i]
        for j in range(X2.shape[0]):
            x2 = X2[j]
            for il in range(lengthscale.shape[0]):
                l = lengthscale[il]
                idx = int(il*2) #each pair of input dimensions describe the limits on one actual dimension in the data
                K_[i,j] *= k(x, x2, idx, l)
    return K_

@jit(nopython=True)
def k(x, x2, idx, l):
    x_type = int(x[-1])
    x2_type = int(x2[-1])
    if (x_type == 0) and (x2_type == 0):
        return k_ff(x[idx], x2[idx], x[idx+1], x2[idx+1], l)
    if (x_type == 0) and (x2_type == 1):
        return k_fu(x[idx], x2[idx], x[idx+1], l)
    if (x_type == 1) and (x2_type == 0):
        return k_fu(x2[idx], x[idx], x2[idx+1], l)                        
    if (x_type == 1) and (x2_type == 1):
        return k_uu(x[idx], x2[idx], l)

    raise Exception("Invalid choice of latent/integral parameter (set the last column of X to 0s and 1s to select this)")

@jit(nopython=True)
def k_ff(t, tprime, s, sprime, lengthscale):
    l = lengthscale
    # print("s, t, sp, tp:", s, t, sprime, tprime)
    # print("The result of kff:", (0.5 * (l ** 2) * (g((t - sprime) / l) + g((tprime - s) / l) - g((t - tprime) / l) - g((s - sprime) / l))))
    # print("the result after division", (0.5 * (l ** 2) * (g((t - sprime) / l) + g((tprime - s) / l) - g((t - tprime) / l) - g((s - sprime) / l))) / (np.absolute(s-t) * np.absolute(sprime-tprime)))
    # print("division:", (np.absolute(s-t) * np.absolute(sprime-tprime)))
    return (0.5 * (l ** 2) * (g((t - sprime) / l) + g((tprime - s) / l) - g((t - tprime) / l) - g((s - sprime) / l))) / (np.absolute(s-t) * np.absolute(sprime-tprime))
    # return 0.5 * (g((t - sprime) / l) + g((tprime - s) / l) - g((t - tprime) / l) - g((s - sprime) / l))

@jit(nopython=True)
def k_uu(t, tprime, lengthscale):
    l = lengthscale
    return np.exp(-((t-tprime)**2) / (l**2)) #rbf
    # return np.exp(-((t-tprime)**2) / 2 * (l**2)) #rbf

@jit(nopython=True)
def k_fu(t, tprime, s, lengthscale):
    l = lengthscale
    return (0.5 * np.sqrt(math.pi) * l * (math.erf((t - tprime) / l) + math.erf((tprime - s) / l))) / (np.absolute(s-t))

@jit(nopython=True)
def g(z):
    return 1.0 * z * np.sqrt(math.pi) * math.erf(z) + np.exp(-(z**2))

# @jit(nopython=False)
# def frb_update_gradients_full_X2_none(dL_dK, X, X2, lengthscale, variance): 
#     if X2 is None:  # we're finding dK_xx/dTheta
#         X2 = X
#     dK_dl_term = np.zeros((X.shape[0], X2.shape[0], lengthscale.shape[0]))
#     k_term = np.zeros((X.shape[0], X2.shape[0], lengthscale.shape[0]))
#     for il in range(lengthscale.shape[0]):
#         l = lengthscale[il]
#         idx = int(il * 2)
#         for i in range(X.shape[0]):
#             x = X[i]
#             for j in range(X2.shape[0]):
#                 ###This line was like this: x2 = X[j] I think it was a bug!!!
#                 x2 = X2[j]
#                 dK_dl_term[i, j, il] = dk_dl(x[-1], x2[-1], x[idx], x2[idx], x[idx+1], x2[idx+1], l)
#                 k_term[i, j, il] = k(x, x2, idx, l)
#     lengthscale_gradient = np.ones((lengthscale.shape[0]))
#     for il in range(lengthscale.shape[0]):
#         l = lengthscale[il]
#         dK_dl = variance[0] * dK_dl_term[:,:,il]
#         # It doesn't work without these three lines but I don't know why is that!!!
#         for jl in range(lengthscale.shape[0]): ##@FARIBA Why do I have to comment this out??
#             l = lengthscale[jl]
#             if jl != il:
#                 dK_dl *= k_term[:,:,jl]
#         lengthscale_gradient[il] = np.sum(dL_dK * dK_dl)
#     dK_dv = frb_calc_K_wo_variance(X, X2, lengthscale) #the gradient wrt the variance is k.
#     variance_gradient = np.sum(dL_dK * dK_dv)
#     return lengthscale_gradient, variance_gradient

# NOT USED, kept for refernece only
# def update_gradients_full_orig(self, dL_dK, X, X2=None): 
#     if X2 is None:  #we're finding dK_xx/dTheta
#         lengthscale_gradient, variance_gradient = frb_update_gradients_full_X2_none(dL_dK, X, X2, np.array(self.lengthscale), np.array(self.variance))
#         for il,_ in enumerate(self.lengthscale):
#             self.lengthscale.gradient[il] = lengthscale_gradient[il]
#         self.variance.gradient = variance_gradient
#         # dK_dl_term = np.zeros([X.shape[0], X.shape[0], self.lengthscale.shape[0]])
#         # k_term = np.zeros([X.shape[0], X.shape[0], self.lengthscale.shape[0]])
#         # dK_dl = np.zeros([X.shape[0], X.shape[0], self.lengthscale.shape[0]])
#         # dK_dv = np.zeros([X.shape[0], X.shape[0]])
#         # for il, l in enumerate(self.lengthscale):
#         #     idx = il * 2
#         #     for i, x in enumerate(X):
#         #         for j, x2 in enumerate(X):
#         #             dK_dl_term[i, j, il] = self.dk_dl(x[-1],x2[-1],x[idx],x2[idx],x[idx+1],x2[idx+1], l)
#         #             k_term[i, j, il] = self.k(x, x2, idx, l)
#         #  ****** when calculating dK_dl, based on how many lengthscale we are having we will have 
#         #  different K for every dimension. So for example if l has 3 dimensions then we have 
#         #  K = K1*K2*K3 now for the dK/dl2 we will have K1*dK2/dl2*K3 
#         # for il,l in enumerate(self.lengthscale):
#         #     dK_dl = self.variance[0] * dK_dl_term[:,:,il]
#         #     print('dK_dl:', dK_dl)
#         #     # It doesn't work without these three lines but I don't know what is that!!!
#         #     for jl, l in enumerate(self.lengthscale): ##@FARIBA Why do I have to comment this out??
#         #         if jl != il:
#         #             dK_dl *= k_term[:,:,jl]
#         #             print('dK_dl inside!! what is this?', dK_dl)
#         #     self.lengthscale.gradient[il] = np.sum(dL_dK * dK_dl)
#         # TODO!!!!
#         # TODO: Why calc_K_wo_variance(X,X) is not calc_K_wo_variance(X,X2)????
#         # dK_dv = self.calc_K_wo_variance(X,X) #the gradient wrt the variance is k.
#         # self.variance.gradient = np.sum(dL_dK * dK_dv)
#     else:     #we're finding dK_xf/Dtheta
#         raise NotImplementedError("Currently this function only handles finding the gradient of a single vector of inputs (X) not a pair of vectors (X and X2)")
