# Written by Mike Smith and modified and extended by Fariba.

from __future__ import division
import numpy as np
from GPy.kern import Kern
# from GPy.kern.src.stationary import Stationary
from GPy.core.parameterization import Param
# from ...util.linalg import tdot
# from ... import util
from paramz.transformations import Logexp
import math
from numba import jit
# import pdb


# If the kernel is stationary, inherit from Stationary in
# GPy.kern.src.stationary.py If the kernel is non-stationary,
# inherit from Kern in GPy.kern.src.kern.py
class Mix_Integral_(Kern):
# class Mix_Integral_(Stationary):

    """
    Integral kernel. This kernel allows 1d histogram or binned data to be modelled.
    The outputs are the counts in each bin. The inputs (on two dimensions) are the start and end points of each bin.
    The kernel's predictions are the latent function which might have generated those binned results.
    """
 
    def __init__(self, input_dim, variance=None, lengthscale=None, ARD=False, active_dims=None, name='mix_integral_'):
        super(Mix_Integral_, self).__init__(input_dim, active_dims, name)
        #  For the stationary case:
        # super(Mix_Integral_, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)

        if lengthscale is None:
            lengthscale = np.ones(1)
        else:
            lengthscale = np.asarray(lengthscale)
        import pdb; pdb.set_trace()
        assert len(lengthscale)==(input_dim-1)/2
        


        self.lengthscale = Param('lengthscale', lengthscale, Logexp()) #Logexp - transforms to allow positive only values...
        self.variance = Param('variance', variance, Logexp()) #and here.
        self.link_parameters(self.variance, self.lengthscale) #this just takes a list of parameters we need to optimise.
        self.ARD = ARD


    #useful little function to help calculate the covariances.
    def g(self, z):
        return 1.0 * z * np.sqrt(math.pi) * math.erf(z) + np.exp(-(z**2))

    def k_ff(self, t, tprime, s, sprime, lengthscale):
        """Covariance between observed values.

        s and t are one domain of the integral (i.e. the integral between s and t)
        sprime and tprime are another domain of the integral (i.e. the integral between sprime and tprime)

        We're interested in how correlated these two integrals are.

        Note: We've not multiplied by the variance, this is done in K."""
     #######   l = lengthscale * np.sqrt(2)###TO REINSTATE
        l = lengthscale
        return 0.5 * (l ** 2) * ( self.g((t - sprime) / l) + self.g((tprime - s) / l) - self.g((t - tprime) / l) - self.g((s - sprime) / l))
    
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
        return 0.5 * np.sqrt(math.pi) * l * (math.erf((t - tprime) / l) + math.erf((tprime - s) / l))

    def k(self, x, x2, idx, l):
        # pdb.set_trace()
        """Helper function to compute covariance in one dimension (idx) between a pair of points.
        The last element in x and x2 specify if these are integrals (0) or latent values (1).
        l = that dimension's lengthscale
        """
        if (x[-1] == 0) and (x2[-1] == 0):
            return self.k_ff(x[idx], x2[idx], x[idx+1], x2[idx+1], l)
        if (x[-1] == 0) and (x2[-1] == 1):
            return self.k_fu(x[idx], x2[idx], x[idx+1], l)
        if (x[-1] == 1) and (x2[-1] == 0):
            return self.k_fu(x2[idx], x[idx], x2[idx+1], l)                        
        if (x[-1] == 1) and (x2[-1] == 1):
            return self.k_uu(x[idx], x2[idx], l)
            # TODO: talk to Mauricio if for kuu we should use the average or the first element only!!!
            #  If we should use the sum then kfu and kuf also should use the some of the dimenstions/2
            # return self.k_uu(x[idx], x2[idx], x[idx+1], x2[idx+1])
        assert False, "Invalid choice of latent/integral parameter (set the last column of X to 0s and 1s to select this)"

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
        
        if (t_type == 0) and (tprime_type == 0): #both integrals
            return l * ( self.h((t - sprime) / l) - self.h((t - tprime) / l) + self.h((tprime - s) / l) - self.h((s - sprime) / l))
        if (t_type == 0) and (tprime_type == 1): #integral vs latent 
            return self.hp((t - tprime) / l) + self.hp((tprime - s) / l)
        if (t_type == 1) and (tprime_type == 0): #integral vs latent 
            return self.hp((tprime - t) / l) + self.hp((t - sprime) / l)
            #swap: t<->tprime (t-s)->(tprime-sprime)
        if (t_type == 1) and (tprime_type == 1): #both latent observations 
            return 2 * (t - tprime) ** 2 / (l ** 3) * np.exp(-((t - tprime) / l) ** 2)
        assert False, "Invalid choice of latent/integral parameter (set the last column of X to 0s and 1s to select this)"

    def update_gradients_full(self, dL_dK, X, X2=None): 
        # import pdb; pdb.set_trace()
        # print('X shape:', X.shape)
        if X2 is None:  # we're finding dK_xx/dTheta
            X2 = X
        dK_dl_term = np.zeros([X.shape[0], X2.shape[0], self.lengthscale.shape[0]])
        # print('dK_dl_term shape:', dK_dl_term.shape)
        k_term = np.zeros([X.shape[0], X2.shape[0], self.lengthscale.shape[0]])
        # dK_dl = np.zeros([X.shape[0], X.shape[0], self.lengthscale.shape[0]])
        # print('dK_dl.shape:', dK_dl.shape)
        # dK_dv = np.zeros([X.shape[0], X.shape[0]])
        for il, l in enumerate(self.lengthscale):
            idx = il * 2
            for i, x in enumerate(X):
                for j, x2 in enumerate(X2):
                    dK_dl_term[i, j, il] = self.dk_dl(x[-1], x2[-1], x[idx], x2[idx], x[idx+1], x2[idx+1], l)
                    k_term[i, j, il] = self.k(x, x2, idx, l)
        for il,l in enumerate(self.lengthscale):
            dK_dl = self.variance[0] * dK_dl_term[:,:,il]
            # print ('dK_dl second shape:', dK_dl.shape)
            # print ('dK_dl_term second shape:', dK_dl_term.shape)
            # print('dK_dl: \n', dK_dl)
            # It doesn't work without these three lines but I don't know what is that!!!
            for jl, l in enumerate(self.lengthscale): ##@FARIBA Why do I have to comment this out??
                if jl != il:
                    dK_dl *= k_term[:,:,jl]
            #         print('dK_dl inside!! what is this? \n', dK_dl)
            # print('dK_dl shape:', dK_dl.shape)
            self.lengthscale.gradient[il] = np.sum(dL_dK * dK_dl)
        # print('dK_dl shape:', dK_dl.shape)
        dK_dv = self.calc_K_wo_variance(X,X2) #the gradient wrt the variance is k.
        self.variance.gradient = np.sum(dL_dK * dK_dv)
    
    def update_gradients_diag(self, dL_dKdiag, X):
        """
        Given the derivative of the objective with respect to the diagonal of
        the covariance matrix, compute the derivative wrt the parameters of
        this kernel and store in the <parameter>.gradient field.

        See also update_gradients_full
        """
        self.variance.gradient = np.sum(dL_dKdiag)
        self.lengthscale.gradient = 0.
    
    # else:     #we're finding dK_xf/Dtheta
    #     raise NotImplementedError("Currently this function only handles finding the gradient of a single vector of inputs (X) not a pair of vectors (X and X2)")

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

    def dk_dz(self, input_type_1, input_type_2, t, tprime, s, sprime, lengthscale):
        # here t stands for z and tprime stands for z' (latent inputs)
        # pdb.set_trace()
        input_type_1 = int(input_type_1)
        input_type_2 = int(input_type_2)
        # print('input_type_1:', input_type_1)
        # print('input_type_2:', input_type_2)
        if (input_type_1 == 0) and (input_type_2 == 1):
            # print('kfu')
            return self.dkfu_dz(t, tprime, s, lengthscale) 
        # TODO: Write the gradient for the below combination--- check if it is correct!!!
        if (input_type_1 == 1) and (input_type_2 == 0):
            # print('kuf')
            return self.dkfu_dz(tprime, t, sprime, lengthscale) 
        if (input_type_1 == 1) and (input_type_2 == 1):
            # TODO: Aren't X and X' here refer to Z and Z' which is t and tprime???
            # return self.dkuu_dz(s, t, sprime, tprime, lengthscale) 
            return self.dkuu_dz(t, tprime, s, sprime, lengthscale) 
        if (input_type_1 == 0) and (input_type_2 == 0):
            # print(input_type_1)
            # print(input_type_2)
            # print('first element of the first input that are both 0')
            #  It is derivative of kff/dz which is zero!
            return 0 
        assert False, 'KFU should have X and Z and KUU should have Z and Z!'
    
    def dkuu_dz(self, t, tprime, s, sprime, lengthscale):
        l = lengthscale
        # z = (s + t) / 2
        # zprime = (sprime + tprime) / 2
        # print ('z:', z)
        # print('zprime:', zprime)
        # print('dkuu result:', -2 * (t - tprime) / (l ** 2) * np.exp(-(t - tprime) ** 2 / (l ** 2)))
        # print('t:', t)
        # print('tprime:', tprime)
        # print('s:', s)
        # print('sprime:', sprime)
        return -2 * (t - tprime) / (l ** 2) * np.exp(-(t - tprime) ** 2 / l ** 2)
        # return -(t - tprime) / l * np.exp(-0.5 * (t - tprime) ** 2 / l ** 2)
        # assert False, 'KFU should have X and Z and KUU should have Z and Z!'

    def dkfu_dz(self, t, tprime, s, lengthscale):
        l = lengthscale
        # if (X[-1] == 0) and (X2[-1] == 1):
        # print('t:', t)
        # print('tprime:', tprime)
        # print('s:', s)
        return -np.exp(-(t - tprime) ** 2 / l ** 2) + np.exp(-(tprime - s) ** 2 / l ** 2) 
        # if (X[-1] == 1) and (X2[-1] == 1):
        #     return (t - tprime) / l**2 * np.exp(-(tprime - s)**2 / l ** 2)
        # assert False, 'KFU should have X and Z and KUU should have Z and Z!'
    
    # def dKfu_dZ(self, dkfu_dz):
    #     return 1#self.variance[0] * dk_dz * dL_dK
    
    # def dKuu_dZ(self, dkuu_dz, X, X2=None):
        # # return 2 * self.variance * dK_dz * dL_dK
        # # self.inducinginputs.gradient[il] = np.sum(dL_dK * dK_dz)?????

    '''
     ****** when calculating dK_dz, based on how many lengthscale we are having we will have 
     different K for every dimension. So for example if l has 3 dimensions then we have 
     K = K1*K2*K3 now for the dK/dz2 we will have K1*dK2/dz2*K3 
    '''
    # def gradients_X(self, dL_dK, X, X2=None):
    #     return self.gradients_X_(dL_dK, X, X2)

    def gradients_X(self, dL_dK, X, X2=None):

        """
        .. math::
            \\frac{\partial L}{\partial X} = \\frac{\partial L}{\partial K}\\frac{\partial K}{\partial X}
        """
        if X2 is None:
            X2 = X
        dK_dz_term = np.zeros((X.shape[0], X2.shape[0], self.lengthscale.shape[0]))
        k_term = np.zeros((X.shape[0], X2.shape[0], self.lengthscale.shape[0]))
        # print ('dKuu_dZ_term shape:', dK_dz_term.shape)
        for il, l in enumerate(self.lengthscale):
            idx = il * 2
            for i, x in enumerate(X):
                for j, x2 in enumerate(X2):
                    # print('x2:', x2)
                    # import pdb; pdb.set_trace()        
                    dK_dz_term[i, j, il] = self.dk_dz(x[-1], x2[-1], x[idx], x2[idx], x[idx+1], x2[idx+1], l)
                    # print ('dK_dz_term: \n', dK_dz_term)
                    k_term[i, j, il] = self.k(x, x2, idx, l)
        #     print('dimension change!!!')   
        # print ('dK_dz_term: \n', dK_dz_term)
        # print ('k_term: \n', k_term)
        # The result of the derivative should be 
        inducing_inputs_gradient = np.ones((X.shape[0], self.lengthscale.shape[0]))
        # print ('inducing_inputs_gradient origgggg:\n', inducing_inputs_gradient)
        for il,l in enumerate(self.lengthscale):
            dK_dz = self.variance[0] * dK_dz_term[:,:,il]
            # print('dK_dz:\n', dK_dz)
        #     # It doesn't work without these three lines but I don't know what is that!!!
            for jl, l in enumerate(self.lengthscale): ##@FARIBA Why do I have to comment this out??
                if jl != il:
                    dK_dz *= k_term[:,:,jl]
                    # print('dK_dz inside \n', dK_dz)
            tmp = dL_dK * dK_dz
            inducing_inputs_gradient[:, il][:, None] = np.sum(tmp, axis=1)[:,None]
            # inducing_inputs_gradient[il] = np.sum(dL_dK * dK_dz)
        # print('inducing_inputs_gradient end***: \n', inducing_inputs_gradient)
        return inducing_inputs_gradient
        # else:
        #     raise NotImplementedError('Currently this function only handles finding the gradient of a single vector of inputs (X) not a pair of vectors (X and X2)")')
    # def gradients_X_diag(self, dL_dKdiag, X):
    #     return np.zeros(X.shape)

    def gradients_X_orig(self, dL_dK, X, X2=None):

        # pdb.set_trace()
        """
        .. math::
            \\frac{\partial L}{\partial X} = \\frac{\partial L}{\partial K}\\frac{\partial K}{\partial X}
        """
        if X2 is None:
            dK_dz_term = np.zeros((X.shape[0], X.shape[0], self.lengthscale.shape[0]))
            k_term = np.zeros((X.shape[0], X.shape[0], self.lengthscale.shape[0]))
            # print ('dKuu_dZ_term shape:', dK_dz_term.shape)
            for il, l in enumerate(self.lengthscale):
                idx = il * 2
                for i, x in enumerate(X):
                    for j, x2 in enumerate(X):
                        # print('x2:', x2)
                        # import pdb; pdb.set_trace()        
                        dK_dz_term[i, j, il] = self.dk_dz(x[-1], x2[-1], x[idx], x2[idx], x[idx+1], x2[idx+1], l)
                        # print ('dK_dz_term: \n', dK_dz_term)
                        k_term[i, j, il] = self.k(x, x2, idx, l)
            #     print('dimension change!!!')   
            # print ('dK_dz_term: \n', dK_dz_term)
            # print ('k_term: \n', k_term)
            # The result of the derivative should be 
            inducing_inputs_gradient = np.ones((X.shape[0], self.lengthscale.shape[0]))
            # print ('inducing_inputs_gradient origgggg:\n', inducing_inputs_gradient)
            for il,l in enumerate(self.lengthscale):
                dK_dz = self.variance[0] * dK_dz_term[:,:,il]
                # print('dK_dz:\n', dK_dz)
            #     # It doesn't work without these three lines but I don't know what is that!!!
                for jl, l in enumerate(self.lengthscale): ##@FARIBA Why do I have to comment this out??
                    if jl != il:
                        dK_dz *= k_term[:,:,jl]
                        # print('dK_dz inside \n', dK_dz)
                tmp = dL_dK * dK_dz
                inducing_inputs_gradient[:, il][:, None] = np.sum(tmp, axis=1)[:,None]
                # inducing_inputs_gradient[il] = np.sum(dL_dK * dK_dz)
            # print('inducing_inputs_gradient end***: \n', inducing_inputs_gradient)
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
    if (x[-1] == 0) and (x2[-1] == 0):
        return k_ff(x[idx], x2[idx], x[idx+1], x2[idx+1], l)
    if (x[-1] == 0) and (x2[-1] == 1):
        return k_fu(x[idx], x2[idx], x[idx+1], l)
    if (x[-1] == 1) and (x2[-1] == 0):
        return k_fu(x2[idx], x[idx], x2[idx+1], l)                        
    if (x[-1] == 1) and (x2[-1] == 1):
        return k_uu(x[idx], x2[idx], l)
    assert False, "Invalid choice of latent/integral parameter (set the last column of X to 0s and 1s to select this)"

@jit(nopython=True)
def k_ff(t, tprime, s, sprime, lengthscale):
    l = lengthscale
    return 0.5 * (l**2) * (g((t - sprime) / l) + g((tprime - s) / l) - g((t - tprime) / l) - g((s - sprime) / l))

@jit(nopython=True)
def k_uu(t, tprime, lengthscale):
    l = lengthscale
    return np.exp(-((t-tprime)**2) / (l**2)) #rbf
    # return np.exp(-((t-tprime)**2) / 2 * (l**2)) #rbf

@jit(nopython=True)
def k_fu(t, tprime, s, lengthscale):
    l = lengthscale
    return 0.5 * np.sqrt(math.pi) * l * (math.erf((t - tprime) / l) + math.erf((tprime - s) / l))

@jit(nopython=True)
def g(z):
    return 1.0 * z * np.sqrt(math.pi) * math.erf(z) + np.exp(-(z**2))

# @jit(nopython=False)
# def frb_update_gradients_full_X2_none(dL_dK, X, X2, lengthscale, variance): 
#     import pdb; pdb.set_trace()
#     dK_dl_term = np.zeros((X.shape[0], X2.shape[0], lengthscale.shape[0]))
#     k_term = np.zeros((X.shape[0], X2.shape[0], lengthscale.shape[0]))
#     for il in range(lengthscale.shape[0]):
#         l = lengthscale[il]
#         idx = int(il * 2)
#         for i in range(X.shape[0]):
#             x = X[i]
#             for j in range(X2.shape[0]):
#                 x2 = X[j]
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

# @jit(nopython=True)
# def frb_update_gradients_full_X2_none(dL_dK, X, lengthscale, variance): 
#     dK_dl_term = np.zeros((X.shape[0], X.shape[0], lengthscale.shape[0]))
#     k_term = np.zeros((X.shape[0], X.shape[0], lengthscale.shape[0]))
#     for il in range(lengthscale.shape[0]):
#         l = lengthscale[il]
#         idx = int(il * 2)
#         for i in range(X.shape[0]):
#             x = X[i]
#             for j in range(X.shape[0]):
#                 x2 = X[j]
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
#     dK_dv = frb_calc_K_wo_variance(X, X, lengthscale) #the gradient wrt the variance is k.
#     variance_gradient = np.sum(dL_dK * dK_dv)
#     return lengthscale_gradient, variance_gradient

@jit(nopython=True)
def dk_dl(t_type, tprime_type, t, tprime, s, sprime, l): #derivative of the kernel wrt lengthscale
    #t and tprime are the two start locations
    #s and sprime are the two end locations
    #if t_type is 0 then t and s should be in the equation
    #if tprime_type is 0 then tprime and sprime should be in the equation.
    
    if (t_type == 0) and (tprime_type == 0): #both integrals
        return l * (h((t - sprime) / l) - h((t - tprime) / l) + h((tprime - s) / l) - h((s - sprime) / l))
    if (t_type == 0) and (tprime_type == 1): #integral vs latent 
        return hp((t - tprime) / l) + hp((tprime - s) / l)
    if (t_type == 1) and (tprime_type == 0): #integral vs latent 
        return hp((tprime - t) / l) + hp((t - sprime) / l)
        #swap: t<->tprime (t-s)->(tprime-sprime)
    if (t_type == 1) and (tprime_type == 1): #both latent observations            
        return 2 * (t - tprime) **2 / (l ** 3) * np.exp(-((t - tprime) / l) ** 2)
    assert False, "Invalid choice of latent/integral parameter (set the last column of X to 0s and 1s to select this)"

@jit(nopython=True)    
def h(z):
    return 0.5 * z * np.sqrt(math.pi) * math.erf(z) + np.exp(-(z**2))

@jit(nopython=True)    
def hp(z):
    return 0.5 * np.sqrt(math.pi) * math.erf(z) - z * np.exp(-(z**2))

# ------------------------------------------------------------------------------------------------------------------------------
# ///////////////////////////////////////start Fariba//////////////////////////////////////////////
    # def frb(self, l):
    #     from functools import partial
    #     from GPy.models import GradientChecker
    #     f = partial(self.K)
    #     df = partial(self.update_gradients_full)
    #     grad = GradientChecker(f, df, l, 'l')
    #     grad.checkgrad(verbose=1)
# ///////////////////////////end Fariba/////////////////////////////////////////////////////////

    # def gradients_X_diag(self, dL_dKdiag, X):
    #     """
    #     The diagonal of the derivative w.r.t. X
    #     """
    #     raise NotImplementedError

    # def update_gradients_diag(self, dL_dKdiag, X):
    #     """ update the gradients of all parameters when using only the diagonal elements of the covariance matrix"""
    #     raise NotImplementedError


# # -----------------------------------------------------------------
# For RBF Kernel!
#     """
#     Not used yet!
#     """
    # def _inv_dist(self, X, X2=None):
    #     """
    #     Compute the elementwise inverse of the distance matrix, expecpt on the
    #     diagonal, where we return zero (the distance on the diagonal is zero).
    #     This term appears in derviatives.
    #     """
    #     dist = self._scaled_dist(X, X2).copy()
    #     return 1./np.where(dist != 0., dist, np.inf)

    # def _gradients_X_pure(self, dL_dK, X, X2=None):
    #     invdist = self._inv_dist(X, X2)
    #     dL_dr = self.dK_dr_via_X(X, X2) * dL_dK
    #     tmp = invdist*dL_dr
    #     if X2 is None:
    #         tmp = tmp + tmp.T
    #         X2 = X

    #     #The high-memory numpy way:
    #     #d =  X[:, None, :] - X2[None, :, :]
    #     #grad = np.sum(tmp[:,:,None]*d,1)/self.lengthscale**2

    #     #the lower memory way with a loop
    #     grad = np.empty(X.shape, dtype=np.float64)
    #     for q in range(self.input_dim):
    #         np.sum(tmp*(X[:,q][:,None]-X2[:,q][None,:]), axis=1, out=grad[:,q])
    #     return grad/self.lengthscale**2

    # def gradients_X(self, dL_dK, X, X2=None):
    #     """
    #     Given the derivative of the objective wrt K (dL_dK), compute the derivative wrt X
    #     """
        
    #     return self._gradients_X_pure(dL_dK, X, X2)

    # def _scaled_dist(self, X, X2=None):
    #     """
    #     Efficiently compute the scaled distance, r.

    #     ..math::
    #         r = \sqrt( \sum_{q=1}^Q (x_q - x'q)^2/l_q^2 )

    #     Note that if thre is only one lengthscale, l comes outside the sum. In
    #     this case we compute the unscaled distance first (in a separate
    #     function for caching) and divide by lengthscale afterwards

    #     """
    #     if self.ARD:
    #         if X2 is not None:
    #             X2 = X2 / self.lengthscale
    #         return self._unscaled_dist(X/self.lengthscale, X2)
    #     else:
    #         return self._unscaled_dist(X, X2)/self.lengthscale

    # def _unscaled_dist(self, X, X2=None):
    #     """
    #     Compute the Euclidean distance between each row of X and X2, or between
    #     each pair of rows of X if X2 is None.
    #     """
    #     #X, = self._slice_X(X)
    #     if X2 is None:
    #         Xsq = np.sum(np.square(X),1)
    #         r2 = -2.*tdot(X) + (Xsq[:,None] + Xsq[None,:])
    #         util.diag.view(r2)[:,]= 0. # force diagnoal to be zero: sometime numerically a little negative
    #         r2 = np.clip(r2, 0, np.inf)
    #         return np.sqrt(r2)
    #     else:
    #         #X2, = self._slice_X(X2)
    #         X1sq = np.sum(np.square(X),1)
    #         X2sq = np.sum(np.square(X2),1)
    #         r2 = -2.*np.dot(X, X2.T) + (X1sq[:,None] + X2sq[None,:])
    #         r2 = np.clip(r2, 0, np.inf)
    #         return np.sqrt(r2)

    # def dK_dr_via_X(self, X, X2):
    #     """
    #     compute the derivative of K wrt X going through X
    #     """
    #     #a convenience function, so we can cache dK_dr
    #     return self.dK_dr(self._scaled_dist(X, X2))

    # def K_of_r(self, r):
    #     return self.variance * np.exp(-0.5 * r**2)

    # def dK_dr(self, r):
    #     return -r*self.K_of_r(r)
