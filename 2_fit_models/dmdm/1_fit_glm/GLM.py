import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
# Import useful functions from ssm package
from ssm.util import ensure_args_are_lists
from ssm.optimizers import adam, bfgs, rmsprop, sgd, lbfgs
import ssm.stats as stats
# Import functions to run RUMNL
from RUMNL import calculate_NestedMultinominalLogit, nested_categorical_logpdf, calculate_tau

class glm(object):
    def __init__(self, M, C, outcome_dict, taus=None, obs='Categorical'):
        """
        GLM class. Currently supports Categorical, RUNML, and DiagonalGaussian
        :param M int: number of predictors
        :param C int: number of classes in the categorical observations
        :param outcome_dict OrderedDict: outcome hierarchy
        :param taus list: dissimilary index for outcomes within a nest
        :param obs string: observation model that GLM assumes
        """
        self.M = M
        self.C = C
        self.outcome_dict = outcome_dict
        self.taus = taus
        self.obs = obs

        # Parameters linking input to state distribution
        if (self.obs == 'Categorical') or (self.obs == 'RUNML'):
            self.Wk = npr.randn(1, C, M + 1)
        elif self.obs == 'DiagonalGaussian':
            self.Wk = npr.randn(1, M + 1)
            self.mus = npr.randn(1, 1)
            self._log_sigmasq = -2 + npr.randn(1, 1)
        else:
            raise NotImplementedError

    @property
    def weights(self):
        return self.Wk

    @weights.setter
    def weights(self, value):
        self.Wk = value

    def log_prior(self):
        return 0

    # Calculate time dependent logits - output is matrix of size Tx1xC
    # Input is size TxM
    def calculate_logits(self, input, Wk=None):
        if Wk is None:
            Wk = self.Wk
        # Update input to include offset term at the end:
        input = np.append(input, np.ones((input.shape[0], 1)), axis=1)
        # Input effect; transpose so that output has dims TxKxC
        time_dependent_logits = np.transpose(np.dot(Wk, input.T), (2, 0, 1))
        if self.obs == 'Categorical':
            time_dependent_logits = time_dependent_logits - logsumexp(
                time_dependent_logits, axis=2, keepdims=True)
        elif self.obs == 'RUNML':
            time_dependent_logits = calculate_NestedMultinominalLogit(time_dependent_logits, 
                                                                      self.outcome_dict,
                                                                      self.taus,
                                                                      self.C)
        else:
            raise NotImplementedError
        
        return time_dependent_logits

    # Calculate time dependent identities - output is mu and sigma for N(mu, sigma)
    # Input is size TxM as well as Tag, which is a 1xM vector of change onset.
    def calculate_identities(self, input, tag, Wk=None):
        if Wk is None:
            Wk = self.Wk
        # Update input to include offset term at the end:
        input = np.append(input, np.ones((input.shape[0], 1)), axis=1)
        # Input effect; transpose so that output has dims Tx1
        time_dependent_identity = np.transpose(np.dot(Wk, input.T), (1, 0))

        # time_dependent_identity = np.add(time_dependent_identity, tag) # wX + COnset

        if self.obs == 'DiagonalGaussian':
            mus = np.mean(time_dependent_identity, axis=0) # use np.average
            sqerr = (time_dependent_identity - mus) ** 2
            log_sigmasq = np.log(np.mean(sqerr, axis=0)) # use np.average
            sigmas = np.exp(log_sigmasq) + 1e-16
        else:
            raise NotImplementedError
        
        return mus, sigmas

    # Calculate log-likelihood of observed data
    def log_likelihoods(self, data, input, mask, tag, Wk=None):
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        if self.obs == 'Categorical':
          time_dependent_logits = self.calculate_logits(input, Wk=Wk)
          ll = stats.categorical_logpdf(data[:, None, :],
                                          time_dependent_logits[:, :, None, :],
                                          mask=mask[:, None, :])
        elif self.obs == 'RUNML':
             time_dependent_logits = self.calculate_logits(input, Wk=Wk)
             ll = nested_categorical_logpdf(data[:, None, :],
                                           time_dependent_logits[:, :, None, :],
                                           mask=mask[:, None, :])
        elif self.obs == 'DiagonalGaussian':
             mus, sigmas = self.calculate_identities(input, tag, Wk=Wk)
             ll = stats.diagonal_gaussian_logpdf(data[:, None, :], mus, sigmas,
                                                 mask=mask[:, None, :])
        else: 
            raise NotImplementedError

        return ll

    # log marginal likelihood of data
    @ensure_args_are_lists
    def log_marginal(self, datas, inputs, masks, tags):
        elbo = self.log_prior()
        for data, input, mask, tag in zip(datas, inputs, masks, tags):
            if (self.taus is None) & (self.obs == 'RUNML'):
                self.taus = calculate_tau(data, self.outcome_dict, self.C)
            lls = self.log_likelihoods(data, input, mask, tag)
            elbo += np.sum(lls)
        return elbo

    # recover log likelihood from weights
    def recover_lls(self, datas, inputs, masks, tags, Weights: list):
        lls_all = []
        for wk in Weights:
            elbo = self.log_prior()
            for data, input, mask, tag in zip(datas, inputs, masks, tags):
                if (self.taus is None) & (self.obs == 'RUNML'):
                    self.taus = calculate_tau(data, self.outcome_dict, self.C)
                lls = self.log_likelihoods(data, input, mask, tag, wk)
                elbo += np.sum(lls)
            lls_all.append(elbo)
        return lls_all

    @ensure_args_are_lists
    def fit_glm(self,
                datas,
                inputs,
                masks,
                tags,
                num_iters=10000,
                optimizer="bfgs", # or lbfgs
                tol=1e-2,
                **kwargs):
        optimizer = dict(adam=adam, bfgs=bfgs, rmsprop=rmsprop,
                         sgd=sgd, lbfgs=lbfgs)[optimizer]

        def _objective(weights, itr):
            self.weights = weights
            obj = self.log_marginal(datas, inputs, masks, tags)
            return -obj

        if (self.obs == 'Categorical') or (self.obs == 'RUNML'):
            Wk_1d, out_1d = optimizer(_objective, 
                                   self.weights,
                                   num_iters=num_iters,
                                   tol=tol,
                                   full_output=True,
                                   **kwargs)
            # weights are flattened in 1d by default
            self.weights = np.reshape(Wk_1d, (self.C, -1))[None,:,:]

            out = [np.reshape(o, (self.C, -1))[None,:,:] for o in out_1d['allvecs']]
            return out

        elif self.obs == 'DiagonalGaussian':
             self.weights, out_1d = optimizer(_objective, 
                                              self.weights,
                                              num_iters=num_iters,
                                               full_output=True,
                                               **kwargs)
             out = [o[None,:] for o in out_1d]
             return out
        else:
            raise NotImplementedError