# GLM class
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
# Import useful functions from ssm package
from ssm.util import ensure_args_are_lists
from ssm.optimizers import adam, bfgs, rmsprop, sgd, lbfgs
import ssm.stats as stats
# Import functions to run RUMNL
from RUMNL import calculate_NestedMultinominalLogit, nested_categorical_logpdf, calculate_tau
npr.seed(65)  # set seed in case of randomization

class glm(object):
    def __init__(self, M, C, outcome_dict, taus=None, obs='Categorical'):
        """
        @param C:  number of classes in the categorical observations
        """
        self.M = M
        self.C = C
        self.outcome_dict = outcome_dict

        # Parameters linking input to state distribution
        self.Wk = npr.randn(1, C, M + 1)
        self.taus = taus
        self.obs = obs
        self.mus = npr.randn(1, 1)
        self._log_sigmasq = -2 + npr.randn(1, 1)

    @property
    def weights(self):
        return self.Wk

    @weights.setter
    def weights(self, value):
        self.Wk = value

    def log_prior(self):
        return 0

    @property
    def sigmasq(self):
        return np.exp(self._log_sigmasq)

    @sigmasq.setter
    def sigmasq(self, value):
        assert np.all(value > 0) and value.shape == (1, 1)
        self._log_sigmasq = np.log(value)

    @property
    def mulogsimasq(self):
        return self.mus, self._log_sigmasq

    @mulogsimasq.setter
    def mulogsimasq(self, value):
        self.mus, self._log_sigmasq = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self._log_sigmasq = self._log_sigmasq[perm]

    # Calculate time dependent logits - output is matrix of size Tx1xC
    # Input is size TxM
    def calculate_logits(self, input):
        # Update input to include offset term at the end:
        input = np.append(input, np.ones((input.shape[0], 1)), axis=1)

        # Input effect; transpose so that output has dims TxKxC
        time_dependent_logits = np.transpose(np.dot(self.Wk, input.T), (2, 0, 1))
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

    def calculate_identities(self, input):
        # Update input to include offset term at the end:
        input = np.append(input, np.ones((input.shape[0], 1)), axis=1)

        # Input effect; transpose so that output has dims TxKxC
        time_dependent_logits = np.transpose(np.dot(self.Wk, input.T), (2, 0, 1))
        if self.obs == 'DiagonalGaussian':
            time_dependent_logits = time_dependent_logits - logsumexp(
                time_dependent_logits, axis=2, keepdims=True)
        else:
            raise NotImplementedError
        
        return time_dependent_logits

    # Calculate log-likelihood of observed data
    def log_likelihoods(self, data, input, mask, tag):
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        if self.obs == 'Categorical':
          time_dependent_logits = self.calculate_logits(input)
          ll = stats.categorical_logpdf(data[:, None, :],
                                          time_dependent_logits[:, :, None, :],
                                          mask=mask[:, None, :])
        elif self.obs == 'RUNML':
             time_dependent_logits = self.calculate_logits(input)
             ll = nested_categorical_logpdf(data[:, None, :],
                                           time_dependent_logits[:, :, None, :],
                                           mask=mask[:, None, :])
        elif self.obs == 'DiagonalGaussian':
             mus, sigmas = self.mus, np.exp(self._log_sigmasq) + 1e-16
             print(mus)
             print(sigmas)
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

    @ensure_args_are_lists
    def fit_glm(self,
                datas,
                inputs,
                masks,
                tags,
                num_iters=10000,
                optimizer="bfgs", # or lbfgs
                tol=1e-4,
                **kwargs):
        optimizer = dict(adam=adam, bfgs=bfgs, rmsprop=rmsprop,
                         sgd=sgd, lbfgs=lbfgs)[optimizer]

        def _objective_logistic(weights, itr):
            self.weights = weights
            obj = self.log_marginal(datas, inputs, masks, tags)
            return -obj

        def _objective_gaussian(sigmasq, itr):
            self.sigmasq = sigmasq
            obj = self.log_marginal(datas, inputs, masks, tags)
            return -obj

        if (self.obs == 'Categorical') or (self.obs == 'RUNML'):
            Wk_1d = optimizer(_objective_logistic, 
                              self.weights,
                              num_iters=num_iters,
                              tol=tol,
                              **kwargs)
            # weights are flattened in 1d by default
            self.weights = np.reshape(Wk_1d, (self.C, -1))[None,:,:]

        elif self.obs == 'DiagonalGaussian':
             self.sigmasq = optimizer(_objective_gaussian, 
                                      self.sigmasq,
                                      num_iters=num_iters,
                                      **kwargs)
        else:
            raise NotImplementedError








