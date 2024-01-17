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
    def __init__(self, M, C, outcome_dict, taus=None, dist='Categorical'):
        """
        @param C:  number of classes in the categorical observations
        """
        self.M = M
        self.C = C
        self.outcome_dict = outcome_dict

        # Parameters linking input to state distribution
        self.Wk = npr.randn(1, C, M + 1)
        self.taus = taus
        self.dist = dist

    @property
    def params(self):
        return self.Wk

    @params.setter
    def params(self, value):
        self.Wk = value

    def log_prior(self):
        return 0

    # Calculate time dependent logits - output is matrix of size Tx1xC
    # Input is size TxM
    def calculate_logits(self, input):
        # Update input to include offset term at the end:
        input = np.append(input, np.ones((input.shape[0], 1)), axis=1)

        # Input effect; transpose so that output has dims TxKxC
        time_dependent_logits = np.transpose(np.dot(self.Wk, input.T), (2, 0, 1))
        if self.dist == 'Categorical':
            time_dependent_logits = time_dependent_logits - logsumexp(
                time_dependent_logits, axis=2, keepdims=True)
        elif self.dist == 'RUNML':
            time_dependent_logits = calculate_NestedMultinominalLogit(time_dependent_logits, 
                                                                      self.outcome_dict,
                                                                      self.taus,
                                                                      self.C)
        else:
            raise NotImplementedError
        
        return time_dependent_logits

    # Calculate log-likelihood of observed data
    def log_likelihoods(self, data, input, mask, tag):
        time_dependent_logits = self.calculate_logits(input)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        if self.dist == 'Categorical':
            ll = stats.categorical_logpdf(data[:, None, :],
                                          time_dependent_logits[:, :, None, :],
                                          mask=mask[:, None, :])
        elif self.dist == 'RUNML':
            ll = nested_categorical_logpdf(data[:, None, :],
                                           time_dependent_logits[:, :, None, :],
                                           mask=mask[:, None, :])
        else:
            raise NotImplementedError

        return ll

    # log marginal likelihood of data
    @ensure_args_are_lists
    def log_marginal(self, datas, inputs, masks, tags):
        elbo = self.log_prior()
        for data, input, mask, tag in zip(datas, inputs, masks, tags):
            if (self.taus is None) & (self.dist == 'RUNML'):
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

        def _objective(params, itr):
            self.params = params
            obj = self.log_marginal(datas, inputs, masks, tags)
            return -obj

        Wk_1d = optimizer(_objective, 
                          self.params,
                          num_iters=num_iters,
                          tol=tol,
                          **kwargs)
        
        # weights are flattened in 1d by default
        self.params = np.reshape(Wk_1d, (self.C, -1))[None,:,:]