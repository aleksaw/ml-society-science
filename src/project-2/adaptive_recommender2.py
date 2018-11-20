# -*- Mode: python -*-
# A simple reference recommender
#
#
# This is a medical scenario with historical data.
#
# General functions
#
# - set_reward
#
# There is a set of functions for dealing with historical data:
#
# - fit_data
# - fit_treatment_outcome
# - estimate_utiltiy
#
# There is a set of functions for online decision making
#
# - predict_proba
# - recommend
# - observe

from recommender import Recommender

from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np

class AdaptiveRecommender2(Recommender):

    #################################
    # Initialise
    #
    # Set the recommender with a default number of actions and outcomes.  This is
    # because the number of actions in historical data can be
    # different from the ones that you can take with your policy.
    def __init__(self, n_actions, n_outcomes, n_draws=1000):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.n_draws = n_draws
        self.reward = self._default_reward

    ##################################

    ## Fit a model from patient data, actions and their effects
    ## Here we assume that the outcome is a direct function of data and actions
    ## This model can then be used in estimate_utility(), predict_proba() and recommend()
    def fit_treatment_outcome(self, data, actions, outcomes, quick=False):
        #print("Fitting treatment outcomes")
        super().fit_treatment_outcome(data, actions, outcomes)
        # alphas of Dirichlet distributions
        self.x_alphas = np.ones((self.n_outcomes, self.n_actions, data.shape[1]))
        self.y_alphas = np.ones((self.n_outcomes, self.n_actions))
        for o in range(self.n_outcomes):
            for a in range(self.n_actions):
                mask = (self.A==a).ravel() & (self.Y==o).ravel()
                self.x_alphas[o, a, :] += data[mask].sum(axis=0)
                self.y_alphas[o, a] += mask.sum()

        return None

    # Return a distribution of effects for a given person's data and a specific treatment.
    # This should be an numpy.array of length self.n_outcomes
    def predict_proba(self, data, treatment):
        a = treatment
        log_x_means = self.x_alphas[:,a,:] / self.x_alphas[:,a,:].sum(axis=0)
        log_p_x_ya = np.dot(np.log(log_x_means), data)
        log_y_a = np.log(self.y_alphas[:,a] / self.y_alphas[:,a].sum(axis=0))
        p_y_xa = np.exp(log_p_x_ya + log_y_a - log_p_x_ya.mean())
        return p_y_xa / p_y_xa.sum()

    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        super().observe(user, action, outcome)
        # Scale down the count by the number of 1s. We don't know how many
        # or which features are responsible.
        self.x_alphas[outcome, action, :] += user
        self.y_alphas[outcome, action] += 1
        return None
