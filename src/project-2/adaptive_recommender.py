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

import numpy as np

class AdaptiveRecommender(Recommender):

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
        self.w_alphas = np.ones((self.n_outcomes, self.n_actions, data.shape[1]))
        self.y_alphas = np.ones((self.n_outcomes, self.n_actions))
        for i in range(len(data)):
            # Scale down the count by the number of 1s. We don't know how many
            # or which features are responsible.
            self.w_alphas[outcome, action, :] += user / user.sum()
            self.y_alphas[outcome, action] += 1

        return None

    # Return a distribution of effects for a given person's data and a specific treatment.
    # This should be an numpy.array of length self.n_outcomes
    def predict_proba(self, data, treatment):
        a = treatment
        # Draw random samples representing the w's in the a-row of W-matrix
        samples = np.array([np.random.dirichlet(self.w_alphas[:,a,i], self.n_draws)
                            for i in range(self.w_alphas.shape[2])])
        # Center each row of w's
        samples -= samples.mean(axis=0)
        # Find the estimated probability of healing given the action
        p_y_a = np.random.dirichlet(self.y_alphas[:,a], self.n_draws)
        # Calculating what we compare, it's not a probability, but we choose
        # outcome with the value, p_y_a adjusted according to data.
        p_ish_y_ax = np.dot(samples.T, data.ravel()).T * p_y_a
        # For each sample, choose the most likely outcome
        y_hats = np.argmax(p_ish_y_ax, axis=1)
        # See how many of each outcome, and convert to probabilities
        return np.bincount(y_hats, minlength=self.n_outcomes) / self.n_draws

    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        super().observe(user, action, outcome)
        # Scale down the count by the number of 1s. We don't know how many
        # or which features are responsible.
        self.w_alphas[outcome, action, :] += user / user.sum()
        self.y_alphas[outcome, action] += 1
        return None
