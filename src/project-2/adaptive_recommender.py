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
            self.observe(data[i], actions[i], outcomes[i])

        #for o in range(self.n_outcomes):
        #    for a in range(self.n_actions):
        #        mask = (self.A==a).values.ravel() & (self.Y==o).values.ravel()
        #        self.w_alphas[o, a, :] += X_data[mask].sum(axis=0)
        #        self.y_alphas[o, a] += mask.sum()

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

    # Return a distribution of recommendations for a specific user datum
    # This should a numpy array of size equal to self.n_actions, summing up to 1
    #def get_action_probabilities(self, user_data):
    #    Er = np.zeros(self.n_actions)
    #    for a in range(self.n_actions):
    #        p = self.predict_proba(user_data, a)
    #        r = self.reward(a, np.arange(self.n_outcomes))
    #        Er[a] = np.dot(p, r)
    #    return np.exp(Er) / (np.exp(Er)).sum()


    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    #def recommend(self, user_data):
        #print("Recommending")
    #    return np.argmax(self.get_action_probabilities(user_data))
        # The above rule might give the best results, but it also will probably
        # never assign placebo treatment given the data we have now.
        # This is an option that reduces that problem
        #return np.random.choice(np.arange(self.n_actions), p=self.get_action_probabilities(user_data))

    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        super().observe(user, action, outcome)
        # Scale down the count by the number of 1s. We don't know how many
        # or which features are responsible.
        self.w_alphas[outcome, action, :] += user / user.sum()
        self.y_alphas[outcome, action] += 1
        return None
