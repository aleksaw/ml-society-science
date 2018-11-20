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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

class HistoricalRecommender3(Recommender):

    #################################
    # Initialise
    #
    # Set the recommender with a default number of actions and outcomes.  This is
    # because the number of actions in historical data can be
    # different from the ones that you can take with your policy.
    def __init__(self, n_actions, n_outcomes, refit_trigger=1.1):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward
        self.scaler = StandardScaler()
        self.refit_trigger = 1.1

    ## Fit a model from patient data, actions and their effects
    ## Here we assume that the outcome is a direct function of data and actions
    ## This model can then be used in estimate_utility(), predict_proba() and recommend()
    def fit_treatment_outcome(self, data, actions, outcomes, quick=False):
        #print("Fitting treatment outcomes")
        self.X = self.scaler.fit_transform(data)
        self.A = actions
        self.Y = outcomes
        self._fit_models(quick=quick)
        return None

    ## Fit a model to the data we have
    def _fit_models(self, quick = False):
        self.data_in_model = self.X.shape[0]
        # We have to find some weights that balance the fact that
        weights = {0: np.mean(self.A)*2, 1: (1-np.mean(self.A))}
        self.model = LogisticRegression(C=10^8, solver='sag', class_weight=weights)
        self.model.fit(self.X, self.A.ravel())
        return None


    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    def recommend(self, user_data, exploring=0):
        #print("Recommending")
        R = self.model.predict(user_data.reshape(1,-1))[0]
        self.obs_R = np.append(self.obs_R, R)
        return R

    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        self.X = np.append(self.X, self.scaler.transform(user.reshape(1,-1)), axis=0)
        self.A = np.append(self.A, action)
        self.Y = np.append(self.Y, outcome)
        # Store observations for final analysis
        super()._store_observation(user, action, outcome)
        # Check how much we have increased out dataset and update models if necessary
        if self.X.shape[0] > self.refit_trigger * self.data_in_model:
            self._fit_models()
        return None
