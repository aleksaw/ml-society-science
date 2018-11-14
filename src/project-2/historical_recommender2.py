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

class HistoricalRecommender2(Recommender):

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
        self.model = LogisticRegression(C=10^8, solver='sag').fit(self.X[:,-2:], self.A)
        return None


    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    def recommend(self, user_data):
        #print("Recommending")
        return self.model.predict(user_data.iloc[:,-2:])[0]
        #return 1*(np.random.uniform() < user_data.iloc[0,128] * 0.4  + user_data.iloc[0,129] * 0.5);

    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        self.X = np.append(self.X, self.scaler.transform(user.reshape(1,-1)))
        self.A = self.A.append(actions)
        self.Y = self.Y.append(outcome)
        # Check how much we have increased out dataset and update models if necessary
        if self.X.shape[0] > self.refit_trigger * self.data_in_model:
            self._fit_models(quick=self.quick_fits)
        return None
