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

class HistoricalRecommender(Recommender):

    def fit_treatment_outcome(self, data, actions, outcomes, quick=False):
        #print("Fitting treatment outcomes")
        if quick==False:
            self.X = data
            self.A = actions
            self.Y = outcomes
        return None

    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    def recommend(self, user_data):
        # If user is in dataset, this should be max_features
        #print(self.X.columns)
        #print(user_data.columns)
        #print(user_data)
        #print(user_data.shape)
        #print(self.X.shape)
        #print(self.X == user_data)
        equal_elements = (self.X.values == user_data.values).sum(axis=1)
        # So if we knew that there were only one of each, and that we always
        # looked up that one, this could be skipped.
        max_equal_elements = equal_elements[np.argmax(equal_elements)]
        # Find historical users with identical data to the one
        users = self.X[equal_elements == max_equal_elements]
        # If there are more historical users with identically close fit
        # we choose randomly among them.
        return np.random.choice(self.A[self.A.index.isin(users.index)].values.ravel())

    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        self.data = np.append(self.data)
        self.actions = self.actions.append(actions)
        self.outcomes = self.outcomes.append(outcome)
        return None