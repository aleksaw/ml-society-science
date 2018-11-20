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

    def fit_treatment_outcome(self, data, actions, outcomes, quick=False,
                              load_historical_data=False):
        #print("Fitting treatment outcomes")
        if load_historical_data==False:
            super().fit_treatment_outcome(data, actions, outcomes)
        return None

    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    def recommend(self, user_data, exploring=0):
        # If user is in dataset, this should be max_features
        equal_elements = (self.X.values == user_data).sum(axis=1)
        # So if we knew that there were only one of each, and that we always
        # looked up that one, this could be skipped.
        max_equal_elements = equal_elements[np.argmax(equal_elements)]
        # Find historical users with identical data to the one
        users = self.X[equal_elements == max_equal_elements]
        # If there are more historical users with identically close fit
        # we choose randomly among them.
        R = np.random.choice(self.A[self.A.index.isin(users.index)].values.ravel())
        self.obs_R = np.append(self.obs_R, R)
        return R
