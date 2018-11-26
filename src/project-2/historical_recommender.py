""" 3 Historical Recommenders """

from recommender import Recommender

import numpy as np
from sklearn.linear_model import LogisticRegression

class HistoricalRecommender(Recommender):
    """ A historical recommender always taking the historical action

    This recommender is not suitable for making recommendations on new data as it
    just looks up the action taken on the historical patient.
    """

    def fit_treatment_outcome(self, data: np.ndarray, actions: np.ndarray,
                              outcomes: np.ndarray, quick: bool=True,
                              load_historical_data: bool=False) -> None:
        """ Fit a model from patient data, actions and their effects

        This method does nothing unless load_historical_data is True. Thus
        if preloaded with data it works even when fitted with partial data
        during CV estimation.

        Parameters
        ----------
        data : np.ndarray(n_observation, n_features)
            The historical data to find pattern in
        actions: np.ndarray(n_observations)
            The actions taken by historical recommender
        outcomes: np.ndarray(n_observations)
            The outcomes of those actions
        quick: bool
            Whether or not to fit quickly. Not always possible.
            Defaults to True
        load_historical_data: bool
            Whether ornot to load in the historical data. Explanation above
            Defaults to False
        """
        #print("Fitting treatment outcomes")
        if load_historical_data==False:
            super().fit_treatment_outcome(data, actions, outcomes)
        return None

    def recommend(self, user_data: np.ndarray, exploring: float=0.1) -> int:
        """ The policy's recommendation for a specific user

        Return recommendation for a specific user datum.
        Exploring is ineffective. Always takes the historical action if
        available. Else the action given to the historical person with
        the largest number of identical features.

        Parameters
        ----------
        user_data : np.ndarray(n_features)
            Features of the user for whom to recommend
        exploring : float
            The probability of exploring vs exploiting. Range [0,1]

        Returns
        ---------
        int
            The recommended action, in range[0, n_actions]
        """
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



class HistoricalRecommender2(Recommender):
    """ A historical recommender mimicking the historical decisions

    This recommender cheats and uses code from data_generation"""

    def recommend(self, user_data: np.ndarray, exploring: float=0.1) -> int:
        """ The policy's recommendation for a specific user

        Return recommendation for a specific user datum. Exploring ineffective.
        Uses code from data_generator to recommend action.

        Parameters
        ----------
        user_data : np.ndarray(n_features)
            Features of the user for whom to recommend
        exploring : float
            The probability of exploring vs exploiting. Range [0,1]

        Returns
        ---------
        int
            The recommended action, in range[0, n_actions]
        """
        #print("Recommending")
        R = 1*(np.random.uniform() < user_data[128] * 0.4  + user_data[129] * 0.5);
        self.obs_R = np.append(self.obs_R, R)
        return R



class HistoricalRecommender3(Recommender):
    """ A historical recommender trying to mimick the historical decisions """

    def __init__(self, n_actions: int, n_outcomes: int,
                     refit_trigger: float=1.1) -> Recommender:
        """ Initialize the class

        Set the recommender with a default number of actions and outcomes.
        This is because the number of actions in historical data can be
        different from the ones that you can take with your policy.

        Parameters
        ----------
        n_actions : int
            Number of possible actions to take
        n_outcomes : int
            Number of possible outcomes
        refit_trigger: float
            A number specifying the increase in data before we refit the model

        Returns
        -------
        Recommender
            Initialized object
        """
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward
        self.refit_trigger = 1.1

    def fit_treatment_outcome(self, data: np.ndarray, actions: np.ndarray,
                              outcomes: np.ndarray, quick: bool=True) -> None:
        """ Fit a model from patient data, actions and their effects

        This historical recommender fits the data to the historical actions
        to try to mimick those.

        Parameters
        ----------
        data : np.ndarray(n_observation, n_features)
            The historical data to find pattern in
        actions: np.ndarray(n_observations)
            The actions taken by historical recommender
        outcomes: np.ndarray(n_observations)
            The outcomes of those actions
        quick: bool
            Whether or not to fit quickly. Not always possible.
            Defaults to True
        """
        #print("Fitting treatment outcomes")
        super().fit_treatment_outcome(data, actions, outcomes)
        self._fit_models(quick=quick)

        return None

    ## Fit a model to the data we have
    def _fit_models(self, quick: bool=False) -> None:
        """ Fit a model to mimick historical actions

        This historical recommender fits the data to the historical actions
        to try to mimick those.

        Parameters
        ----------
        quick: bool
            Whether or not to fit quickly. No effect here
            Defaults to True
        """
        # We have to find some weights that balance the fact that we have
        # more placebo than treatment actions in history.
        weights = {0: np.mean(self.A)*2, 1: (1-np.mean(self.A))}
        self.model = LogisticRegression(C=100000., solver='lbfgs', class_weight=weights)
        self.model.fit(self.X, self.A.ravel())
        return None


    def recommend(self, user_data: np.ndarray, exploring: float=0.1) -> int:
        """ The policy's recommendation for a specific user

        Return recommendation for a specific user datum. Exploring is
        ineffective.

        Parameters
        ----------
        user_data : np.ndarray(n_features)
            Features of the user for whom to recommend
        exploring : float
            The probability of exploring vs exploiting. Range [0,1]

        Returns
        ---------
        int
            The recommended action, in range[0, n_actions]
        """
        #print("Recommending")
        R = self.model.predict(user_data.reshape(1,-1))[0]
        self.obs_R = np.append(self.obs_R, R)
        return R
