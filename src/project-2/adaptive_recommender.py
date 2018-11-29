""" 2 Adaptive Recommenders """

from recommender import Recommender

import numpy as np

class AdaptiveRecommender(Recommender):
    """ Adaptive Recommender estimating the W-matrix determining outcomes.

    It does this by counting up as a set of Dirichlet
    (or when n_outcomes=2, Beta) distributions the number of each outcome
    where each feature is present. We then center these so that they mostly
    say something about the importance of the features. The non-contextual
    probability of each outcome is modeled as a separate Dirichlet distribution.
    The estimated probability of each outcome is then a combination of
    the contextual probability and the non-contextual probability."""

    def __init__(self, n_actions: int, n_outcomes: int,
                 n_draws: int=100) -> Recommender:
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
        n_draws : int
            Number of draws from each distribution to predict probabilities
            Defaults to 100

        Returns
        -------
        Recommender
            Initialized object
        """
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.n_draws = n_draws
        self.reward = self._default_reward

    def fit_treatment_outcome(self, data: np.ndarray, actions: np.ndarray,
                              outcomes: np.ndarray, quick: bool=True) -> None:
        """ Fit a model from patient data, actions and their effects

        Counts up the part each feature has in each outcome

        Parameters
        ----------
        data : np.ndarray(n_observation, n_features)
            The historical data to find pattern in
        actions: np.ndarray(n_observations)
            The actions taken by historical recommender
        outcomes: np.ndarray(n_observations)
            The outcomes of those actions
        quick: bool
            Whether or not to fit quickly. No effect
            Defaults to True
        """
        #print("Fitting treatment outcomes")
        super().fit_treatment_outcome(data, actions, outcomes)
        # alphas of Dirichlet distributions
        self.w_alphas = np.ones((self.n_outcomes, self.n_actions, data.shape[1]))
        self.y_alphas = np.ones((self.n_outcomes, self.n_actions))
        for i in range(len(data)):
            # Scale down the count by the number of 1s. We don't know how many
            # or which features are responsible.
            self.w_alphas[outcomes[i], actions[i], :] += data[i] / data[i].sum()
            self.y_alphas[outcomes[i], actions[i]] += 1

        return None

    def predict_proba(self, user: np.ndarray, treatment: int) -> np.ndarray:
        """ Predict outcome probabilities

        Return a distribution of effects for a given person's data and a
        specific treatment.

        This is done by drawing samples from each underlying distribution
        estimating a W-entry, finding the most likely outcome for that sample
        and then using the sample estimates to create a probability distribution.

        Parameters
        ----------
        user : np.ndarray(n_features)
            Features of the user for whom to predict
        treatment: int
            The actions to take

        Returns
        ---------
        np.ndarray(n_outcomes)
            The probability of each outcome given the data and the treatment
        """
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
        p_ish_y_ax = np.dot(samples.T, user.ravel()).T * p_y_a
        # For each sample, choose the most likely outcome
        y_hats = np.argmax(p_ish_y_ax, axis=1)
        # See how many of each outcome, and convert to probabilities
        return np.bincount(y_hats, minlength=self.n_outcomes) / self.n_draws

    def observe(self, user: np.ndarray, action: int, outcome: int) -> None:
        """ Observe the effect of an action.

        This is an opportunity for the recommender to refit the models, to
        take the new information into account.

        Parameters
        ----------
        user : np.ndarray(n_features)
            Features of the user
        action : int
            Action taken
        outcome: int
            Outcome observed
        """
        super().observe(user, action, outcome)
        # Scale down the count by the number of 1s. We don't know how many
        # or which features are responsible.
        self.w_alphas[outcome, action, :] += user / user.sum()
        self.y_alphas[outcome, action] += 1
        return None



class AdaptiveRecommender2(Recommender):

    def __init__(self, n_actions: int, n_outcomes: int,
                 n_draws: int=100) -> Recommender:
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
        n_draws : int
            Number of draws from each distribution to predict probabilities.
            Has no effect at the moment, but might be used like in #1
            Defaults to 100

        Returns
        -------
        Recommender
            Initialized object
        """
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.n_draws = n_draws
        self.reward = self._default_reward

    def fit_treatment_outcome(self, data: np.ndarray, actions: np.ndarray,
                              outcomes: np.ndarray, quick: bool=True) -> None:
        """ Fit a model from patient data, actions and their effects

        Counts up the number of each outcome feach feature is involved in per
        action as well as the total number of each outcome.

        Parameters
        ----------
        data : np.ndarray(n_observation, n_features)
            The historical data to find pattern in
        actions: np.ndarray(n_observations)
            The actions taken by historical recommender
        outcomes: np.ndarray(n_observations)
            The outcomes of those actions
        quick: bool
            Whether or not to fit quickly. Ineffective
            Defaults to True
        """
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

    def predict_proba(self, user: np.ndarray, treatment: int) -> np.ndarray:
        """ Predict outcome probabilities

        Return a distribution of effects for a given person's data and a
        specific treatment.
        Attempts to model x_i|y,a ~ Dirichlet with each x_i being a feature of
        one user with prior all 1s.
        Then attempts to model y|a ~ Dirichlet with all 1s as prior.
        Then calculates p(x|y,a) = \prod_{features} p(x_i|y,a)I(x_i \in user)
        From this we get p(y|x,a) \propto p(y|a) * p(x|y,a)
        There is one major problem with this approach, in that this equality
        p(x|y,a) = \prod_{features} p(x_i|y,a)I(x_i \in user)
        doesn't hold when the features are correlated.
        One way around this may be reducing the feature space so we can model for
        each possible combination, but then to get the same (high!) amount
        of estimates x_is we would have to cut down to 7 features (2^7=128).
        This largely defeats the purpose of this class, which is to be used
        in the low-data setting where we have gene-targeting treatments. It
        can then never capture treatments targetting genes we have excluded as
        features.
        Another approach is like the other adaptive recommender, to sample from
        the distributions instead of using the means and trying to use this to
        rectify the problem, but it doesn't change the underlying issue.

        Parameters
        ----------
        user : np.ndarray(n_features)
            Features of the user for whom to predict
        treatment: int
            The actions to take

        Returns
        ---------
        np.ndarray(n_outcomes)
            The probability of each outcome given the data and the treatment
        """
        a = treatment
        log_x_means = self.x_alphas[:,a,:] / self.x_alphas[:,a,:].sum(axis=0)
        log_p_x_ya = np.dot(np.log(log_x_means), user)
        log_y_a = np.log(self.y_alphas[:,a] / self.y_alphas[:,a].sum(axis=0))
        p_y_xa = np.exp(log_p_x_ya + log_y_a - log_p_x_ya.mean())
        return p_y_xa / p_y_xa.sum()

    def observe(self, user: np.ndarray, action: int, outcome: int) -> None:
        """ Observe the effect of an action.

        This is an opportunity for the recommender to refit the models, to
        take the new information into account.

        Parameters
        ----------
        user : np.ndarray(n_features)
            Features of the user
        action : int
            Action taken
        outcome: int
            Outcome observed
        """
        super().observe(user, action, outcome)
        self.x_alphas[outcome, action, :] += user
        self.y_alphas[outcome, action] += 1
        return None


from sklearn.decomposition import PCA

class AdaptiveRecommender3(Recommender):

    def __init__(self, n_actions: int, n_outcomes: int,
                 n_bins: int=30, refit_trigger: float=1.1) -> Recommender:
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
        n_bins : int
            Number of bins from each distribution of xs.
            Defaults to 30

        Returns
        -------
        Recommender
            Initialized object
        """
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.n_bins = n_bins
        self.refit_trigger = refit_trigger
        self.reward = self._default_reward
        self.pca = PCA()


    def fit_data(self, data: np.ndarray) -> None:
        """ Fit a model from patient data.

        Runs a Principal Component Analysis on the data

        Parameters
        ----------
        data : np.ndarray(n_observation, n_features)
            The historical data to find pattern in
        """
        #print("Preprocessing data")
        super().fit_data(data)
        self.Xpca = self.pca.fit_transform(data)
        return None

    def fit_treatment_outcome(self, data: np.ndarray, actions: np.ndarray,
                              outcomes: np.ndarray, quick: bool=True) -> None:
        """ Fit a model from patient data, actions and their effects

        Counts up the number of each outcome feach feature is involved in per
        action as well as the total number of each outcome.

        Parameters
        ----------
        data : np.ndarray(n_observation, n_features)
            The historical data to find pattern in
        actions: np.ndarray(n_observations)
            The actions taken by historical recommender
        outcomes: np.ndarray(n_observations)
            The outcomes of those actions
        quick: bool
            Whether or not to fit quickly. Ineffective
            Defaults to True
        """
        #print("Fitting treatment outcomes")
        super().fit_treatment_outcome(data, actions, outcomes)
        self.fit_data(data)

        self.data_in_model = np.zeros(self.n_actions)
        self.bin_edges = np.empty((self.n_actions, self.n_outcomes, data.shape[1], self.n_bins+1))
        self.density = np.ones((self.n_actions, self.n_outcomes, data.shape[1], self.n_bins+1))
        self.y_alphas = np.ones((self.n_actions, self.n_outcomes))
        for a in range(self.n_actions):
            self._fit_model(a)

        return None


    def _fit_model(self, action):
        a = action
        self.data_in_model[a] = self.X[self.A==a].shape[0]
        for o in range(self.n_outcomes):
            mask = (self.A==a).ravel() & (self.Y==o).ravel()
            self.y_alphas[a,o] += mask.sum()
            Xp = self.Xpca[mask,:]
            for i in range(Xp.shape[1]):
                hist = np.histogram(Xp[:,i], bins=self.n_bins)
                self.density[a,o,i,1:] += hist[0]
                self.bin_edges[a,o,i,:] = hist[1]
                self.density[a,o,i,:] /= self.density[a,o,i,:].sum()

    def predict_proba(self, user: np.ndarray, treatment: int) -> np.ndarray:
        """ Predict outcome probabilities

        Return a distribution of effects for a given person's data and a
        specific treatment.
        Attempts to model x_i|y,a ~ Dirichlet with each x_i being a feature of
        one user with prior all 1s.
        Then attempts to model y|a ~ Dirichlet with all 1s as prior.
        Then calculates p(x|y,a) = \prod_{features} p(x_i|y,a)I(x_i \in user)
        From this we get p(y|x,a) \propto p(y|a) * p(x|y,a)
        There is one major problem with this approach, in that this equality
        p(x|y,a) = \prod_{features} p(x_i|y,a)I(x_i \in user)
        doesn't hold when the features are correlated.
        One way around this may be reducing the feature space so we can model for
        each possible combination, but then to get the same (high!) amount
        of estimates x_is we would have to cut down to 7 features (2^7=128).
        This largely defeats the purpose of this class, which is to be used
        in the low-data setting where we have gene-targeting treatments. It
        can then never capture treatments targetting genes we have excluded as
        features.
        Another approach is like the other adaptive recommender, to sample from
        the distributions instead of using the means and trying to use this to
        rectify the problem, but it doesn't change the underlying issue.

        Parameters
        ----------
        user : np.ndarray(n_features)
            Features of the user for whom to predict
        treatment: int
            The actions to take

        Returns
        ---------
        np.ndarray(n_outcomes)
            The probability of each outcome given the data and the treatment
        """
        a = treatment
        user = self.pca.transform(user.reshape(1,-1)).ravel()
        log_p_x_ya = np.zeros(self.n_outcomes)
        for o in range(self.n_outcomes):
            for i in range(self.bin_edges.shape[2]):
                bin_ = np.argmax(1*(self.bin_edges[a,o,i,:] >= user[i]))
                log_p_x_ya[o] += np.log(self.density[a,o,i,bin_])

        log_y_a = np.log(self.y_alphas[a,:] / self.y_alphas[a,:].sum(axis=0))
        p_y_xa = np.exp(log_p_x_ya + log_y_a)
        return p_y_xa / p_y_xa.sum()

    def observe(self, user: np.ndarray, action: int, outcome: int) -> None:
        """ Observe the effect of an action.

        This is an opportunity for the recommender to refit the models, to
        take the new information into account.

        Parameters
        ----------
        user : np.ndarray(n_features)
            Features of the user
        action : int
            Action taken
        outcome: int
            Outcome observed
        """
        super().observe(user, action, outcome)
        data_for_model = self.X[self.A==action].shape[0]
        if data_for_model > self.refit_trigger * self.data_in_model[action]:
            self.fit_data(self.X)
            self._fit_model(action)
        # How do we actually use this for Thompson sampling, though?
        self.y_alphas[action, outcome] += 1
        return None
