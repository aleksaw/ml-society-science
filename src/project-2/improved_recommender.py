""" Improved Recommender """
import warnings
from sklearn.exceptions import DataConversionWarning
# We get useless DataConversionWarning from StandardScaler
warnings.filterwarnings("ignore", category=DataConversionWarning)

from recommender import Recommender

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

class ImprovedRecommender(Recommender):

    def __init__(self, n_actions: int, n_outcomes: int, quick_fits: bool=True,
                 refit_trigger: float=1.1) -> "Recommender":
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
        quick_fits : bool
            Whether to fit the models quickly or slowly.
            Defaults to True
        refit_trigger : float
            Increase over previous data in model before we refit.
            Defaults to 1.1

        Returns
        -------
        Recommender
            Initialized object
        """
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward
        self.scaler = StandardScaler()
        self.quick_fits = quick_fits
        self.refit_trigger = 1.1

    def fit_treatment_outcome(self, data: np.ndarray, actions: np.ndarray,
                              outcomes: np.ndarray, quick: bool=True) -> None:
        """ Fit a model from patient data, actions and their effects

        Here we assume that the outcome is a direct function of data and
        actions. This model can then be used in estimate_utility(),
        predict_proba() and recommend()

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
        self.X = self.scaler.fit_transform(data)
        self.model = [self._fit_model(a, quick=quick)
                      for a in range(self.n_actions)]

        return None

    def _fit_model(self, action: int, quick: bool=True) -> 'Classifier':
        """ Fit a model from patient data for a specific action

        We use pre-stored patient data to fit a suitable classifier for each
        action.
        If we have too few observations of each outcome we can't fit a
        Logistic Classifier, so we use one that supports is.

        Parameters
        ----------
        action: int
            The action to fit a model for
        quick: bool
            Whether or not to fit quickly.
            Defaults to True

        Returns
        ----------
        Classifier: A classifier for the action
        """
        self.data_in_model = self.X.shape[0]
        a = action

        # Set defaults and triggerpoints
        # Number of observations in each class needed to trigger Logistic
        bandit_logistic_trigger = 1
        # Number of observations needed to fit slowly if quick==False
        quick_trigger = 10
        # Regularization parameter to use in quick fitting
        quick_c = 100000000.

        # Settings for slow-fit, selecting amount of regularization
        # Ratio of smallest outcome class to all observations below which
        # we need to use custom accuracy score.
        acc_rr_threshold = 0.2
        # Number of observations needed to use 10-fold CV
        FiveFold_threshold = 200
        fits = 200 #n_repeats * n_folds
        # Regularization parameters to choose from
        Cs = np.logspace(-1, 6, 30)

        # Custom accuracy score for unbalanced outcome classes
        def acc_rr(y_pred, y_te):
            tp = (y_pred&yte).sum()
            fp = ((y_pred==1)&(y_te==0)).sum()
            fn = ((y_pred==0)&(y_te==1)).sum()
            # If there are no 1's in either array we get no result, but then it's a match.
            return (tp / (tp+fp+fn)) if (tp+fp+fn)>0 else 1

        class DummyClassifier:
            def __init__(self, probabilities):
                self.p = probabilities
            def predict_proba(self, data):
                return self.p

        X_A = self.X[(self.A==a),:]
        Y_A = self.Y[(self.A==a)]
        smallest_class_of_outcomes = np.min(np.bincount(Y_A, minlength=self.n_outcomes))

        if smallest_class_of_outcomes == 0:
            # We can't fit anything with only one class in data
            ps = np.append([1], np.zeros(self.n_outcomes-1))
            return DummyClassifier(ps)

        if smallest_class_of_outcomes < quick_trigger:
            # We're likely to run into trouble fitting classifier in
            # KFold, so we drop that.
            quick = True

        if quick:
            return LogisticRegression(C=quick_c, solver='lbfgs', max_iter=2000).fit(X_A, Y_A)

        if Y_A.sum() < acc_rr_threshold * len(Y_A):
            measure = acc_rr
        else:
            measure = accuracy_score

        n_folds = 10 if len(Y_A) > FiveFold_threshold else 5
        n_repeats = round(fits / n_folds)
        accuracies = np.zeros(len(Cs))
        for i, c in enumerate(Cs):
            accuracy = np.zeros(n_folds * n_repeats)
            j = 0
            rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats)
            for train, test in rskf.split(X=X_A, y=Y_A, groups=Y_A):
                Xtr, ytr = (X_A[train,], Y_A[train])
                Xte, yte = (X_A[test,], Y_A[test])
                y_pred = LogisticRegression(C=c, solver='lbfgs', max_iter=2000).fit(Xtr, ytr).predict(Xte)
                accuracy[j] = measure(y_pred, yte)
                j += 1
            accuracies[i] = np.mean(accuracy)

        # I'll go with highest accuracy. We could have gone most parsimonious
        # within 1se f.ex, but test indicate less regularization is better.
        best_c = Cs[np.argmax(accuracies)]
        return LogisticRegression(C=best_c, solver='sag').fit(X_A, Y_A)



    def predict_proba(self, user: np.ndarray, treatment: int) -> np.ndarray:
        """ Predict outcome probabilities

        Return a distribution of effects for a given person's data and a
        specific treatment.

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
        return self.model[treatment].predict_proba(user.reshape(1,-1))

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
        self.X = np.append(self.X, self.scaler.transform(user.reshape(1,-1)), axis=0)
        self.A = np.append(self.A, action)
        self.Y = np.append(self.Y, outcome)
        # Store observations for final analysis
        super()._store_observation(user, action, outcome)
        # Check how much we have increased out dataset and update models if necessary
        if self.X.shape[0] > self.refit_trigger * self.data_in_model:
            self.model = [self._fit_model(a, quick=self.quick_fits)
                          for a in range(self.n_actions)]
        return None
