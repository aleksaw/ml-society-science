import numpy as np
import pandas
from tqdm import tqdm

def default_reward_function(action, outcome):
    return outcome-0.1*(action>0)

def test_policy(generator, policy, reward_function, T):
    print("Testing for ", T, "steps")
    policy.set_reward(reward_function)
    u = 0
    for t in tqdm(range(T)):
        x = generator.generate_features()
        a = policy.recommend(x, strategy='backinduction')
        y = generator.generate_outcome(x, a)
        r = reward_function(a, y)
        u += r
        policy.observe(x, a, y)
        #print(a)
        #print("x: ", x, "a: ", a, "y:", y, "r:", r)
    return u

features = pandas.read_csv('data/medical/historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv('data/medical/historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv('data/medical/historical_Y.dat', header=None, sep=" ").values
observations = features[:, :128]
labels = features[:,128] + features[:,129]*2

import data_generation
from noncontextual_recommenders import RandomRecommender, OptimisticRecommender, HomeopathicRecommender
from historical_recommender import HistoricalRecommender, HistoricalRecommender2, HistoricalRecommender3
from adaptive_recommender import AdaptiveRecommender, AdaptiveRecommender2, AdaptiveRecommender3
from improved_recommender import ImprovedRecommender
from improved_adaptive_recommender import ImprovedAdaptiveRecommender
policy_factories = [#RandomRecommender,
                    #HistoricalRecommender2,
                    #HistoricalRecommender3,
                    #OptimisticRecommender,
                    #HomeopathicRecommender,
                    #ImprovedRecommender,
                    #AdaptiveRecommender,
                    #AdaptiveRecommender2,
                    #AdaptiveRecommender3,
                    ImprovedAdaptiveRecommender]

policy_names = [#'Random', 'Historical2', 'Historical3', 'Optimistic',
                #'Homeopathic', 'Improved','Adaptive', 'Adaptive2', 'Adaptive3',
                'Improved Adaptive']
#import reference_recommender
#policy_factory = reference_recommender.RandomRecommender

## First test with the same number of treatments
for i, policy_factory in enumerate(policy_factories):
    print('----------{} Recommender --------'.format(policy_names[i]))
    print("---- Testing with only two treatments ----")

    print("Setting up simulator")
    generator = data_generation.DataGenerator(matrices="./generating_matrices.mat")
    print("Setting up policy")
    policy = policy_factory(generator.get_n_actions(), generator.get_n_outcomes())
    ## Fit the policy on historical data first
    print("Fitting historical data to the policy")
    policy.fit_treatment_outcome(features, actions, outcome)
    ## Run an online test with a small number of actions
    print("Running online tests")
    """
    for n_tests in np.logspace(2, 4.3, 5):
        print('Number of tests: {}'.format(int(n_tests)))
        result = test_policy(generator, policy, default_reward_function, int(n_tests))
        print("Average reward: {:.3f}".format(result/n_tests))
        print("Final analysis of results")
        policy.final_analysis()"""

    ## First test with the same number of treatments
    print("--- Testing with an additional experimental treatment and 126 gene silencing treatments ---")
    print("Setting up simulator")
    generator = data_generation.DataGenerator(matrices="./big_generating_matrices.mat")
    print("Setting up policy")
    policy = policy_factory(generator.get_n_actions(), generator.get_n_outcomes())
    ## Fit the policy on historical data first
    print("Fitting historical data to the policy")
    policy.fit_treatment_outcome(features, actions, outcome)
    ## Run an online test with a small number of actions
    print("Running online tests")
    for n_tests in np.logspace(2, 4.3, 5):
        print('Number of tests: {}'.format(int(n_tests)))
        result = test_policy(generator, policy, default_reward_function, int(n_tests))
        print("Average reward: {:.3f}".format(result/n_tests))
        print("Final analysis of results")
        policy.final_analysis()
