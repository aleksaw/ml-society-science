import warnings
from sklearn.exceptions import DataConversionWarning
# We get useless DataConversionWarning from StandardScaler
warnings.filterwarnings("ignore")#, category=DataConversionWarning)
# We get a DeprecationWarning initially
#warnings.filterwarnings("ignore", category=warnings.DeprecationWarning)

import numpy as np
import pandas
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from cycler import cycler

def default_reward_function(action, outcome):
    return outcome-0.1*(action>0)

def test_policy(generator, policy, reward_function, T):
    #print("Testing for ", T, "steps")
    policy.set_reward(reward_function)
    u = 0
    for t in range(T):
        x = generator.generate_features()
        a = policy.recommend(x)
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
policy_factories = [RandomRecommender,
                    #HistoricalRecommender2,
                    #HistoricalRecommender3,
                    #OptimisticRecommender,
                    #HomeopathicRecommender,
                    #ImprovedRecommender,
                    #AdaptiveRecommender,
                    AdaptiveRecommender2,
                    #AdaptiveRecommender3,
                    ImprovedAdaptiveRecommender
                    ]

policy_names = ['Random',
                #'Historical2',
                #'Historical3',
                #'Optimistic',
                #'Homeopathic',
                #'Improved',
                #'Adaptive',
                'Adaptive2',
                #'Adaptive3',
                'Improved Adaptive'
                ]
#import reference_recommender
#policy_factory = reference_recommender.RandomRecommender
n_tests = 10000
boots = 100
policy_mean_total_utility = np.empty((2, len(policy_factories), boots, n_tests))
## First test with the same number of treatments
n_treatments =(2, 129)
generator = [data_generation.DataGenerator(matrices="./generating_matrices.mat"),
             data_generation.DataGenerator(matrices="./big_generating_matrices.mat")]
titles = ["---- Testing with only two treatments ----",
          "--- Testing with an additional experimental treatment and 126 gene silencing treatments ---"]
for j, treatments in enumerate(n_treatments):
    for i, policy_factory in enumerate(policy_factories):
        successes = np.empty((boots, treatments, n_tests))
        treated = np.empty((boots, treatments, n_tests))
        mean_utility = np.empty((boots, treatments, n_tests))
        exploration = np.empty((boots, treatments, n_tests))
        exploit_success_rate = np.empty((boots, treatments, n_tests))
        success_rate = np.empty((boots, treatments, n_tests))
        mean_utility = np.empty((boots, treatments, n_tests))
        mean_total_utility = np.empty((boots, n_tests))
        print('----------{} Recommender --------'.format(policy_names[i]))
        print(titles[j])

        print("Setting up simulator")
        print("Setting up policy")

        ## Run an online test with a small number of actions
        print("Running online tests")
        for n_boot in tqdm(range(boots)):
            policy = policy_factory(generator[j].get_n_actions(), generator[j].get_n_outcomes())
            policy.fit_treatment_outcome(features, actions, outcome)
            # print('Number of tests: {}'.format(int(n_tests)))
            result = test_policy(generator[j], policy, default_reward_function, int(n_tests))
            # print("Average reward: {:.3f}".format(result/n _tests))
            # print("Final analysis of results")
            dict = policy.final_analysis(quiet=True)
            successes[n_boot,:,:] = dict['successes']
            treated[n_boot,:,:] = dict['actions']
            exploration[n_boot,:,:] = dict['exploration']
            exploit_success_rate[n_boot,:,:] = dict['exploit_success_rate']
            success_rate[n_boot,:,:] = dict['success_rate']
            mean_utility[n_boot,:,:] = dict['mean_utility']
            mean_total_utility[n_boot,:] = dict['mean_total_utility']
            policy_mean_total_utility[j,i,n_boot,:] = dict['mean_total_utility']

        # Save data
        to_save = {'successes': successes, 'treated': treated,
                   'exploration': exploration,
                   'exploit_success_rate': exploit_success_rate,
                   'success_rate': success_rate, 'mean_utility': mean_utility,
                   'mean_total_utility': mean_total_utility}
        np.savez_compressed(f"{policy_names[i]}_treatment{j}_{n_tests}_{nboots}", **to_save)

        # Plot graphs
        epochs = list(range(n_tests))
        data = [successes, treated, exploration, success_rate,
                exploit_success_rate, mean_utility]
        log_scale = [True, True, True, False, False, False]
        plot_title = ['Successes for each treatment',
                      'Actions for each treatment',
                      'Actions taken for exploration',
                      'Rate of success of treatments',
                      'Rate of success of treatment when exploiting',
                      'Mean utility for treatment']
        action_labels = ['_' for _ in range(min(3,treatments))]
        action_labels[0] = 'Placebo'
        action_labels[1] = 'Old'
        if treatments > 2:  action_labels[2] = 'New'

        plt.rcParams['figure.figsize'] = [20, 14]
        col1 = sns.color_palette("bright", 4)[1:]
        col2 = sns.color_palette("Blues_d", treatments-3)
        plt.rcParams['axes.prop_cycle'] = cycler(color=(col1+col2))
        fig, axs = plt.subplots(4, 2)
        for k in range(len(data)):
            r = k % 3;      c = k // 3
            means = data[k].mean(axis=0)
            quantiles = np.quantile(data[k], q=(0.05, 0.95), axis=0)
            axs[r,c].plot(epochs, means.T)
            for l in range(treatments):
                axs[r,c].fill_between(epochs, quantiles[0,l,:], quantiles[1,l,:], alpha=0.2)
            axs[r,c].legend(action_labels)
            axs[r,c].set_title(plot_title[k])
            if log_scale[k]:
                axs[r,c].set_yscale('log')
                axs[r,c].set_ylim(bottom=0.3)
        axs[3,1].plot(epochs, mean_total_utility.mean(axis=0))
        quantiles = np.quantile(mean_total_utility, q=(0.05, 0.95), axis=0)
        axs[3,1].fill_between(epochs, quantiles[0,:], quantiles[1,:], alpha=0.2)
        axs[3,1].set_title('Mean total utility over time')
        fig.suptitle(policy_names[i], fontsize=16, y=1.)
        plt.tight_layout()
        plt.savefig(f'analysis_{policy_names[i]}_treatment{j}_{n_tests}_{nboots}.png')

np.savez("finalfull", policy_mean_total_utility=policy_mean_total_utility)

plt.rcParams['figure.figsize'] = [15, 10]
col = sns.color_palette("husl", len(policy_names))
plt.rcParams['axes.prop_cycle'] = cycler(color=col)
fig, axs = plt.subplots(2,1)
means = policy_mean_total_utility.mean(axis=2)
for i in range(len(n_treatments)):
    quantiles = np.quantile(policy_mean_total_utility, q=(0.05, 0.95), axis=2)
    axs[i].plot(epochs, means[i,:,:].T)
    for j in range(len(policy_names)):
        axs[i].fill_between(epochs, quantiles[0,i,j,:], quantiles[1,i,j,:], alpha=0.2)
    axs[i].legend(policy_names)
    axs[i].set_title(f'{n_treatments[i]} treatments available')
fig.suptitle('Policy comparison', fontsize=16, y=1.)
plt.savefig('final_analysis.png')
