import numpy as np
import pandas
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from cycler import cycler


policy_names = ['Random',
                #'Historical2',
                #'Historical3',
                #'Optimistic',
                #'Homeopathic',
                #'Improved',
                #'Adaptive',
                #'Adaptive2',
                #'Adaptive3',
                #'Improved Adaptive'
                ]

## First test with the same number of treatments
n_tests = 2000
boots = 100
n_treatments = (2, 129)
titles = ["---- Testing with only two treatments ----",
          "--- Testing with an additional experimental treatment and 126 gene silencing treatments ---"]
for j, treatments in enumerate(n_treatments):
    for i, policy_name in enumerate(policy_names):
        data = np.load(f"policy{i}_treatment{j}.npz")
        successes = data['successes']
        treated = data['treated']
        mean_utility = data['mean_utility']
        exploration = data['exploration']
        exploit_success_rate = data['exploit_success_rate']
        success_rate = data['success_rate']
        mean_utility = data['mean_utility']
        mean_total_utility = data['mean_total_utility']

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
        plt.savefig(f'analysis_policy{i}_treatment{j}.png')

policy_mean_total_utility = np.load(f"final.npz")['policy_mean_total_utility']

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
