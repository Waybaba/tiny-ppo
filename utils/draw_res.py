import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



def load(path):
    return np.load(path, allow_pickle=True)

envs = ['HalfCheetah-v2', 'Hopper-v2', 'InvertedDoublePendulum-v2',
'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2',
'Walker2d-v2']
folder = './data/res/'
for env in envs:
    res = load(folder+env+'.npy').item()

    # sns.set_theme(style="darkgrid")

    # # Load an example dataset with long-form data
    # x = np.linspace(0, 15, 31)
    # data = np.sin(x) + np.random.rand(10, 31) + np.random.randn(10, 1)
    # df = pd.DataFrame(data).melt()

    # # Plot the responses for different events and regions
    # sns.lineplot(x="timepoint", y="signal",
    #             hue="region", style="event",
    #             data=df)
    # plt.savefig('./output/'+env+'.png')
    # print('finish')



    # df = experience.df
    # data = experience.total_reward_list
    # episode = list(range(len(data)))
    # # print(episode)
    # data = pd.DataFrame(zip(*[data, episode]), columns=['Reward',"Episode"])
    # # print(data)
    # sns.lineplot(x="Episode", y="Reward",
    # #              hue="region", style="event",
    #              data=data)
    # reward_history = experience.total_reward_list


    reward_history = [each[0] for each in res['r_trans']]
    reward_epis = [each[1] for each in res['r_trans']]
    episodes = len(reward_history)
    window = int(episodes/50)
    # fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[4, 4])
    fig, ((ax1)) = plt.subplots(1, 1, sharey=True, figsize=[6, 5])

    
    rolling_mean = pd.Series(reward_history).rolling(window).mean() # rolling是滑动取数，会将附近的n个数放到一起，一般会取均值或者方差
    std = pd.Series(reward_history).rolling(window).std()
    # ax1.plot(rolling_mean, color='#8172B2')
    ax1.plot(reward_epis, rolling_mean, color='#8172B2')
    ax1.fill_between(
        reward_epis, rolling_mean -  std,
        rolling_mean+std, color='#A192D2', alpha=0.2) # 这里会画出来一个区域
    ax1.set_title(env)
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    plt.xticks(np.arange(0, 1000001, 1000000))
    plt.locator_params(axis='y', nbins=6)
    ax1.get_xaxis().get_major_formatter().set_scientific(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    # ax2.plot(reward_history)
    # ax2.set_title('Episode Length')
    # ax2.set_xlabel('Episode')
    # ax2.set_ylabel('Episode Length')

    fig.tight_layout(pad=2)
    ax1.set_facecolor('#f0f0f0')
    plt.grid(axis = 'y', color='white')
    plt.savefig('./output/'+env+'.png')
    print('Saved: '+'./output/'+env+'.png')
    plt.clf() 