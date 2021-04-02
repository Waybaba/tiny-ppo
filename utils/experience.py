import gym
import random
import numpy as np
import pandas as pd
import itertools

"""
    shorthand transition as trans
    shorthand episode as epis
    extra is a dict
    gain is stored in extra with key name as 'gain'
"""


class Transition(object):
    def __init__(self, s, a, r, s_, is_done, extra=None):
        if extra is None:
            extra = {}
        self.data = [s, a, r, s_, is_done, extra]

    '''utils'''
    @property
    def s(self):
        return self.data[0]

    @property
    def a(self):
        return self.data[1]

    @property
    def r(self):
        return self.data[2]

    @property
    def s_(self):
        return self.data[3]

    @property
    def is_done(self):
        return self.data[4]

    @property
    def extra(self):
        return self.data[-1]

    def __iter__(self):
        return iter(self.data)

    def __str__(self):
        return "Transition: \r\n\tstate: {0}\r\n\taction: {1}\r\n\treward: {2}\r\n\tstate_: {3}\r\n\tis_done: {4}\r\n".\
            format(self.s, self.a, self.r, self.s_, self.is_done)

    def __getitem__(self, index):
        return self.trans_list[index]

    def __len__(self):
        return len(self.data)


class Episode(object):
    def __init__(self):
        self.trans_list = []
        # self.name = str(e_id)
    '''core'''

    def push(self, trans):
        '''store transition and calculate total reward'''
        self.trans_list.append(trans)
        return True

    def sample(self, size=1):
        return random.sample(self.trans_list, size)

    def calculate_gain(self, gamma, norm=True):
        running = 0.
        for i in range(len(self))[::-1]:
            running = self[i].r + gamma * running
            self.trans_list[i].extra['gain'] = running
            # print(running)
        if norm:
            gains = np.array([each.extra['gain'] for each in self.trans_list])
            gains = (gains - gains.mean()) / (gains.std() + 1e-5)
            for i in range(len(self)):
                self.trans_list[i].extra['gain'] = gains[i]

    def calculate_gae(self, gamma, lambda_, norm=True):
        running = 0.
        prev_return = 0.
        for i in range(len(self))[::-1]:
            pass
            ### TODO calculate gae here
            # self.trans_list[i].extra['gain'] = self[i].r + gamma * prev_return
            # prev_return = self.trans_list[i].extra['gain']

        if norm:
            pass
            ### TODO normalize gae here
            # gains = np.array([each.extra['gain'] for each in self.trans_list])
            # gains = (gains - gains.mean()) / (gains.std() + 1e-5)
            # for i in range(len(self)):
            #     self.trans_list[i].extra['gain'] = gains[i]

            ### GAE in PPO code for reference
            # returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
            # deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values[i]
            # # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
            # advantages[i] = deltas[i] + args.gamma * args.lamda * prev_advantage * masks[i]

            # prev_return = returns[i]
            # prev_value = values[i]
            # prev_advantage = advantages[i]

    '''utils'''

    def __str__(self):
        return "Episode:\r\n\tlen: {0}\r\n\ttotal_reward: {1}\r\n\treward_list: {2}\r\n### Transition table: \r\n{3}".\
            format(
                len(self),
                self.total_reward,
                self.reward_list,
                # self.trans_list[-1] if self.trans_list else "None"
                self.df.__str__()
            )
        return

    def __len__(self):
        return len(self.trans_list)

    def __iter__(self):
        return iter(self.trans_list)

    def __getitem__(self, index):
        return self.trans_list[index]

    @property
    def is_done(self):
        if len(self) == 0:
            return False
        return self.trans_list[-1].is_done

    @property
    def reward_list(self):
        return [each.r for each in self.trans_list]

    @property
    def total_reward(self):
        if len(self) == 0:
            return 0
        return sum(self.reward_list)

    @property
    def df(self):
        tmp = self.trans_list

        df = pd.DataFrame(
            data=tmp,
            columns=["state", "action", "reward", "state_", "is_done", "extra"]
        )
        return df


class Experience():

    """ Full memory of agent

    Full memory of agent with each in it as Episode. Default processed with transition.

    Attributes:
        --- core
        push_epis: 
        push_trans:
        sample_epis:
        sample_trans:
        info: 
        detail: 
        head: return pd.DataFrame()
        --- property
        @df: 
        @total_reward_list: 
        @epis_len_list: 
        @total_reward_mean: 
        @epis_len_mean: 
        --- utils
        pop: 

    """

    def __init__(self, capacity):
        self.capacity = capacity  # epis capacity instead of trans
        self.epis_list = []
        self.log_list = []
        self.log_key = ['reward']
        

    '''core'''

    def push_epis(self, epis):
        raise NotImplementedError
        if self.capacity <= 0:
            exit('Capacity can not be negtive.')
        while len(self) > self.capacity:
            self.epis_list.pop(0)
        self.epis_list.append(epis)

    def push_trans(self, trans, epis_start=None, custom_log=False):
        if self.capacity <= 0:
            exit('Capacity can not be negtive.')
        while len(self) >= self.capacity:
            episode = self.epis_list.pop(0)

        if epis_start is not None:  # not assigned, assert according to end of last
            if epis_start:
                self.epis_list.append(Episode())
                self.log_list.append([])
            self.epis_list[-1].push(trans)
            self.log_append(custom_log, trans.r)
            
        else:  # assigned, assert according to epis_end
            if len(self) == 0 or self.epis_list[-1].is_done:
                self.epis_list.append(Episode())
                self.log_list.append([])
            self.epis_list[-1].push(trans)
            self.log_append(custom_log, trans.r)

        return self.epis_list[-1][-1]

    def sample_epis(self, size=1):
        return random.sample(self.epis_list, k=size)

    def sample_trans(self, size=1):
        sample_trans = []
        for _ in range(size):
            index = int(random.random() * len(self))
            sample_trans += self.epis_list[index].sample()
        return sample_trans

    def clear(self):
        self.epis_list = []
    
    def calculate_gain(self, gamma, norm=True):
        for each in self:
            each.calculate_gain(gamma, norm)

    def calculate_gae(self, gamma, lambda_, norm=True):
        for each in self:
            each.calculate_gain(gamma, lambda_, norm)

    def save(self, path, print_log=False):
        tmp = np.array(self.epis_list, dtype=object)
        np.save(path, tmp, allow_pickle=True)
        if print_log:
            print("Experience saved at {0}. Episode num: {1}, Transition num: {2}.".
                  format(path, self.epis_num, self.trans_num)
                  )

    def load(self, path, print_log=False):
        self.epis_list = np.load(path, allow_pickle=True).tolist()
        if print_log:
            print("Experience load from {0}. Episode num: {1}, Transition num: {2}.".
                  format(path, self.epis_num, self.trans_num)
                  )
        if self.epis_num > self.capacity:
            print("WARNING! Number of loaded episodes({0}) is bigger than capacity({1}).".format(
                self.epis_num, self.capacity))

    def info(self):
        return self.__str__()

    def describe(self):
        return self.__str__() + "\r\n### Sample of Episode(-1): \r\n{0}".\
            format(
                self.epis_list[-1] if self.epis_list else "None",
        )

    def head(self, size=5):
        tmps = self.sample_trans(size)  # TODO use sample as head
        df = pd.DataFrame(
            data=tmps,
            columns=["state", "action", "reward", "state_", "is_done", "extra"]
        )
        return df


    '''property'''

    @property
    def df(self):
        tmp = sum([each.trans_list for each in self.epis_list], [])
        df = pd.DataFrame(
            data=tmp,
            columns=["state", "action", "reward", "state_", "is_done", "extra"]
        )
        return df
    
    @property
    def log_df(self):
        tmp = sum(self.log_list, [])
        
        df = pd.DataFrame(
            data=tmp,
            columns=self.log_key
        )
        return df

    @property
    def total_reward_list(self):
        return [sum([each_[0] for each_ in each]) for each in self.log_list]
    
    @property
    def latest_reward(self):
        return sum([each[0] for each in self.log_list[-1]])
        
    @property
    def epis_len_list(self):
        return [len(each) for each in self.log_list]

    @property
    def total_reward_mean(self):
        return np.mean(self.total_reward_list)

    @property
    def epis_len_mean(self):
        return np.mean(self.epis_len_list)

    @property
    def trans_num(self):
        return sum(self.epis_len_list)

    @property
    def epis_num(self):
        return len(self)
    
    @property
    def is_custom_log(self):
        if len(self.log_key) == 1 and self.log_key[0] == 'reward':
            return False
        else:
            return True


    def __str__(self):
        return "Experience:\r\n\tepis_num: {4}\r\n\tepis_len_mean: {1:.2f}\r\n\ttotal_reward_mean: {3:.2f}\r\n\tepis_len_list: {0}\r\n\ttotal_reward_list: {2}\r\n".\
            format(
                self.epis_len_list if len(
                    self.epis_len_list) < 10 else self.epis_len_list[:10]+["..."],
                self.epis_len_mean,
                self.total_reward_list if len(
                    self.total_reward_list) < 10 else self.total_reward_list[:10]+["..."],
                self.total_reward_mean,
                len(self),
            )

    def __len__(self):
        return len(self.epis_list)

    def __getitem__(self, index):
        return self.epis_list[index]

    def __iter__(self):
        return iter(self.epis_list)


    '''utils'''

    def pop(self, idx):
        raise NotImplementedError()
        return

    def log_append(self, custom_log, r):
        if custom_log:
            if self.is_custom_log: # custom append
                self.log_list[-1].append([custom_log[each] for each in self.log_key])
            else: # log key init
                self.log_key = [each for each in custom_log.keys()]
                self.log_list[-1].append([custom_log[each] for each in self.log_key])
        else: # default
            self.log_list[-1].append([r])

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    experience = Experience(capacity=100)

    for epis_i in range(10):
        s = env.reset()
        for step_i in range(100):
            a = env.action_space.sample()
            s_, r, is_done, _ = env.step(a)  # take a random action
            experience.push_trans(Transition(s, a, r, s_, is_done))
            s = s_
            if is_done:
                break
        env.close()

    print(experience.info())  # basic info = print(__str__)
    print(experience.describe())  # basic info + details
    print(experience.head())  # sample first 5 and print as table
    print(experience.df)  # print pd tables of all transition
    print(experience)  # print experience
    print(experience.sample_trans(10)[0])  # print transition
    print(experience.sample_epis(2)[0])  # print episode
    print(experience.calculate_gain(0.5))
    print(experience.df)  # print pd tables of all transition
    experience.save("./test_memory.npy", print_log=True)
    experience.load("./test_memory.npy", print_log=True)
    print(experience.log_df)
    print(experience.total_reward_list)
    print(experience.epis_len_list)
    # s, a, r, s_, is_done, extra = zip(*experience[-1])
    # tmp = [each['gain'] for each in extra]
    # print(tmp)

    # TODO: 通过is_done判断结束是不合理的，那些超过最大长度的episode最后一步依然是is_done=False
