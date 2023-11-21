import gym
import time

def run():
    env = gym.make('gym_anm:ANM6Easy-v0')
    o = env.reset()

    for i in range(100):
        a = env.action_space.sample()
        o, r, done, info = env.step(a)
        env.render()
        time.sleep(0.5)  # otherwise the rendering is too fast for the human eye.

    env.close()

if __name__ == '__main__':
    run()