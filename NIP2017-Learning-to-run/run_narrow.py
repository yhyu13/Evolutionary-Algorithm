"""
Simple code for Distributed ES proposed by OpenAI.
Based on this paper: Evolution Strategies as a Scalable Alternative to Reinforcement Learning
Details can be found in : https://arxiv.org/abs/1703.03864

Visit more on my tutorial site: https://morvanzhou.github.io/tutorials/
"""

"""
Modified by Hang Yu on Sep25th
"""
import argparse
import numpy as np
import gym
import multiprocessing as mp
import time
from helper import *

POOL = None                # multiprocess pool
ENVS = None                # environment list for effective reuse
DIFF = 1
LOAD_MODEL = True         # load training result
N_KID = 2                 # half of the training population
N_STEP = 1                # frame skip
N_GENERATION = 5000         # training step
LR = .05                   # learning rate
SIGMA = .05                 # mutation strength or step size
N_CORE = mp.cpu_count()-1
CONFIG = [
    dict(game="CartPole-v0",
         n_feature=4, n_action=2, continuous_a=[False], ep_max_step=700, eval_threshold=500),
    dict(game="MountainCar-v0",
         n_feature=2, n_action=3, continuous_a=[False], ep_max_step=200, eval_threshold=-120),
    dict(game="Pendulum-v0",
         n_feature=3, n_action=1, continuous_a=[True, 2.], ep_max_step=200, eval_threshold=-180),
    dict(game="opensim",
         n_feature=58, n_action=18, continuous_a=[True, 1.], ep_max_step=1000, eval_threshold=3)
][3]    # choose your game


def sign(k_id): return -1. if k_id % 2 == 0 else 1.  # mirrored sampling


class SGD(object):                      # optimizer with momentum
    def __init__(self, params, learning_rate, momentum=0.9,b1=0.1, b2=0.001):
        self.v = np.zeros_like(params).astype(np.float32)
        self.lr, self.momentum,self.b1,self.b2 = learning_rate, momentum,b1,b2
        self.t = 0

    def get_gradients(self, gradients):
        self.t += 1
        fix1 = 1. - (1. - self.b1)**self.t
        fix2 = 1. - (1. - self.b2)**self.t
        lr_t = self.lr * (np.sqrt(fix2) / fix1)
        self.v = self.momentum * self.v + (1. - self.momentum) * gradients
        return self.lr * self.v

class ADAM(object):                      # optimizer with momentum
    def __init__(self, params, learning_rate, b1=0.1, b2=0.001, e=1e-8):
        self.v = np.zeros_like(params).astype(np.float32)
        self.lr, self.b1, self.b2, self.e = learning_rate, b1,b2,e
        self.t = 0

    def get_gradients(self, gradients):
        self.t += 1
        fix1 = 1. - (1. - self.b1)**self.t
        fix2 = 1. - (1. - self.b2)**self.t
        lr_t = self.lr * (np.sqrt(fix2) / fix1)

        m_t = (self.b1 * gradients) + ((1. - self.b1) * self.v)
        v_t = (self.b2 * np.sqrt(abs(gradients))) + ((1. - self.b2) * self.v)
        g_t = m_t / (np.sqrt(v_t) + self.e)
        return lr_t * g_t


def params_reshape(shapes, params):     # reshape to be a matrix
    p, start = [], 0
    for i, shape in enumerate(shapes):  # flat params to matrix
        n_w, n_b = shape[0] * shape[1], shape[1]
        p = p + [params[start: start + n_w].reshape(shape),
                 params[start + n_w: start + n_w + n_b].reshape((1, shape[1]))]
        start += n_w + n_b
    return p


def get_reward(shapes, params, env, ep_max_step, continuous_a, seed_and_id=None,):
    # perturb parameters using seed
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        env = ENVS[k_id]
        np.random.seed(seed)
        params += sign(k_id) * SIGMA * np.random.randn(params.size)
    p = params_reshape(shapes, params)
    # run episode
    s = env.reset()
    e_a = np.ones(18)*0.05#engineered_action(0.1)
    s = env.step(e_a)[0]
    s1 = env.step(e_a)[0]
    s = process_state_3(s,s1,diff=0)
    ep_r = 0.
    for step in range(ep_max_step):
        a = get_action(p, s, continuous_a)
        temp_r = 0
        for i in range(N_STEP):
            s2, r, done, _ = env.step(a)
            if done: break
            if i == N_STEP - 2: s1 = s2
            if i == N_STEP - 1: s1 = process_state_3(s1,s2,diff=0)
            temp_r += r
        s = s1
        s1 = s2
        ep_r += temp_r
        if done: break
    return ep_r

# hangyu5 Sep25
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x,-5,5)))

def get_action(params, x, continuous_a):
    x = np.expand_dims(x, axis=0)
    x = np.maximum((x.dot(params[0]) + params[1]),0.0)
    x = np.maximum((x.dot(params[2]) + params[3]),0.0)
    x = x.dot(params[4]) + params[5]
    return sigmoid(x)[0]                # for continuous action


def build_net():
    def linear(n_in, n_out,last_layer=False):  # network linear layer
        w = np.random.randn(n_in * n_out).astype(np.float32) * .1
        b = np.random.randn(n_out).astype(np.float32) * .1
        if last_layer:
            b -= 3. # samrt sigmoid
        return (n_in, n_out), np.concatenate((w, b))
    s0, p0 = linear(CONFIG['n_feature'], 30)
    s1, p1 = linear(30, 20)
    s2, p2 = linear(20, CONFIG['n_action'],last_layer=True)
    return [s0, s1, s2], np.concatenate((p0, p1, p2))

def build_net_2():
    def linear(n_in, n_out,last_layer=False):  # network linear layer
        w = np.random.randn(n_in * n_out).astype(np.float32) * .1
        b = np.random.randn(n_out).astype(np.float32) * .1
        if last_layer:
            b -= 3. # samrt sigmoid
        return (n_in, n_out), np.concatenate((w, b))
    s0, p0 = linear(CONFIG['n_feature'], 256)
    s1, p1 = linear(256, 256)
    s2, p2 = linear(256, CONFIG['n_action'],last_layer=True)
    return [s0, s1, s2], np.concatenate((p0, p1, p2))
    


def train(net_shapes, net_params, optimizer, utility):
    # pass seed instead whole noise matrix to parallel will save your time
    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID).repeat(2)    # mirrored sampling

    # distribute training in parallel
    jobs = [POOL.apply_async(get_reward, (net_shapes, net_params, None, CONFIG['ep_max_step'], CONFIG['continuous_a'],
                                          [noise_seed[k_id], k_id], )) for k_id in range(N_KID*2)]
    rewards = np.array([j.get() for j in jobs])
    kids_rank = np.argsort(rewards)[::-1]               # rank kid id by reward

    cumulative_update = np.zeros_like(net_params)       # initialize update values
    for ui, k_id in enumerate(kids_rank):
        np.random.seed(noise_seed[k_id])                # reconstruct noise using seed
        cumulative_update += utility[ui] * sign(k_id) * np.random.randn(net_params.size)

    gradients = optimizer.get_gradients(cumulative_update/(2*N_KID*SIGMA))
    return net_params + gradients, rewards

def main():
    global ENVS
    global POOL
    global DIFF
 
    # utility instead reward for update parameters (rank transformation)
    base = N_KID * 2    # *2 for mirrored sampling
    rank = np.arange(1, base + 1)
    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base

    # training
    net_shapes, net_params = build_net()
    
    if LOAD_MODEL:
        # load model, keep training
        print("Load Model Success!")
        net_params = np.load('./models_narrow/model_reward_3'+'.npy')
        
    #env = gym.make(CONFIG['game']).unwrapped
    ENVS = [ei(vis=True,seed=0,diff=DIFF) for _ in range(N_KID*2)]
    optimizer = SGD(net_params, LR)#ADAM(net_params, LR)
    POOL = mp.Pool(processes=N_CORE)
    mar = None      # moving average reward
    t_start = time.time()
    try:
        print('START TRAINING%----------------------------------------------------%')
        for g in range(N_GENERATION):
            t0 = time.time()
            net_params, kid_rewards = train(net_shapes, net_params, optimizer, utility)

            # test trained net without noise
            net_r = get_reward(net_shapes, net_params, ENVS[0],CONFIG['ep_max_step'], CONFIG['continuous_a'],None,)
            mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r       # moving average reward
            print(
                'Gen: ', g,
                '| Net_R: %.2f' % mar,
                '| Kid_avg_R: %.2f' % kid_rewards.mean(),
                '| Gen_T: %.2f' % (time.time() - t0),
                '| Total_T: %.2f' % ((time.time() - t_start)/3600)+'h',)
            if mar >= CONFIG['eval_threshold']+0.5:
                # succeed, save model and break
                print("Success, save model!")
                np.save('./models_narrow/model_reward_'+str(CONFIG['eval_threshold']),net_params)
                CONFIG['eval_threshold'] += 1


    except KeyboardInterrupt:
        print("Ctrl-c received! Sending kill to process...")
        POOL.terminate()
        del ENVS
    else:
        print("Normal termination")
        POOL.close()
        del ENVS
    POOL.join()

if __name__ == "__main__":
    main()
