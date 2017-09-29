import numpy as np
import tensorflow as tf


# Helper Function------------------------------------------------------------------------------------------------------------
# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.


def dlrelu(x, alpha=0.1):
  return tf.nn.relu(x) - alpha * tf.nn.relu(0.05-x) - (1 - alpha) *  tf.nn.relu(x-0.95)
  
class RunningStat(object):
    def __init__(self, shape, eps):
        self.sum = np.zeros(shape, dtype=np.float32)
        self.sumsq = np.full(shape, eps, dtype=np.float32)
        self.count = eps

    def increment(self, s, ssq, c):
        self.sum += s
        self.sumsq += ssq
        self.count += c

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def std(self):
        return np.sqrt(np.maximum(self.sumsq / self.count - np.square(self.mean), 1e-2))

    def set_from_init(self, init_mean, init_std, init_count):
        self.sum[:] = init_mean * init_count
        self.sumsq[:] = (np.square(init_mean) + np.square(init_std)) * init_count
        self.count = init_count
     
def normalize(x):
    return (x - np.mean(x)) / np.std(x)

# process state (the last 3 entires are obstacle info which should not be processed)
def process_state(s,s1,center=True,diff=0):
    s = np.asarray(s)
    s1 = np.asarray(s1)
    s_14 = (s1[22:36]-s[22:36]) / 0.01
    s_3 = (s1[38:]-s[38:]) / 0.01
    s = np.hstack((s1[:36],s_14,s1[36:],s_3))
    if diff == 0:
        #s[-6:] = 0.0 # if diff = 0, then manully turn off all obstacles
        s = np.delete(s,[52,53,54,55,56,57])

    if center:
      # transform into all relative quantities
      x_pos = [1,22,24,26,28,30,32,34]
      y_pos = [i+1 for i in x_pos]
      for i in x_pos:
          s[i] -= s[18]
      for j in y_pos:
          s[j] -= s[19]
      
      x_vs = [i+14 for i in x_pos]
      x_vs[0] = 4
      y_vs = [i+1 for i in x_vs]
      for i in x_vs:
          s[i] -= s[20]
      for j in y_vs:
          s[j] -= s[21]
      # transform cm as origin
      #s[18:22] = 0.0
      s = np.delete(s,[18,19,20,21])  
    return normalize(s)
        
def n_step_transition(episode_buffer,n_step,gamma):
    _,_,_,s1,done = episode_buffer[-1]
    s,action,_,_,_ = episode_buffer[-1-n_step]
    r = 0
    for i in range(n_step):
      r += episode_buffer[-1-n_step+i][2]*gamma**i
    return [s,action,r,s1,done]

def engineered_action(seed):
    test = np.ones(18)*0.05
    if seed < 0.5:
        test[0] = 0.3
        test[3] = 0.8
        test[4] = 0.5
        test[6] = 0.3
        test[8] = 0.8
        test[9] = 0.3
        test[11] = 0.5
        test[14] = 0.3
        test[17] = 0.5
    else:
        test[9] = 0.3
        test[12] = 0.8
        test[13] = 0.5
        test[15] = 0.3
        test[17] = 0.8
        test[0] = 0.3
        test[2] = 0.5
        test[3] = 0.3
        test[8] = 0.5
            
    return test

# [Hacked] the memory might always be leaking, here's a solution #58
# https://github.com/stanfordnmbl/osim-rl/issues/58 
# separate process that holds a separate RunEnv instance.
# This has to be done since RunEnv() in the same process result in interleaved running of simulations.

import opensim as osim
from osim.http.client import Client
from osim.env import *

import multiprocessing
from multiprocessing import Process, Pipe

def standalone_headless_isolated(conn,vis,seed,diff):
    e = RunEnv(visualize=vis)
    while True:
        try:
            msg = conn.recv()

            # messages should be tuples,
            # msg[0] should be string

            if msg[0] == 'reset':
                o = e.reset(difficulty=diff,seed=seed)
                conn.send(o)
            elif msg[0] == 'step':
                ordi = e.step(msg[1])
                conn.send(ordi)
            else:
                conn.close()
                del e
                return
        except:
            conn.close()
            del e
            raise

# class that manages the interprocess communication and expose itself as a RunEnv.
class ei: # Environment Instance
    def __init__(self,vis,seed,diff):
        self.pc, self.cc = Pipe()
        self.p = Process(
            target = standalone_headless_isolated,
            args=(self.cc,vis,seed,diff,)
        )
        self.p.daemon = True
        self.p.start()

    def reset(self):
        self.pc.send(('reset',))
        return self.pc.recv()

    def step(self,actions):
        self.pc.send(('step',actions,))
        try:
            return self.pc.recv()
        except :  
            print('Error in recv()')
            raise

    def __del__(self):
        self.pc.send(('exit',))
        #print('(ei)waiting for join...')
        self.p.join()
	try:
	    del self.pc
	    del self.cc
	    del self.p
	except:
	    raise


