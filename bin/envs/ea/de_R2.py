from __future__ import division
import numpy as np
import random
from numpy.random import rand
import gym
from gym import spaces
from gym.utils import seeding
import math
from scipy.spatial import distance
import time
from scipy.stats import rankdata
from collections import Counter
from optproblems import *
from optproblems.cec2005 import *
import cocoex
import os

def rand1(population, samples, scale, best, i): # DE/rand/1
    r0, r1, r2 = samples[:3]
    return (population[r0] + scale * (population[r1] - population[r2]))

def rand2(population, samples, scale, best, i): # DE/rand/2
    r0, r1, r2, r3, r4 = samples[:5]
    return (population[r0] + scale * (population[r1] - population[r2] + population[r3] - population[r4]))

def rand_to_best2(population, samples, scale, best, i): # DE/rand-to-best/2
    r0, r1, r2, r3, r4 = samples[:5]
    return (population[r0] + scale * (population[best] - population[r0] + population[r1] - population[r2] + population[r3] - population[r4]))

def current_to_rand1(population, samples, scale, best, i): # DE/current-to-rand/1
    r0, r1, r2 = samples[:3]
    return (population[i] + scale * (population[r0] - population[i] + population[r1] - population[r2]))

def select_samples(popsize, candidate, number_samples):
    """
    obtain random integers from range(popsize),
    without replacement.  You can't have the original candidate either.
    """
    idxs = list(range(popsize))
    idxs.remove(candidate)
    return(np.random.choice(idxs, 5, replace = False))


def min_max(a, mi, mx):
    if a < mi:
        mi = a
    if a > mx:
        mx = a
    return mi, mx

def normalise(a, mi, mx):
    a = (a - mi) / (mx - mi);
    return a

def count_success(popsize, gen_window, j, n_ops, Off_met):
    # gen_window needs to be an array here for count operations.
    c_s = np.zeros(n_ops); c_us = np.zeros(n_ops)
    for op in range(n_ops):
        c_s[op] = np.sum((gen_window[j, :, 0] == op) & (~np.isnan(gen_window[j, :, Off_met])))
        c_us[op] = np.sum((gen_window[j, :, 0] == op) & (np.isnan(gen_window[j, :, Off_met])))
    return c_s, c_us

def function_at_generation(n_ops, gen_window, j, Off_met, function):
    value = np.zeros(n_ops)
    a = gen_window[0, :, Off_met]
    for op in range(n_ops):
        b = np.where((gen_window[0, :, 0] == op) & ~np.isnan(a), a, 0.0)
        value[op] = function(b)
    return value

def min_gen(max_gen, gen_window):
    return np.minimum(max_gen, gen_window.shape[0])

                                                        ##########################Success based###########################################

# Applicable for fix number of generations
def Success_Rate1(popsize, n_ops, gen_window, Off_met, max_gen):
    print("Success rate1")
    gen_window = np.array(gen_window)
    gen_window_len = len(gen_window)
    max_gen = min_gen(max_gen, gen_window)
    state_value = np.zeros(n_ops)
    for j in range(gen_window_len - max_gen, gen_window_len):
        total_success, total_unsuccess = count_success(popsize, gen_window, j, n_ops, Off_met) # Counting works fine.
        n_applications = total_success + total_unsuccess # Sum works fine
        n_applications[n_applications == 0] = 1
        state_value += total_success / n_applications
    #for i in range(n_ops):
        #appl = 0; t_s = 0
        #for j in range(len(gen_window)-1, len(gen_window)-max_gen-1, -1):
            #total_success = 0; total_unsuccess = 0
            #if np.any(gen_window[j, :, 0] == i):
                #total_success, total_unsuccess = count_success(popsize, gen_window, i, j, Off_met)
                #t_s += total_success
                #appl += total_success + total_unsuccess
        #if appl != 0:
            #state_value[i] = t_s / appl
    return state_value

                                                        ##########################Weighted offspring based################################

# Applicable for fix number of generations
def Weighted_Offspring1(popsize, n_ops, gen_window, Off_met, max_gen):
    print("Weighted offspring1", gen_window)
    gen_window = np.array(gen_window)
    gen_window_len = len(gen_window)
    max_gen = min_gen(max_gen, gen_window)
    state_value = np.zeros(n_ops)
    for j in range(gen_window_len - max_gen, gen_window_len):
        total_success, total_unsuccess = count_success(popsize, gen_window, j, n_ops, Off_met)
        n_applications = total_success + total_unsuccess
        n_applications[n_applications == 0] = 1
        state_value += function_at_generation(n_ops, gen_window, j, Off_met, np.sum) / n_applications
    state_value = state_value / np.sum(state_value)
    #for i in range(n_ops):
        #appl = 0
        #for j in range(len(gen_window)-1, len(gen_window)-max_gen-1, -1):
            #total_success = 0; total_unsuccess = 0
            #if np.any(gen_window[j, :, 0] == i):
                #total_success, total_unsuccess = count_success(popsize, gen_window, i, j, Off_met)
                #state_value[i] += np.sum(gen_window[j, np.where((gen_window[j, :, 0] == i) & (gen_window[j, :, Off_met] != -1)), Off_met])
                #appl += total_success + total_unsuccess
        #if appl != 0:
            #state_value[i] = state_value[i] / appl
    #if np.sum(state_value) != 0:
        #state_value = state_value / np.sum(state_value)
    return state_value


# Applicable for fix window size
def Weighted_Offspring2(popsize, n_ops, window, Off_met, max_gen):
    state_value = np.zeros(n_ops)
    window = window[window[:, 0] != -1][:, :]
    for i in range(n_ops):
        if np.sum((window[:, 0] == i) & (window[:, Off_met] != -1)) > 0:
            state_value[i] = np.sum(window[np.where((window[:, 0] == i) & (window[:, Off_met] != -1)), Off_met]) / np.sum((window[:, 0] == i) & (window[:, Off_met] != -1));
    if np.sum(state_value) != 0:
        state_value = state_value / np.sum(state_value)
    return state_value

                                                        ##########################Best offspring based#############################

# Applicable for last two generations
def Best_Offspring1(popsize, n_ops, gen_window, Off_met, max_gen):
    # gen_window is of type list here.
    print("Best offspring1")
    gen_window = np.array(gen_window)
    gen_window_len = len(gen_window)
    total_success_t, total_unsuccess_t = count_success(popsize, gen_window, gen_window_len - 1, n_ops, Off_met)
    best_t = function_at_generation(n_ops, gen_window, gen_window_len - 1, Off_met, np.max)
    if gen_window_len >= 2:
        total_success_t_1, total_unsuccess_t_1 = count_success(popsize, gen_window, gen_window_len - 2, n_ops, Off_met)
        best_t_1 = function_at_generation(n_ops, gen_window, gen_window_len - 2, Off_met, np.max)
    else:
        total_success_t_1 = 0; total_unsuccess_t_1 = 0
        best_t_1 = np.zeros(n_ops)
    state_value = np.fabs(best_t - best_t_1)
    n_applications = total_success_t + total_unsuccess_t - (total_success_t_1 + total_unsuccess_t_1)
    n_applications[n_applications == 0] = 1
    best_t_1[best_t_1 == 0] = 1
    state_value = state_value / (best_t_1 * np.fabs(n_applications))
    #state_value = np.zeros(n_ops)
    #gen_window = np.array(gen_window)
    #best_t = np.zeros(n_ops); best_t_1 = np.zeros(n_ops)
    #for i in range(n_ops):
        # for last 2 generations
        #n_applications = np.zeros(2)
        # Calculating best in current generation
        #if np.any((gen_window[len(gen_window)-1, :, 0] == i) & (gen_window[len(gen_window)-1, :, Off_met] != -1)):
            #total_success, total_unsuccess = count_success(popsize, gen_window, i, len(gen_window)-1, Off_met)
            #n_applications[0] = total_success + total_unsuccess
            #best_t[i] = np.max(gen_window[len(gen_window)-1, np.where((gen_window[len(gen_window)-1, :, 0] == i) & (gen_window[len(gen_window)-1, :, Off_met] != -1)), Off_met])
        # Calculating best in last generation
        #if len(gen_window)>=2 and np.any((gen_window[len(gen_window)-2,:,0] == i) & (gen_window[len(gen_window)-2, :, Off_met] != -1)):
            #total_success, total_unsuccess = count_success(popsize, gen_window, i, len(gen_window)-2, Off_met)
            #n_applications[1] = total_success + total_unsuccess
            #best_t_1[i] = np.max(gen_window[len(gen_window)-2, np.where((gen_window[len(gen_window)-2, :, 0] == i) & (gen_window[len(gen_window)-2, :, Off_met] != -1)), Off_met])
        #if best_t_1[i] != 0 and np.fabs(n_applications[0] - n_applications[1]) != 0:
            #state_value[i] = np.fabs(best_t[i] - best_t_1[i]) / ((best_t_1[i]) * (np.fabs(n_applications[0] - n_applications[1])))
        #elif best_t_1[i] != 0 and np.fabs(n_applications[0] - n_applications[1]) == 0:
            #state_value[i] = np.fabs(best_t[i] - best_t_1[i]) / (best_t_1[i])
        #elif best_t_1[i] == 0 and np.fabs(n_applications[0] - n_applications[1]) != 0:
            #state_value[i] = np.fabs(best_t[i] - best_t_1[i]) / (np.fabs(n_applications[0] - n_applications[1]))
        #else:
            #state_value[i] = np.fabs(best_t[i] - best_t_1[i])
    if np.sum(state_value) != 0:
        state_value = state_value / np.sum(state_value)
    return state_value


# Applicable for fix number of generations
def Best_Offspring2(popsize, n_ops, gen_window, Off_met, max_gen):
    state_value = np.zeros(n_ops)
    gen_window = np.array(gen_window)
    gen_window_len = len(gen_window)
    max_gen = min_gen(max_gen, gen_window)
    for j in range(gen_window_len - max_gen, gen_window_len):
        state_value += function_at_generation(n_ops, gen_window, j, Off_met, np.max)
    
    #for i in range(n_ops):
        #gen_best = []
        #for j in range(len(gen_window)-1, len(gen_window)-max_gen-1, -1):
            #if np.any((gen_window[j,:,0] == i) & (gen_window[j, :, Off_met] != -1)):
                #gen_best.append(np.max(np.hstack(gen_window[j, np.where((gen_window[j,:,0] == i) & (gen_window[j, :, Off_met] != -1)), Off_met])))
                #state_value[i] += np.sum(gen_best)
    if np.sum(state_value) != 0:
        state_value = state_value / np.sum(state_value)
    return state_value

                                                ##########################update window###########################################

def update_window(window, window_size, second_dim, opu, i, copy_F, F1):
    # Update window FIFO style
    which = (window[:, 1] == np.inf)
    if np.any(which):
        last_empty = np.max(np.flatnonzero(which))
        window[last_empty] = second_dim
        return window
    which = (window[:, 0] == opu[i])
    if np.any(which):
        last = np.max(np.flatnonzero(which))
    else:
        last = np.argmin(window[:,1])
    window[1:(last+1), :] = window[0:last, :]
    window[0, :] = second_dim
    return window

                                                ##########################class DEEnv###########################################

mutations = [rand1, rand2, rand_to_best2, current_to_rand1]

class DEEnv(gym.Env):
    def __init__(self, func_choice, FF, NP, CR, FE, max_gen, W):
        # Content common to all episodes
        self.n_ops = 4
        self.action_space = spaces.Discrete(self.n_ops)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(99,), dtype = np.float32)
        #self.max_gen = 10
        #self.window_size = 50
        #self.FF = 0.5; self.CR = 1.0
        #self.budget = 1e2
        #self.NP = 100
        self.func_choice = func_choice
        self.FF = FF
        self.CR = CR
        self.NP = NP
        self.budget = FE
        self.max_budget = self.budget
        self.max_gen = max_gen
        self.window_size = W
        self.number_metric = 5
        self.NP = 10

        # CEC2005
        # self.func_choice = [unimodal.F1, unimodal.F2, unimodal.F5, basic_multimodal.F6, basic_multimodal.F8, basic_multimodal.F10, basic_multimodal.F11, basic_multimodal.F12, expanded_multimodal.F13, expanded_multimodal.F14, f15.F15, f19.F19, f20.F20, f21.F21, f22.F22, f24.F24]
        # self.d_choice = [10, 30]
        # self.func_select = [(func, d) for func in self.func_choice for d in self.d_choice]
        # self.fun_index = 0
        # BBOB
        suite_name = "bbob"
        suite_options = "dimensions: 2, 3, 5, 10, 20, 40"
        self.suite = cocoex.Suite(suite_name, "", suite_options)
        # First "" takes following arguments: year, instances; Second "" takes following arguments: dimensions, dimension_indices, function_indices, instance_indices
        self.fun_index = 0
    
    def step(self, action):
        assert action >=0 and action < 4
        self.opu[self.i] = action
        mutate = mutations[action]
        
        # Evolution of parent i
        bprime = mutate(self.population, self.r, self.FF, self.best, self.i)
        bprime[bprime < self.lbounds[0]] = self.lbounds[0]
        bprime[bprime > self.ubounds[0]] = self.ubounds[0]
        self.crossovers = (np.random.rand(self.dim) < self.CR)
        self.crossovers[self.fill_points[self.i]] = True
        self.u[self.i, :] = np.where(self.crossovers, bprime, self.X[self.i, :])
        self.F1[self.i] = self.fun(self.u[self.i])
    
        reward = 0
        second_dim = np.full(self.number_metric, np.nan)
        second_dim[0] = self.opu[self.i]
        if self.F1[self.i] < self.copy_F[self.i]:
            # Fitness improvement wrt parent
            second_dim[1] = self.copy_F[self.i] - self.F1[self.i]
            # Fitness improvement wrt best parent
            if self.F1[self.i] < self.fmin:
                second_dim[2] = self.fmin - self.F1[self.i]
            # Fitness improvement wrt bsf
            if self.F1[self.i] < self.best_so_far:
                second_dim[3] = self.best_so_far - self.F1[self.i]
                #self.best_so_far = self.F1[self.i]
                #self.best_so_far_position = self.population[self.i]
                self.stagnation_count = 0;
                reward = 10
            else:
                self.stagnation_count += 1
                reward = 1
            # Fitness improvement wrt median population fitness
            if self.F1[self.i] < self.fmedian:
                second_dim[4] = self.fmedian - self.F1[self.i]
            
            self.window = update_window(self.window, self.window_size, second_dim, self.opu, self.i, self.copy_F, self.F1)
            
            #self.best = np.argmin(self.copy_F)
            #self.best_so_far = self.copy_F(self.best)
            #self.best_so_far_position = self.population[self.best]
            #self.worst_so_far = np.max(self.copy_F)
            
            #if self.worst_so_far < self.F1[self.i]:
                #self.worst_so_far = self.F1[self.i]
            self.F[self.i] = self.F1[self.i]
            self.X[self.i, :] = self.u[self.i, :]
        
        self.third_dim.append(second_dim)
        
        self.max_std = np.std((np.repeat(self.best_so_far, self.NP/2), np.repeat(self.worst_so_far, self.NP/2)))

        self.budget -= 1
        self.i = self.i+1
        
        if self.i >= self.NP:
            self.gen_window.append(self.third_dim);
            # Generation based statistics
            self.copy_ob = np.zeros(64)
            self.copy_ob[0:4] = Success_Rate1(self.NP, self.n_ops, self.gen_window, 1, self.max_gen)
            self.copy_ob[4:8] = Success_Rate1(self.NP, self.n_ops, self.gen_window, 2, self.max_gen)
            self.copy_ob[8:12] = Success_Rate1(self.NP, self.n_ops, self.gen_window, 3, self.max_gen)
            self.copy_ob[12:16] = Success_Rate1(self.NP, self.n_ops, self.gen_window, 4, self.max_gen)
            
            self.copy_ob[16:20] = Weighted_Offspring1(self.NP, self.n_ops, self.gen_window, 1, self.max_gen)
            self.copy_ob[20:24] = Weighted_Offspring1(self.NP, self.n_ops, self.gen_window, 2, self.max_gen)
            self.copy_ob[24:28] = Weighted_Offspring1(self.NP, self.n_ops, self.gen_window, 3, self.max_gen)
            self.copy_ob[28:32] = Weighted_Offspring1(self.NP, self.n_ops, self.gen_window, 4, self.max_gen)
            
            self.copy_ob[32:36] = Best_Offspring1(self.NP, self.n_ops, self.gen_window, 1, self.max_gen)
            self.copy_ob[36:40] = Best_Offspring1(self.NP, self.n_ops, self.gen_window, 2, self.max_gen)
            self.copy_ob[40:44] = Best_Offspring1(self.NP, self.n_ops, self.gen_window, 3, self.max_gen)
            self.copy_ob[44:48] = Best_Offspring1(self.NP, self.n_ops, self.gen_window, 4, self.max_gen)
            
            self.copy_ob[48:52] = Best_Offspring2(self.NP, self.n_ops, self.gen_window, 1, self.max_gen)
            self.copy_ob[52:56] = Best_Offspring2(self.NP, self.n_ops, self.gen_window, 2, self.max_gen)
            self.copy_ob[56:60] = Best_Offspring2(self.NP, self.n_ops, self.gen_window, 3, self.max_gen)
            self.copy_ob[60:64] = Best_Offspring2(self.NP, self.n_ops, self.gen_window, 4, self.max_gen)

            self.third_dim = []
            self.opu = np.ones(self.NP) * 4
            self.i = 0
            self.fill_points = np.random.randint(self.dim, size = self.NP)
            self.generation = self.generation + 1
            self.population = np.copy(self.X)
            self.copy_F = np.copy(self.F)
            # Best index in current population
            self.best = np.argmin(self.copy_F)
            
            if self.fmin > self.copy_F[self.best]:
                self.best_so_far = self.copy_F[self.best]
                self.best_so_far_position = self.population[self.best]
            self.fmin = self.copy_F[self.best]
            # Worst so far
            if self.worst_so_far < np.max(self.copy_F):
                self.worst_so_far = np.max(self.copy_F)
            # median of parent population
            self.fmedian = np.median(self.copy_F)
            # Average poupulation fitness
            self.pop_average = np.average(self.copy_F)
            # Standard Deviation
            self.pop_std = np.std(self.copy_F)
        
        assert self.i < self.NP and self.i >= 0
        # Preparation for observation to give for next action decision
        self.r = select_samples(self.NP, self.i, 5)

        ob = np.zeros(99); ob[19:83] = np.copy(self.copy_ob)

        # Parent fintness
        ob[0] = normalise(self.copy_F[self.i], self.best_so_far, self.worst_so_far)
        # Population fitness statistic
        ob[1] = normalise(self.pop_average, self.best_so_far, self.worst_so_far)
        ob[2] = self.pop_std / self.max_std
        ob[3] = self.budget / self.max_budget
        ob[4] = self.dim / 50
        ob[5] = self.stagnation_count / self.max_budget

        # Random sample based observations
        ob[6:12] = distance.cdist(self.population[[self.r[0],self.r[1],self.r[2],self.r[3],self.r[4],self.best]], np.expand_dims(self.population[self.i], axis=0)).T / self.max_dist
        ob[12:18] = np.fabs(self.copy_F[[self.r[0],self.r[1],self.r[2],self.r[3],self.r[4],self.best]] - self.copy_F[self.i]) / (self.worst_so_far - self.best_so_far)
        # Best candidate seen so far
        ob[18] = distance.euclidean(self.best_so_far_position, self.population[self.i]) / self.max_dist;
        
        # Window based statistics
        ob[83:87] = Weighted_Offspring2(self.NP, self.n_ops, self.window, 1, self.max_gen)
        ob[87:91] = Weighted_Offspring2(self.NP, self.n_ops, self.window, 2, self.max_gen)
        ob[91:95] = Weighted_Offspring2(self.NP, self.n_ops, self.window, 3, self.max_gen)
        ob[95:99] = Weighted_Offspring2(self.NP, self.n_ops, self.window, 4, self.max_gen)
        #assert np.all(ob >= 0.0) and np.all(ob <= 1.0)

        if self.budget <= 0:
            print("time taken for one episode:", time.time() - self.a)
            print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",self.budget, self.best_so_far,"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            return ob, reward, True, {}
        else:
            return ob, reward, False, {}


    def reset(self):
        # Content common to one episode and may or may not be same for others.
        # CEC2005
        # self.dim = self.func_select[self.fun_index][1]
        # self.fun = self.func_select[self.fun_index][0](self.dim)
        # self.lbounds = self.fun.min_bounds; self.lbounds = np.array(self.lbounds); print(self.lbounds)
        # self.ubounds = self.fun.max_bounds; self.ubounds = np.array(self.ubounds); print(self.ubounds)
        # opti = self.fun.get_optimal_solutions()
        # for o in opti:
            # sol = np.copy(o.phenome)
        # self.best_value = self.fun.objective_function(sol)
        # print("Function info: fun= ", self.func_select[self.fun_index], " with dim = ", self.dim, " with best value= ", self.best_value)
        
        self.a = time.time()
        # BBOB function selection
        self.fun = self.suite[int(self.func_choice[self.fun_index])]
        self.dim = self.fun.dimension
        self.lbounds = self.fun.lower_bounds
        self.ubounds = self.fun.upper_bounds
        print("Function info: fun= {} with dim = {}" .format(self.fun, self.dim))
        
        self.generation = 0
        range = self.lbounds - self.ubounds
        center = self.lbounds + range / 2
        x0 = center + 0.8 * range * (np.random.rand(self.dim)-0.5)
        self.X = self.lbounds + ((self.ubounds - self.lbounds) * np.random.rand(self.NP, self.dim))
        self.X[0, :] = x0
        #self.F = [self.fun(x) for x in self.X]
        self.F = np.apply_along_axis(self.fun, 1, self.X)
        #self.u = [[0 for z in range(int(self.dim))] for k in range(int(self.NP))]
        self.u = np.full((self.NP, self.dim), 0.0)
        self.F1 = np.apply_along_axis(self.fun, 1, self.u)
        #self.F1 = np.zeros(int(self.NP));
        self.budget -= self.NP
        
        # X and F are updated after each individual evaluation; population, used to pick random solutions, and copy_F are updated after a population is evaluated.
        self.population = np.copy(self.X)
        self.copy_F = np.copy(self.F)
        
        # Best in current population
        self.best = np.argmin(self.copy_F)
        # Best so far fitness and position
        self.best_so_far = self.copy_F[self.best]
        self.best_so_far_position = self.population[self.best]
        # Worst so far candidate
        self.worst_so_far = np.max(self.copy_F)
        # Best parent in parent population
        self.fmin = np.min(self.copy_F)
        # median of parent population
        self.fmedian = np.median(self.copy_F)
        
        self.i = 0;
        self.r = select_samples(self.NP, self.i, 5)
        
        #self.window = [[np.inf for j in range(self.number_metric)] for i in range(self.window_size)]; self.window = np.array(self.window);
        self.window = np.full((self.window_size, self.number_metric), np.inf); self.window[:, 0].fill(-1)
        self.gen_window = []
        self.third_dim = []
        self.opu = np.ones(self.NP) * 4
        
        # Randomly selects from [0,dim-1] of size NP
        self.fill_points = np.random.randint(self.dim, size = self.NP)
        
        self.pop_average = np.average(self.copy_F)
        self.pop_std = np.std(self.copy_F)
        
        ob = np.zeros(99); self.copy_ob = np.zeros(64)
        
        self.max_dist = distance.euclidean(self.lbounds, self.ubounds)
        self.max_std = np.std((np.repeat(self.best_so_far, self.NP/2), np.repeat(self.worst_so_far, self.NP/2)))
        self.stagnation_count = 0;
        
        self.fun_index += 1
        if self.fun_index == len(self.func_choice):
            self.fun_index = 0
            random.shuffle(self.func_choice)
        
        return ob


