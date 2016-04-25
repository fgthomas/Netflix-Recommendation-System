"""
RBM.py
Contains the class definition of a Restricted Boltzmann Machine designed to make
predictions on items based on user ratings
Author: Forest Thomas
"""
import math
import pickle
from random import *

class RBM:
    def __init__(self, V, F, K, rate):
        """
        Initializes the RBM
        V - length of the input vector
        F - the number of hidden units
        K - the number of ratings
        rate - the learning rate
        """
        #set number of visible and hidden nodes
        self.F = F
        self.K = K
        self.V = V
        
        #initialize Weight Matrix with normally distributed values
        self.W = []
        for i in range(0, V):
            self.W.append([])
            for j in range(0, F):
                self.W[i].append([])
                for k in range(0, K):
                    self.W[i][j].append(normalvariate(0, .01))
                    
        #initialize biases
        self.vis_bias = []
        for i in range(0, V):
            self.vis_bias.append([])
            for k in range(0, K):
                self.vis_bias[i].append(normalvariate(0, .01))

        self.hid_bias = []
        for j in range(0, F):
            self.hid_bias.append(normalvariate(0, .01))

        #set learning rate
        self.eps = rate

        #create hidden states
        self.h = []
        for j in range(0, F):
            self.h.append(0)

    def set_input_biases(self, b):
        for i in range(0, self.V):
            for k in range(0, self.K):
                self.vis_bias[i][j] = b[i][j]

    def learn_batch(self, T, V, rated):
        """
        Does T steps of T-step contrastive divergence
        """
        # initialization
        del_W = []
        for i in range(0, self.V):
            del_W.append([])
            for j in range(0, self.F):
                del_W[i].append([])
                for k in range(0, self.K):
                    del_W[i][j].append(k)
        del_vis_bias = []
        for i in range(0, self.V):
            del_vis_bias.append([])
            for k in range(0, self.K):
                del_vis_bias[k].append(0)

        del_hid_bias = []
        for j in range(0, self.F):
            del_hid_bias.append(0)

        # run all vectors in the batch
        for v in V:
            v_last = v
            h = []
            for t in range(0, T):
                h = self.get_hidden_states(v_last, rated)
                v_last = self.rebuild_input(h)
                
            #get change in visible bias
            for i in range(0, self.V):
                for k in range(0, self.K):
                    del_vis_bias[i][k] += eps*(v_orig[i][k] - v_last[i][j])

            #get change in hidden bias
            hdata = self.get_hidden_probabilities(v_orig, rated)
            hmodel = self.get_hidden_probabilities(v_last, rated)
            for j in range(0, self.F):
                del_hid_bias[j] += eps*(hdata[j] - hmodel[j])

            #get changes in weights
            for i in rated:
                for j in range(0, self.F):
                    for k in range(0, self.K):
                        del_W[i][j][k] += eps*(hdata[j]*v[i][k] - hmodel*v_last[i][k])
        #update weights
        for i in range(0, self.V):
            for k in range(0, self.K):
                self.vis_bias[i][k] += del_vis_bias[i][k]
        for j in range(0, self.F):
            self.hid_bias[j] += del_hid_bias[j]
        for i in range(0, self.V):
            for j in range(0, self.F):
                for k in range(0, self.K):
                    self.W[i][j][k] += del_W[i][j][k]


    def get_hidden_probabilities(self, v, rated):
        """
        Gives the probabilities
        """
        probs = []
        W = self.W
        for j in range(0, self.F):
            s = 0
            for i in rated:
                for k in range(0, self.K):
                    s += v[i][k]*W[i][j][k]
            probs.append(sig(self.hid_bias[j] + s))
        return probs
    
        
    def rebuild_input(self, h):
        """
        Rebuilds the input vector, given the set of hidden states
        returns probabilities of v
        """
        b = self.vis_bias
        v = []
        for i in range(0, self.V):
            v.append([])
            for k in range(0, self.K):
                prob = bias[i][k]
                for j in range(0, self.F):
                    prob += h[j]*W[i][j][k]
                prob = sig(prob)
                if prob > random():
                    v[i].append(1)
                else:
                    v[i].append(0)
        return v

        
    def get_hidden_states(self, v, rated):
        """
        gives the hidden states of the rbm given an input vector
        """
        W = self.W
        h = []
        for j in range(0, self.F):
            s = 0
            for i in rated:
                for k in range(0, self.K):
                    s += v[i][k]*W[i][j][k]
            prob = sig(self.hid_bias[j] + s)
            if prob > random():
                h.append(1)
            else:
                h.append(0)
        return h

                
    def get_prediction(self, v, q, k, rated):
        """
        Gets the probability of rating r of movie m given movies watched, watched
        """
        W = self.W
        Gamma = math.e ** (v[q][k]*self.vis_bias[q][k])
        prod = 1
        for j in range(0, self.F):
            s = 1 # 1 + stuff in for loop
            for i in rated:
                for r in range(0, self.K):
                    s += v[i][r]*W[i][j][r] + v[q][k]*W[q][j][k] + self.hid_bias[j]
            prod *= s
        return Gamma * prod
        
        pass

    def save_RBM(self, filename):
        """
        Saves the current RBM to a file
        """
        with open(filename, "w") as fout:
            pickle.dump(self, fout)

    def load_RBM(filename):
        """
        Loads an RBM from a file
        """
        with open(filename, "r") as fin:
            rbm = pickle.load(fin)
        return rbm

    def set_hidden_states(self, v, rated):
        """
        sets the hidden states of the rbm given an input vector
        """
        W = self.W
        for j in range(0, self.F):
            s = 0
            for i in rated:
                for k in range(0, self.K):
                    s += v[i][k]*W[i][j][k]
            prob = sig(self.hid_bias[j] + s)
            if prob > random():
                self.h[j] = 1
            else:
                self.h[j] = 0

    def set_hidden_by_prob(self, probabilities):
        """
        Sets the hidden states using a vector of probabilities
        """
        for j in range(0, self.F):
            if probabilities[j] > random():
                self.h[j] = 1
            else:
                self.h[j] = 0

    
def sig(x):
    return 1 / (1 + math.e ** -x)
