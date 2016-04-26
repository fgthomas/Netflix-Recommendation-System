"""
RBM.py
Contains the class definition of a Restricted Boltzmann Machine designed to make
predictions on items based on user ratings
Author: Forest Thomas
Notes:
    This RBM is designed to be used for sparse, two-dimensional inputs
    Therefore, most input vectors are going to be two dimensional
    eg: input[m][r]
        -m is the movie (or item)
        -r is the rating for that item
    The sparsity is handled by ignoring inputs with no ratings, therefore the calculations
    will be dynamic based on which movies have been rated.
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
        K - the number of possible ratings
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
                self.vis_bias[i].append(0)

        self.hid_bias = []
        for j in range(0, F):
            self.hid_bias.append(0)

        #set learning rate
        self.eps = rate

    def set_input_biases(self, b):
        """
        Sets the input biases to the vector b
        b should be the same dimensions as the length of the input vector
        b - the new bias vector
        return - None
        """
        for i in range(0, self.V):
            for k in range(0, self.K):
                self.vis_bias[i][k] = b[i][k]

    def learn_batch(self, T, V):
        """
        Does T steps of T-step contrastive divergence
        T - The number of steps in CD
        V - the set of input vectors
        return - None
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
                del_vis_bias[i].append(0)

        del_hid_bias = []
        for j in range(0, self.F):
            del_hid_bias.append(0)

        eps = self.eps / len(V)
        # run all vectors in the batch
        for v in V:
            rated = getRated(v)
            v_last = v
            h = []
            for t in range(0, T):
                h = self.get_hidden_states(v_last, rated)
                v_last = self.rebuild_input(rated, h)

            #get change in visible bias
            for i in rated:
                for k in range(0, self.K):
                    del_vis_bias[i][k] += eps*(v[i][k] - v_last[i][k])

            #get change in hidden bias
            hdata = self.get_hidden_probabilities(v, rated)
            hmodel = self.get_hidden_probabilities(v_last, rated)
            for j in range(0, self.F):
                del_hid_bias[j] += eps*(hdata[j] - hmodel[j])

            #get changes in weights
            for i in rated:
                for j in range(0, self.F):
                    for k in range(0, self.K):
                        del_W[i][j][k] += eps*(hdata[j]*v[i][k] - hmodel[j]*v_last[i][k])
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
        Gives the probabilities of the hidden layer given an input vector
        v - an input vector
        rated - the movies rated in the input vector
        return - a list of probabilities for the hidden layer
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
    
        
    def rebuild_input(self, rated, h):
        """
        Rebuilds the input vector, given the set of hidden states
        returns probabilities of v
        h - The a binary vector representing the hidden layer
        """
        b = self.vis_bias
        v = [[]]*self.V
        W = self.W
        for i in rated:
            for k in range(0, self.K):
                prob = b[i][k]
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
        v - input vector
        rated - movies rated in the input vector (list of indices)
        return - a binary vector representing the hidden layer
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


    def save_RBM(self, filename):
        """
        Saves the current RBM to a file
        filename - the name of the file to save to
        """
        with open(filename, "wb") as fout:
            pickle.dump(self, fout)

    def load_RBM(filename):
        """
        Loads an RBM from a file
        filename - the name of the file to load from
        """
        with open(filename, "rb") as fin:
            rbm = pickle.load(fin)
        return rbm
        

    def get_prediction(self, movie, h):
        """
        Gives the prediction of a movie given a hidden layer
        This is calculated by taking Expected value over all ratings for the movie
        movie - the movie to predict
        h - the hidden layer
        return - a prediction for the movie
        """
        prob = 0
        for k in range(0, 5):
            s = self.vis_bias[movie][k]
            for j in range(0, self.F):
                s += h[j]*self.W[movie][j][k]
            prob += (sig(s)*(k+1))
        return prob

    
    def get_highest_rating(self, v, rated):
        """
        Gives a movie recommendation based on an input vector
        v -  the input vector
        rated - the movies rated in the input vector
        return - the index of the movie recommended along with the probability
        that the machine gives for that movie at the maximum rating
        """
        h = self.get_hidden_probabilities(v, rated)
        highest = [] #contains probabilities for the highest rating available
        for i in range(0, self.V):
            highest.append(0)
            if i in rated:
                continue
            highest[i] = self.get_prediction(i, h)
            print("movie: {0} prediction: {1}".format(i, highest[i]))
            
        # find highest probability
        m, index = (highest[0], 0)
        for i in range(1, len(highest)):
            if highest[i] > m:
                (m, index) = (highest[i], i)
        return index, m
        
def sig(x):
    """
    Sigmoid function
    x - real number input
    return - sigmoid(x)
    """
    return 1 / (1 + math.e ** -x)

def getRated(v):
    """
    Gives the indices of movies that hae a rating (i.e. not None) in the vector
    v - vector containing mostly None with numbers
    return - an array of indices that have been rated in the vector
    """
    rated = []
    for i in range(0, len(v)):
        if v[i] != None:
            rated.append(i)
    return rated
