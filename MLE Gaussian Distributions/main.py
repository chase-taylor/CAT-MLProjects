import math
import numpy as np
import pandas as pd
import sys

RT2PI = math.sqrt(2*math.pi)

class Gaussian:
    def __init__(self,data):
        self.points = data
        self.probs = [0 for i in range(len(data))]
        self.gammas = [0 for i in range(len(data))]
        self.mean = 0
        self.variance = 0
        self.prior = 0

    # calculates probability of each point in the gaussian
    def calc_prob(self):
        stdev = math.sqrt(self.variance)
        for i in range(len(self.points)):
            self.probs[i] = (1 / (stdev * RT2PI)) * math.e ** (-0.5 * ((self.points[i] - self.mean) / stdev) ** 2)

    # calculates gamma for each point in the gaussian
    def calc_gammas(self, gaussians):
        for i in range(len(self.points)):
            if self.probs[i] != 0:
                sum = 0
                tmp = self.prior * self.probs[i]
                for j in range(len(gaussians)):
                    sum += gaussians[j].prior * gaussians[j].probs[i]
                self.gammas[i] = tmp / sum
            else:
                self.gammas[i] = 0

    # calculates mean based off of gamma values and point values
    def calc_mean(self):
        summation = 0
        for i in range(len(self.points)):
            summation += self.gammas[i] * self.points[i]
        self.mean = summation / sum(self.gammas)

    # calculates variance (σ^2)
    def calc_variance(self):
        summation = 0
        for i in range(len(self.points)):
            summation += self.gammas[i] * (self.points[i] - self.mean) ** 2
        self.variance = summation / sum(self.gammas)

    # calculates prior probability off of gammas
    def calc_prior(self):
        self.prior = sum(self.gammas)/len(self.gammas)

# creates gaussians with starting parameters
def initialize_gaussians(data, num_gauss):
    arr = [Gaussian(data) for i in range(num_gauss)]
    # starting points for each gaussian
    for i in range(len(data)):
        arr[i % num_gauss].probs[i] = 1
    for i in range(len(arr)):
        # setting mean
        summation = 0
        count = 0
        for j in range(len(arr[i].points)):
            if arr[i].probs[j] != 0:
                summation += arr[i].points[j]
                count += 1
        arr[i].mean = summation / count
        # setting variance
        tmp = 0
        for j in range(len(arr[i].points)):
            if arr[i].probs[j] != 0:
                tmp += (arr[i].points[j] - arr[i].mean) ** 2
        arr[i].variance = tmp / count
        # setting prior probability
        arr[i].prior = sum(arr[i].probs) / len(arr[i].probs)
    return arr

# updates probability tables for each gaussian
def calc_probs(gaussians):
    for gaussian in gaussians:
        gaussian.calc_prob()
    for i in range(len(gaussians[0].points)):
        summ = 0
        for gaussian in gaussians:
            summ += gaussian.probs[i]
        for gaussian in gaussians:
            gaussian.probs[i] = gaussian.probs[i]/summ

def E_step(gaussians):
    calc_probs(gaussians)

def M_step(gaussians):
    for gaussian in gaussians:
        gaussian.calc_gammas(gaussians)
        gaussian.calc_mean()
        gaussian.calc_variance()
        gaussian.calc_prior()

# performs EM learning
def EM(data, num_gauss, num_iter):
    gaussians = initialize_gaussians(data, num_gauss)
    for i in range(num_iter + 1):
        print("After iteration %i:" % i)
        for j in range(num_gauss):
            print("Gaussian %d: mean = %.4f, variance = %.4f, prior = %.4f" % (j + 1, gaussians[j].mean, gaussians[j].variance, gaussians[j].prior))
        print()
        E_step(gaussians)
        M_step(gaussians)


# open file
f = open(sys.argv[1],'r')
arr = []
for line in f:
    arr.append(float(line.strip()))

EM(arr, int(sys.argv[2]), int(sys.argv[3]))