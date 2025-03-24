import numpy as np
import pandas as pd
import sys


num_states = int(sys.argv[1])
num_output_symbols = int(sys.argv[2])
num_iter = int(sys.argv[3])
start = [float(line.strip()) for line in open(sys.argv[4],'r')]
trans = pd.read_csv(sys.argv[5], header=None, delim_whitespace=True).to_numpy()
emit = pd.read_csv(sys.argv[6], header=None, delim_whitespace=True).to_numpy()
data = pd.read_csv(sys.argv[7], header=None, delim_whitespace=True).to_numpy()

# converts "o1" "o2" etc. to their 0-based int equivalent for calculations
def data_to_int():
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i,j] = int(data[i,j][1:]) - 1


def forward(o):
    num_rows = len(o)
    arr = [[0 for i in range(num_states)] for i in range(num_rows)]
    for t in range(num_rows):
        if t == 0:
            for i in range(len(start)):
                arr[0][i] = start[i]
        else:
            for j in range(num_states):
                summation = 0
                for i in range(num_states-1):
                    summation += arr[t-1][i] * trans[i,j] * emit[i,o[t-1]]
                arr[t][j] = summation
    return arr


def backward(o):
    num_rows = len(o)
    arr = [[0 for i in range(num_states)] for i in range(num_rows+1)]
    arr[num_rows][num_states-1] = 1
    for t in range(num_rows-1,-1,-1):
        for i in range(num_states-1):
            if t == num_rows-1:
                arr[t][i] = trans[i,num_states-1] * emit[i,o[t]]
            else:
                summation = 0
                for j in range(num_states-1):
                    summation += trans[i,j] * emit[i,o[t]] * arr[t+1][j]
                arr[t][i] = summation
    return arr


def pti(F,B):
    arr = [[0 for i in range(num_states)] for i in range(len(F))]
    summation = 0
    for t in range(len(arr)):
        for i in range(len(arr[0])):
            summation += F[t][i] * B[t][i]
    for t in range(len(arr)):
        for i in range(num_states):
            arr[t][i] = F[t][i] * B[t][i] / summation
    return arr, summation


def ptij(F,B,summation,o):
    arr = [[[0 for j in range(num_states)] for i in range(num_states-1)] for t in range(len(F))]
    for t in range(len(arr)):
        for i in range(num_states-1):
            for j in range(num_states):
                arr[t][i][j] = F[t][i] * trans[i,j] * emit[i,o[t]] * B[t+1][j] / summation
    return arr

def calc_gi(arr):
    return [arr[0][i] for i in range(num_states)]


def calc_gij(arr, obs):
    temp = [[0 for j in range(num_states)] for i in range(num_states-1)]
    for i in range(num_states-1):
        for j in range(num_states):
            for t in range(len(obs)):
                temp[i][j] += arr[t][i][j]
    return temp

def calc_gio(arr, obs):
    temp = [[0 for o in range(emit.shape[1])] for i in range(num_states-1)]
    for t in range(len(obs)):
        o = obs[t]
        for i in range(num_states-1):
            temp[i][o] += arr[t][i]
    return temp


def E_step():
    gi = [[] for i in range(data.shape[0])]
    gij = [[]for i in range(data.shape[0])]
    gio = [[]for i in range(data.shape[0])]
    for o in range(data.shape[0]):
        F = forward(data[o,:])
        B = backward(data[o,:])
        apti, summ = pti(F, B)
        aptij = ptij(F, B, summ, data[o,:])
        gi[o] = calc_gi(apti)
        gij[o] = calc_gij(aptij,data[o,:])
        gio[o] = calc_gio(apti,data[o,:])

    return gi, gij, gio

def start_update(arr):
    sum1 = 0
    for k in range(num_states):
        for o in range(len(arr)):
            sum1 += arr[o][k]

    for i in range(num_states):
        sum2 = 0
        for o in range(len(arr)):
            sum2 += arr[o][i]
        start[i] = sum2/sum1


def trans_update(arr):
    for i in range(num_states-1):
        sum1 = 0
        for k in range(num_states):
            for o in range(len(arr)):
                sum1 += arr[o][i][k]

        for j in range(num_states):
            sum2 = 0
            for o in range(len(arr)):
                sum2 += arr[o][i][j]
            trans[i,j] = sum2/sum1

def emit_update(arr):
    for i in range(num_states-1):
        sum1 = 0
        for o1 in range(emit.shape[1]):
            for o2 in range(len(arr)):
                sum1 += arr[o2][i][o1]

        for o1 in range(emit.shape[1]):
            sum2 = 0
            for o2 in range(len(arr)):
                sum2 += arr[o2][i][o1]
            emit[i,o1] = sum2/sum1


def M_step(gi,gij,gio):
    start_update(gi)
    trans_update(gij)
    emit_update(gio)


def display_data(iteration):
    print("After iteration " + str(iteration+1) + ":")
    for i in range(len(start)):
        print("pi_" + str(i+1) + ": %0.4f" % start[i], end=" ")
    print()
    for i in range(trans.shape[0]):
        for j in range(trans.shape[1]):
            print("a_" + str(i+1) + str(j+1) + ": %0.4f" % float(trans[i,j]), end=" ")
        print()
    for i in range(emit.shape[0]):
        for j in range(emit.shape[1]):
            print("b_" + str(i+1) + "(o" + str(j+1) + "): %0.4f" % float(emit[i,j]), end=" ")
        print()


def EM():
    for iteration in range(num_iter):
        i,ij,io = E_step()
        M_step(i,ij,io)
        display_data(iteration)


data_to_int()
EM()
