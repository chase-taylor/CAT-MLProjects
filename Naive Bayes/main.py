import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import sys

train = pd.read_table(sys.argv[1])
test = pd.read_table(sys.argv[2])

# split df based on class
def split(df):
    df0, df1 = pd.DataFrame(), pd.DataFrame()
    for i in range(len(df)):
        if df.iat[i,-1] == 0:
            #df0 = df0.append(df.iloc[i:i+1,:-1],ignore_index=True)
            df0 = pd.concat([df0,df.iloc[i:i+1,:-1]],ignore_index=True)
        if df.iat[i,-1] == 1:
            #df1 = df1.append(df.iloc[i:i+1,:-1],ignore_index=True)
            df1 = pd.concat([df1,df.iloc[i:i+1,:-1]],ignore_index=True)
    return df0, df1

# calculate probabilities of each attribute in a dataframe
def calc_probabilities(df):
    arr, prob0, prob1 = [], [], []
    total_examples = len(df)

    for i in range(len(df.columns)):
        arr.append(int(df.iloc[:,i:i+1].sum()))

    for i in arr:
        prob0.append(1 - (i/total_examples))
        prob1.append(i/total_examples)

    return prob0, prob1

def display_probabilities(names,a0,a1,c):
    # print class
    print("P(" + str(names[-1]) + "=" + str(c[0]) + ")=%.2f" % c[1], end=" ")
    # print attributes
    for i in range(len(a0)):
        print("P(" + str(names[i]) + "=0|" + str(c[0]) + ")=%.2f" % a0[i], end=" ")
        print("P(" + str(names[i]) + "=1|" + str(c[0]) + ")=%.2f" % a1[i], end=" ")
    print()

# tests a dataframe off of the learned dataset
def test_probability(df, test_set_name, a0_c0, a1_c0, a0_c1, a1_c1, c0, c1):
    correct = 0
    total = len(df)
    for i in range(total):
        row = df.iloc[i:i+1,:-1]
        result = np.argmax([test_probability_helper(row,a0_c0,a1_c0,c0),test_probability_helper(row,a0_c1,a1_c1,c1)])
        if result == df.iat[i,-1]:
            correct = correct + 1
    correctness = 100 * correct / total
    print("\nAccuracy on " + test_set_name + " set (" + str(total) + " instances): %.2f" % correctness + "%")

# calculates probability of a class using given attributes
def test_probability_helper(row,a0,a1,c):
    prob = 0
    if c[1] != 0:
        prob = np.log(c[1])
    for i in range(len(row.columns)):
        val = row.iat[0,i]
        if val == 0 and a0[i] > 0:
            prob = prob + np.log(a0[i])
        elif val == 1 and a1[i] > 0:
            prob = prob + np.log(a1[i])
        else:
            prob = prob - 10

    return prob


# split dataframe by class
df_c0, df_c1 = split(train)
# calculate probability of each class and attributes
class_counts = [len(df_c0),len(df_c1)]
c0 = [0, class_counts[0] / (class_counts[0] + class_counts[1])]
c1 = [1, 1 - c0[1]]
a0_c0, a1_c0 = calc_probabilities(df_c0)
a0_c1, a1_c1 = calc_probabilities(df_c1)
# display probabilities
col_headers = list(train.columns.values)
display_probabilities(col_headers,a0_c0,a1_c0,c0)
display_probabilities(col_headers,a0_c1,a1_c1,c1)
# test
test_probability(train, "training", a0_c0, a1_c0, a0_c1, a1_c1, c0, c1)
test_probability(test, "test", a0_c0, a1_c0, a0_c1, a1_c1, c0, c1)

