import math
import random

import numpy as np

FORWARD_ACCEL = 1
BACKWARD_ACCEL = 0


class QLearningAgent:
    def __init__(self, lr, gamma, track_length, epsilon=0, policy='greedy'):
        """
        A function for initializing your agent
        :param lr: learning rate
        :param gamma: discount factor
        :param track_length: how far the ends of the track are from the origin.
            e.g., while track_length is 2.4,
            the x-coordinate of the left end of the track is -2.4,
            the x-coordinate of the right end of the track is 2.4,
            and x-coordinate of the the cart is 0 initially.
        :param epsilon: epsilon for the mixed policy
        :param policy: can be 'greedy' or 'mixed'
        """
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.track_length = track_length
        self.policy = policy
        random.seed(11)
        np.random.seed(11)
        # q table for discretized
        self.q_table = [ [0]*2 for i in range(2223)]
        pass

    def reset(self):
        # resets object aside from q table
        temp = self.q_table
        self.__init__(self.lr, self.gamma, self.track_length, self.epsilon, self.policy)
        self.q_table = temp

    def get_action(self, x, x_dot, theta, theta_dot):
        """
        main.py calls this method to get an action from your agent
        :param x: the position of the cart
        :param x_dot: the velocity of the cart
        :param theta: the angle between the cart and the pole
        :param theta_dot: the angular velocity of the pole
        :return:
        """
        if self.policy == 'mixed' and random.random() < self.epsilon:
            action = random.sample([FORWARD_ACCEL, BACKWARD_ACCEL], 1)[0]
        else:
            i = self.discretize(x,x_dot,theta,theta_dot)
            action = self.argmax(self.q_table,i)

        return action

    def update_Q(self, prev_state, prev_action, cur_state, reward):
        """
        main.py calls this method so that you can update your Q-table
        :param prev_state: previous state, a tuple of (x, x_dot, theta, theta_dot)
        :param prev_action: previous action, FORWARD_ACCEL or BACKWARD_ACCEL
        :param cur_state: current state, a tuple of (x, x_dot, theta, theta_dot)
        :param reward: reward, 0.0 or -1.0
        e.g., if we have S_i ---(action a, reward)---> S_j, then
            prev_state is S_i,
            prev_action is a,
            cur_state is S_j,
            rewards is reward.
        :return:
        """
        # discretizes both states and updates q table
        x, x_dot, theta, theta_dot = prev_state
        d1 = self.discretize(x, x_dot, theta, theta_dot)
        x, x_dot, theta, theta_dot = cur_state
        d2 = self.discretize(x, x_dot, theta, theta_dot)
        self.q_table[d1][prev_action] = (1 - self.lr) * self.q_table[d1][prev_action] + self.lr * (reward + self.gamma * max(self.q_table[d2][0],self.q_table[d2][1]))
        return

    def discretize(self, x, x_dot, theta, theta_dot):
        # vars
        index = 0
        third_track = self.track_length/3
        twelve_degrees = 0.2094384
        three_degrees = twelve_degrees/4

        # discretizing variables
        if x < -third_track:
            pass
        elif x > third_track:
            index += 2000
        else:
            index += 1000

        if x_dot < -1/3:
            pass
        elif x_dot > 1/3:
            index += 200
        else:
            index += 100

        if theta < -three_degrees:
            pass
        elif theta > three_degrees:
            index += 20
        else:
            index += 10

        if theta_dot < -0.5:
            pass
        elif theta_dot > 0.5:
            index += 2
        else:
            index += 1

        return index

    def argmax(self, arr, i):
        # argmax method
        if arr[i][1] > arr[i][0]:
            return 1
        else:
            return 0









if __name__ == '__main__':
    pass