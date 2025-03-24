import numpy as np
import random

from agent import QLearningAgent

# constants
TILTED = 0
MAX_STEPS = 100000
MAX_TRIALS = 10000
GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = (MASSPOLE + MASSCART)
LENGTH = 0.5  # actually half the pole's length
POLEMASS_LENGTH = (MASSPOLE * LENGTH)
FORCE_MAG = 10.0
TAU = 0.02  # seconds between state updates
FOURTHIRDS = 1.3333333333333

# actions
FORWARD_ACCEL = 1
BACKWARD_ACCEL = 0
NO_ACCEL = -1

# switches
DEBUG = False


class PoleBalancing:
    def __init__(self, track_length=2.4):
        self.x = 0  # cart position, meters
        self.x_dot = 0  # cart velocity
        self.theta = 0  # pole angle, radians
        self.theta_dot = 0  # pole angular velocity
        self.track_length = track_length  # track length on each side

    def step(self, action):
        if action == FORWARD_ACCEL:
            force = FORCE_MAG
        elif action == BACKWARD_ACCEL:
            force = -FORCE_MAG
        else:
            raise NotImplementedError
        costheta = np.cos(self.theta)
        sintheta = np.sin(self.theta)
        temp = (force + POLEMASS_LENGTH * self.theta_dot * self.theta_dot * sintheta) / TOTAL_MASS
        thetaacc = (GRAVITY * sintheta - costheta * temp) / (
                LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta / TOTAL_MASS))
        xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS
        # Update the four state variables, using Euler's method.
        self.x += TAU * self.x_dot
        self.x_dot += TAU * xacc
        self.theta += TAU * self.theta_dot
        self.theta_dot += TAU * thetaacc

    def reset_cart(self):
        self.x = 0
        self.x_dot = 0
        self.theta = 0
        self.theta_dot = 0

    def get_states(self):
        return self.x, self.x_dot, self.theta, self.theta_dot

    def fail(self):
        twelve_degrees = 0.2094384
        if self.theta < -twelve_degrees or \
                self.theta > twelve_degrees or \
                self.x < -self.track_length or \
                self.x > self.track_length:
            return True
        return False


def main():
    # set different configurations here
    lr = 0.2                # learning rate
    gamma = 0.99            # discount factor
    track_length = 2.4      # how far the ends of the track are from the origin.
                            # e.g., while track_length is 2.4,
                            # the x-coordinate of the left end of the track is -2.4,
                            # the x-coordinate of the right end of the track is 2.4,
                            # and the x-coordinate of the the cart is 0 initially.
    epsilon = 0.005         # epsilon for the mixed policy
    policy = 'mixed'        # policy can be 'mixed' or 'greedy'

    # init
    problem = PoleBalancing()
    agent = QLearningAgent(lr, gamma, track_length, epsilon, policy)

    steps = 0
    failures = 0
    best_steps = 0
    best_trial = 0
    t = 1

    random.seed(11)
    np.random.seed(11)

    if DEBUG:
        print('{:<8}{:<8}{:<8}{:<8}{:<8}{:<8}'.format('steps', 'x', 'x_dot', 'theta', 'theta_dot', 'action'))
    while steps < MAX_STEPS and failures < MAX_TRIALS:
        cur_state = problem.get_states()
        steps += 1
        action = agent.get_action(*problem.get_states())
        if DEBUG:
            print('{:<8}{:<8.2f}{:<8.2f}{:<8.2f}{:<8.2f}{:<8}'.format(steps, *cur_state, action))
        problem.step(action)
        next_state = problem.get_states()
        if problem.fail():
            failures += 1
            if steps > best_steps:
                best_steps = steps
                best_trial = failures
            if failures % t == 0:
                print(f'Trial {failures} was {steps} steps. Current max number of steps is {best_steps}')
            # Call agent with negative feedback for learning
            agent.update_Q(cur_state, action, next_state, -1)
            agent.reset()
            problem.reset_cart()
            steps = 0
        else:
            agent.update_Q(cur_state, action, next_state, 0)

    if failures >= MAX_TRIALS:
        print(f'Pole not balanced. Stopping after {failures} failures.')
        print(f'High water mark: {best_steps} steps in trial {best_trial}.')
    else:
        print(f'Pole balanced successfully for at least {steps - 1} steps in trial {failures + 1}.')


if __name__ == '__main__':
    main()