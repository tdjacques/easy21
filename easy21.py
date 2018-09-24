from random import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

import numpy as np
import csv

from game import Easy21

def alpha(s, a, na):  # step size
    return 1. / na[s[0]][s[1]][a]


def fq(state, action, theta):
    # State Q when using approximation of game-state Phi
    return np.inner(phi(state, action), theta)


def policy(s, q, n0, ns):
    epsilon = float(n0) / (n0 + ns[s[0]][s[1]])
    if q[s[0]][s[1]][1] > q[s[0]][s[1]][0]:
        prob_hit = 1 - epsilon / 2
    else:
        prob_hit = epsilon / 2
    if prob_hit > random():
        action = 1
    else: 
        action = 0
    return action


def policy_phi(state, theta, epsilon):
    if fq(state, 1, theta) > fq(state, 0, theta):
        prob_hit = 1 - epsilon/2
    else:
        prob_hit = epsilon/2
    if prob_hit > random():
        action = 1
    else:
        action = 0
    return action


def phi_test(i_player, i_dealer, i_action, state, action):
    test_dealer = 0 < float(state[1] - 3 * i_dealer) / 4. <= 1
    test_player = 0 < float(state[0] - 3 * i_player) / 6. <= 1
    test_action = action != i_action
    if test_player and test_dealer and test_action:
        return 1
    else:
        return 0


def phi(state, action):
    return [phi_test(i, j, k, state, action) for k in range(2) for j in range(3) for i in range(6)]


def sarsa(n_iterations, n0):
    # global Q
    print 'Beginning Sarsa run'
    # initialise error array
    err = np.zeros([11])
    err0 = []
    err1 = []

    # initialise playthrough
    lambda1 = np.array([l for l in np.arange(0., 1.1, 0.1)])
    for il in range(11):

        # initialise value function
        #  [PLAYER DEALER ACTION]
        Q = np.random.rand(22, 11, 2)
        Q[0, :, :] = Q[:, 0, :] = 0

        # initialise counters
        Ns = np.zeros([22, 11])  # times state has been visited
        Na = np.zeros([22, 11, 2])  # times action has been selected

        print 'Beginning lambda = ' + str(lambda1[il])
        iEp = 0  # episode counter
        iReflect = iReflect0 = 1000  # Print every j-th episode
        while iEp < n_iterations:
            E = np.zeros([22, 11, 2])  # Eligibility traces
            episode = Easy21()
            action = policy(episode.state(), Q, n0, Ns)

            # Run the episode
            nStep = 0
            while not episode.is_terminal():

                init_state = episode.state()
                # Take action A, observe R, S'
                episode.step(action)

                # Choose A' from S' using policy derived from Q
                next_action =  policy(episode.state(), Q, n0, Ns)

                # update error
                [s0,s1] = init_state
                [sPrime0,sPrime1] = episode.state()

                # Update counters
                Ns[s0][s1] += 1  # Counter for this state
                Na[s0][s1][action] += 1  # counter for this state-action pair

                delta = episode.reward + Q[sPrime0][sPrime1][next_action] - Q[s0][s1][action]

                E[s0][s1][action] += 1
                alpha_t = alpha(init_state, action, Na)

                Q += (alpha_t * delta) * E
                E *= lambda1[il]

                action = next_action
                nStep += 1

            # Error analysis
            if iEp == iReflect:
                print  'end of run ' + str(iEp)

                err[il] = np.sum((Q - Qstar) ** 2)
                if lambda1[il] == 0.:
                    err0.append(err[il])
                if lambda1[il] == 1.:
                    err1.append(err[il])
                print 'error = ' + str(err[il])
                iReflect += iReflect0
            iEp += 1

        # Export array
        V = np.array([[0, 0, 0]])
        for i in range(22):
            for j in range(11):
                V = np.append(V, [[i, j, np.amax(Q, axis=2)[i][j]]], axis=0)

        V = np.delete(V, 0, 0)

        with open('out_sarsa_' + str(lambda1[il]) + '.csv', 'wb') as f:
            writer = csv.writer(f, delimiter=',')
            for i in range(len(V)):
                writer.writerow(V[i])

    with open('out_sarsa_error.csv', 'wb') as g:
        writer = csv.writer(g, delimiter=',')
        for i in range(11):
            writer.writerow([lambda1[i], err[i]])
    g.close()

    with open('out_sarsa_error0.csv', 'wb') as g:
        writer = csv.writer(g, delimiter=',')
        for i in range(len(err0)):
            writer.writerow([1000 * (i + 1), err0[i]])
    g.close()

    with open('out_sarsa_error1.csv', 'wb') as g:
        writer = csv.writer(g, delimiter=',')
        for i in range(len(err1)):
            writer.writerow([1000 * (i + 1), err1[i]])
    g.close()

    return Q


def mc_control(n_iterations, n0):
    # initialise value function
    #  [PLAYER DEALER ACTION]
    Q = np.array([[[random() for j in range(2)] for i in range(11)] for k in range(22)])
    Q[0, :, :] = Q[:, 0, :] = 0

    # initialise counters
    # Index zero is card 10?
    Ns = np.array([[0. for j in range(11)] for i in range(22)])  # times state has been visited
    Na = np.array([[[0. for j in range(2)] for i in range(11)] for k in range(22)])  # times action has been selected

    # initialise playthrough
    iEpisode = 0  # Number of episodes
    iReflect = iReflect0 = 10000 #print every iRelect'th episode
    while iEpisode < n_iterations:
        episode = Easy21() #initial state
        st = [episode.state()]  # State history
        at = []  # action history
        # Run the episode
        while not episode.is_terminal():
            action =  policy(episode.state(),  Q, n0, Ns)
            episode.step(action)
            st = np.append(st, [episode.state()], axis=0)
            at = np.append(at, action)
        G = episode.reward  # final reward

        # Update counters and value functions for each state in episode
        for t in range(len(st) - 1):
            s0 = int(st[t][0])
            s1 = int(st[t][1])
            a = int(at[t])
            Ns[s0][s1] += 1  # Counter for this state
            Na[s0][s1][a] += 1  # counter for this state-action pair
            Q[s0][s1][a] += alpha(st[t], a, Na) * (G - Q[s0][s1][a])  # Update reward
        if iEpisode == iReflect:
            print  'end of run ' + str(iEpisode)
            iReflect += iReflect0
        iEpisode += 1

    # Export array
    V = np.array([[0, 0, 0]])
    for i in range(22):
        for j in range(11):
            V = np.append(V, [[i, j, max(Q[i][j][0], Q[i][j][1])]], axis=0)

    V = np.delete(V, 0, 0)

    with open('out_V.csv', 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        for i in range(len(V)):
            writer.writerow(V[i])
    f.close

    with open('out_Qstar.csv', 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        for i in range(len(Q)):
            writer.writerow(Q[i])
    f.close

    Qstar = Q

    # Qstar = np.array([[[random() for j in range(2)] for i in range(11)] for k in range(22)])
    Vstar = np.amax(Qstar, axis=2)

    # plot value function
    x = np.arange(0, 11)
    y = np.arange(0, 22)
    xs, ys = np.meshgrid(x, y)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_wireframe(xs, ys, Vstar, rstride=1, cstride=1)
    plt.savefig('value.pdf', bbox_inches="tight")

    return Qstar


def sarsa_phi(n_iterations, n0, epsilon, alpha):
    # global Q
    print 'Beginning Sarsa run'
    # initialise error array
    err = np.zeros([11])
    err0 = []
    err1 = []

    # initialise playthrough
    lambda1 = np.array([l for l in np.arange(0., 1.1, 0.1)])
    for il in range(11):

        # initialise value function
        #  [PLAYER DEALER ACTION]
        theta = np.random.rand(36)

        # initialise counters
        Ns = np.zeros([22, 11])  # times state has been visited
        Na = np.zeros([22, 11, 2])  # times action has been selected

        print 'Beginning lambda = ' + str(lambda1[il])
        iEp = 0  # episode counter
        iReflect = iReflect0 = 1000  # Print every j-th episode
        while iEp < n_iterations:
            E = np.zeros(36)  # Eligibility traces
            episode = Easy21()
            action = policy_phi(episode.state(),theta,epsilon)

            # Run the episode
            nStep = 0
            while not episode.is_terminal():

                init_state = episode.state()
                # Take action A, observe R, S'
                episode.step(action)

                # Choose A' from S' using policy derived from Q
                next_action =  policy_phi(episode.state(), theta, epsilon)

                # update error
                [s0,s1] = init_state
                [sPrime0,sPrime1] = episode.state()

                # Update counters
                Ns[s0][s1] += 1  # Counter for this state
                Na[s0][s1][action] += 1  # counter for this state-action pair

                delta = episode.reward + fq(episode.state(),next_action,theta) - fq(init_state,action,theta)

                E += np.inner(np.ones(36),phi(init_state,action))

                theta += (alpha * delta) * E
                E *= lambda1[il]

                action = next_action
                nStep += 1

            # Error analysis
            if iEp == iReflect:
                print  'end of run ' + str(iEp)
                #print Q

                err[il] = 0
                for s0 in range(1,22):
                    for s1 in range(1,11):
                        for action in range(2):
                            err[il] += (fq([s0,s1],action,theta) - Qstar[s0][s1][action])**2
                #err[il] = np.sum((Q - Qstar) ** 2)
                if lambda1[il] == 0.:
                    err0.append(err[il])
                if lambda1[il] == 1.:
                    err1.append(err[il])
                print 'error = ' + str(err[il])
                iReflect += iReflect0
            iEp += 1

        # Export array
        V = [np.amax(theta[i],theta[i+18]) for i in range(18)]

        with open('out_sarsaPhi_' + str(lambda1[il]) + '.csv', 'wb') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(V)

    with open('out_sarsaPhi_error.csv', 'wb') as g:
        writer = csv.writer(g, delimiter=',')
        for i in range(11):
            writer.writerow([lambda1[i], err[i]])
    g.close()

    with open('out_sarsaPhi_error0.csv', 'wb') as g:
        writer = csv.writer(g, delimiter=',')
        for i in range(len(err0)):
            writer.writerow([1000 * (i + 1), err0[i]])
    g.close()

    with open('out_sarsaPhi_error1.csv', 'wb') as g:
        writer = csv.writer(g, delimiter=',')
        for i in range(len(err1)):
            writer.writerow([1000 * (i + 1), err1[i]])
    g.close()

    return 1

"""
Begin main function
"""

N0 = 200
n_iterations = 1 * 10 ** 6
Qstar = mc_control(n_iterations, N0)

n_iterations = 10 ** 5
sarsa_phi(n_iterations, N0,0.05,0.01)


