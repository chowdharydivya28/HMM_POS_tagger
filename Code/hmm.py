from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################

        # for s in range(S):
        #     alpha[s][0] = self.B[s][self.obs_dict[Osequence[0]]] * self.pi[s]
        # for t in range(1, L):
        #     for s in range(S):
        #         summation = sum((alpha[k][t - 1] * self.A[k][s]) for k in range(S))
        #         alpha[s][t] = self.B[s][self.obs_dict[Osequence[t]]] * summation
        # print("alpha before:", alpha)

        for t in range(L):
            states = self.B[:, self.obs_dict[Osequence[t]]]
            # print("states", states)
            if t == 0:
                alpha[:, t] = states * self.pi
                # print("alpha", alpha)
            else:
                alpha[:, t] = states * np.matmul(self.A.T, alpha[:, t-1])
                # print("alpha:", alpha)

        # print("alpha before:", alpha)
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ##################################################
        # for s in range(S):
        #     beta[s][L - 1] = 1
        # for t in range(L - 2, -1, -1):
        #     for s in range(S):
        #         beta[s][t] = sum(
        #             (self.A[s][k] * self.B[k][self.obs_dict[Osequence[t + 1]]] * beta[k][t + 1]) for k in range(S))
        # print("beat before :", beta)

        for t in range(L-1, -1, -1):
            if t == L-1:
                beta[:, t] = 1.0
            else:
                states = self.B[:, self.obs_dict[Osequence[t+1]]]
                # print("states:", states)
                beta[:, t] = np.matmul(self.A, beta[:, t+1]*states)

                # print("beta:", beta)

        # ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        S = len(self.pi)
        alpha = self.forward(Osequence)
        # print("alpha in sequence prob ", alpha)
        # for s in range(S):
        #     prob += alpha[s][len(Osequence) - 1]
        # print("sequence probability is :", prob)

        prob = np.sum(alpha[:, -1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        prob_O = self.sequence_prob(Osequence)
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        # for t in range(L):
        #     for s in range(S):
        #         prob[s][t] = (alpha[s][t] * beta[s][t]) / prob_O
        prob = np.array([[(alpha[s][t] * beta[s][t]) / prob_O for t in range(L)] for s in range(S)])
        ###################################################
        return prob

    # TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        prob_O = self.sequence_prob(Osequence)
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        # for t in range(L - 1):
        #     for s in range(S):
        #         for s_dash in range(S):
        #             prob[s][s_dash][t] = (alpha[s][t] * self.A[s][s_dash] * self.B[s_dash][self.obs_dict[Osequence[
        #                 t + 1]]] * beta[s_dash][t + 1]) / prob_O
        prob = np.array([[[(alpha[s][t] * self.A[s][s_dash] * self.B[s_dash][self.obs_dict[Osequence[
            t + 1]]] * beta[s_dash][t + 1]) / prob_O for t in range(L - 1)] for s_dash in range(S)] for s in range(
            S)])
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        S = len(self.pi)
        L = len(Osequence)
        delta = np.zeros([S, L])
        del_ta = np.zeros([S, L])
        path_idx = np.zeros(L, dtype=int)
        # for s in range(S):
        #     delta[s][0] = self.pi[s] * self.B[s][self.obs_dict[Osequence[0]]]
        # for t in range(1, L):
        #     for s in range(S):
        #         delta[s][t] = self.B[s][self.obs_dict[Osequence[t]]] * max(
        #             (self.A[k][s] * delta[k][t - 1]) for k in range(S))
        #         del_ta[s][t] = np.argmax([self.A[k][s] * delta[k][t - 1] for k in range(S)])

        for t in range(L):
            states = self.B[:, self.obs_dict[Osequence[t]]]
            if t == 0:
                delta[:, t] = self.pi * states
            else:
                delta[:, t] = states * np.max(np.multiply(self.A.T, delta[:, t-1]), axis=1)
                del_ta[:, t] = np.argmax(np.multiply(self.A.T, delta[:, t-1]), axis=1)

        # backtracking
        state_key_list = list(self.state_dict.keys())
        state_val_list = list(self.state_dict.values())
        # max_index_state = [delta[k][L - 1] for k in range(S)]
        # path_idx[L - 1] = np.argmax([delta[k][L - 1] for k in range(S)])
        path_idx[L - 1] = np.argmax(delta[:, L - 1])
        # print("path_idx", path_idx)
        # print("del_ta", del_ta)
        for t in range(L - 1, 0, -1):
            # print("L is :", t)
            path_idx[t - 1] = del_ta[path_idx[t]][t]
            # print("path_idx", path_idx)
        path = [state_key_list[state_val_list.index(i)] for i in path_idx]
        ###################################################
        return path
