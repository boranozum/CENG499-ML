import numpy as np
class HMM:
    def __init__(self, A, B, Pi):
        self.A = A
        self.B = B
        self.Pi = Pi

    def forward_log(self, O: list):
        """
        :param O: is the sequence (an array of) discrete (integer) observations, i.e. [0, 2,1 ,3, 4]
        :return: ln P(O|λ) score for the given observation, ln: natural logarithm
        """

        # Initialization of dp table
        dp = np.zeros((len(O), self.A.shape[0]))

        # Initialization of first column with multiplying the initial state probabilities with
        # the probabilities of the first observation in the sequence
        dp[0, :] = self.Pi * self.B[:, O[0]]

        # Sum of the probabilities of the first time step
        c_t = np.log(1.0/np.sum(dp[0, :]))

        # Normalization of the first time step
        dp[0, :] /= np.sum(dp[0, :])

        # Loop over the time steps
        for t in range(1, len(O)):
            # Loop over the states
            for s in range(self.A.shape[0]):
                # Calculate the probability of the current state given the previous state with the current observation
                # in the sequence
                dp[t, s] = np.sum(dp[t-1, :] * self.A[:, s]) * self.B[s, O[t]]

            # Sum of the probabilities of the current time step
            c_t += np.log(1.0/np.sum(dp[t, :]))

            # Normalization of the current time step
            dp[t, :] /= np.sum(dp[t, :])

        # Return the sum of ln c_t
        return -c_t


    def viterbi_log(self, O: list):
        """
        :param O: is an array of discrete (integer) observations, i.e. [0, 2,1 ,3, 4]
        :return: the tuple (Q*, ln P(Q*|O,λ)), Q* is the most probable state sequence for the given O
        """

        # Initialization of delta array which will hold the probabilities and previous states
        # with the first time step considering the equation (7) in the assignment pdf, and previous states of the first
        # time step is initialized with None
        delta = [{}]
        for s in range(self.A.shape[0]):
            delta[0][s] = {"prob": np.log(self.Pi[s]) + np.log(self.B[s, O[0]]), "prev": None}

        # Loop over the time steps
        for t in range(1, len(O)):
            delta.append({})

            # Loop over the states
            for s in range(self.A.shape[0]):

                # Determine the maximum probability of transitioning to the current state given the previous state and
                # the previous state with the maximum probability considering the equation (8) in the assignment pdf
                max_prob = delta[t-1][0]["prob"] + np.log(self.A[0, s])
                max_prev = 0
                for prev_s in range(1, self.A.shape[0]):
                    prob = delta[t-1][prev_s]["prob"] + np.log(self.A[prev_s, s])
                    if prob > max_prob:
                        max_prob = prob
                        max_prev = prev_s

                # Update the delta array with the maximum probability and the previous state with the maximum probability
                delta[t][s] = {"prob": max_prob + np.log(self.B[s, O[t]]), "prev": max_prev}

        sequence = []

        # Find the maximum probability value of the last time step
        max_prob = max(value["prob"] for value in delta[-1].values())
        prev = None

        # Loop over the states in the last time step and determine the state with the maximum probability value
        for s, s_info in delta[-1].items():
            if s_info["prob"] == max_prob:
                sequence.append(s)
                prev = s
                break

        # Backtrack the previous states to determine the most probable state sequence
        for t in range(len(delta) - 2, -1, -1):
            sequence.insert(0, delta[t + 1][prev]["prev"])
            prev = delta[t + 1][prev]["prev"]

        return max_prob, sequence