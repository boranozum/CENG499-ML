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
        c_t = np.log(np.sum(dp[0, :]))

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
            c_t += np.log(np.sum(dp[t, :]))

            # Normalization of the current time step
            dp[t, :] /= np.sum(dp[t, :])

        # Return the sum of ln c_t
        return c_t


    def viterbi_log(self, O: list):
        """
        :param O: is an array of discrete (integer) observations, i.e. [0, 2,1 ,3, 4]
        :return: the tuple (Q*, ln P(Q*|O,λ)), Q* is the most probable state sequence for the given O
        """

        pass