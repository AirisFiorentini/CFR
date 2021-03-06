from pytreemap import TreeMap
# from typing import List
import numpy as np
import math  # math.inf

# Kuhn Poker definitions
# PASS = 0
# BET = 1
# NUM_ACTIONS = 2
# NUM_HANDS = 2


class MNode:
    # Kuhn node definitions for matrices
    def __init__(self,
                 NUM_ACTIONS: int,
                 NUM_CARDS: int,
                 NUM_HANDS: int = 2):
        self.positiveRegretSum = np.zeros((NUM_CARDS, NUM_ACTIONS))
        self.negativeRegretSum = np.zeros((NUM_CARDS, NUM_ACTIONS))
        self.strategy = np.zeros((NUM_CARDS, NUM_ACTIONS))
        self.strategySum = np.zeros((NUM_CARDS, NUM_ACTIONS))
        self.infoSet = ""
        self.NUM_ACTIONS = NUM_ACTIONS
        self.NUM_CARDS = NUM_CARDS
        self.NUM_HANDS = NUM_HANDS

    # Get current information set mixed strategy through regret-matching
    def getStrategy(self,
                    t: int,
                    gamma: float,
                    realizationWeight: np.ndarray,
                    active_player_n: int,
                    curr_player_n: int) -> np.ndarray:

        regretSum = self.positiveRegretSum + self.negativeRegretSum

        _gamma = (t / (t + 1)) ** gamma
        if active_player_n == curr_player_n:
            self.strategySum *= _gamma

        normalizingSum = np.zeros(self.NUM_CARDS)
        for k in range(self.NUM_CARDS):
            for a in range(self.NUM_ACTIONS):
                self.strategy[k][a] = regretSum[k][a] if regretSum[k][a] > 0 else 0
                normalizingSum[k] += self.strategy[k][a]
        for k in range(self.NUM_CARDS):
            for a in range(self.NUM_ACTIONS):
                if normalizingSum[k] > 0:
                    self.strategy[k][a] /= normalizingSum[k]
                else:
                    self.strategy[k][a] = 1.0 / self.NUM_ACTIONS
                if active_player_n == curr_player_n:
                    self.strategySum[k][a] += realizationWeight[k] * self.strategy[k][a]
        return self.strategy

    # Get average information set mixed strategy across all training iterations
    def getAverageStrategy(self) -> np.ndarray:
        avgStrategy = np.zeros((self.NUM_CARDS, self.NUM_ACTIONS))
        normalizingSum = np.zeros(self.NUM_CARDS)
        for k in range(self.NUM_CARDS):
            normalizingSum[k] += np.sum(self.strategySum[k])
        for k in range(self.NUM_CARDS):
            if normalizingSum[k] > 0:
                avgStrategy[k] = self.strategySum[k] / normalizingSum[k]
            else:
                avgStrategy[k] = 1 / self.NUM_ACTIONS
        return avgStrategy

    # show the results with i-card
    def toString_m(self):  # Get information set string representation
        AvSt = self.getAverageStrategy()
        for i in range(3):
            print('{:4}: {},\n regret {}'.format(str(i + 1) + self.infoSet, AvSt[i],
                                                 (self.positiveRegretSum + self.negativeRegretSum)[i]))

    def toString(self) -> str:  # Get information set string representation
        return '{:4}: {},\n regret {}'.format(self.infoSet,
                                              self.getAverageStrategy(),
                                              self.positiveRegretSum + self.negativeRegretSum)


class MKuhnTrainer:
    # Kuhn Poker definitions
    def __init__(self):
        self.nodeMap = TreeMap()
        self.PASS = 0
        self.BET = 1
        self.NUM_ACTIONS = 2
        self.NUM_CARDS = 3

    def is_terminal(self, history: str) -> bool:
        return history in ['bp', 'bb', 'pp', 'pbb', 'pbp']

    def count_param(self,
                    phi: float,
                    iter_n: int) -> float:
        if math.isinf(phi) and phi > 0:
            return 1
        elif math.isinf(phi) and phi < 0:
            return 0
        else:
            return (iter_n ** phi) / (iter_n ** phi + 1)

    # Information set node class definition (node class above)
    # Counterfactual regret minimization iteration
    def m_dcfr(self,
               iter_n: int,
               history: str,
               p0: np.ndarray,
               p1: np.ndarray,
               curr_player_n: int,
               alpha: float,
               beta: float,
               gamma: float) -> np.ndarray:  # curr_player_n - the number of player, which we count regrets for

        plays = len(history)
        player = plays % 2  # active player
        # current player, we count for

        # return payoff for terminal states
        if plays > 1:
            if self.is_terminal(history):
                terminalPass = history[plays - 1] == 'p'
                doubleBet = history[plays - 2: plays] == "bb"
                U = np.zeros((self.NUM_CARDS, self.NUM_CARDS))
                for i in range(self.NUM_CARDS):
                    for j in range(self.NUM_CARDS):
                        if i != j:
                            if terminalPass:
                                if history == "pp":
                                    U[i][j] = 1 if i > j else -1
                                elif history == 'bp':
                                    U[i][j] = 1
                                else:  # pbp
                                    U[i][j] = -1
                            elif doubleBet:
                                U[i][j] = 2 if i > j else -2
                return U

        infoSet = history
        # Get information set node or create it if nonexistent
        node = self.nodeMap.get(infoSet)
        if node is None:
            node = MNode(self.NUM_ACTIONS, self.NUM_CARDS)
            node.infoSet = infoSet
            self.nodeMap.put(infoSet, node)

        # For each action, recursively call m_cfr with additional history and probability
        strategy = node.getStrategy(iter_n, gamma, p0 if player == 0 else p1, player, curr_player_n)
        util = np.zeros((self.NUM_CARDS, self.NUM_CARDS, self.NUM_ACTIONS))
        nodeUtil = np.zeros((self.NUM_CARDS, self.NUM_CARDS))
        for a in range(self.NUM_ACTIONS):
            nextHistory = history + ("p" if a == 0 else "b")
            if player == 0:
                util[:, :, a] = self.m_dcfr(iter_n, nextHistory, p0 * strategy[:, a], p1, curr_player_n,
                                            alpha, beta, gamma)
                for i in range(self.NUM_CARDS):
                    nodeUtil[i, :] += util[i, :, a] * strategy[i, a]
            else:
                util[:, :, a] = self.m_dcfr(iter_n, nextHistory, p0, p1 * strategy[:, a], curr_player_n,
                                            alpha, beta, gamma)
                for j in range(self.NUM_CARDS):
                    nodeUtil[:, j] += util[:, j, a] * strategy[j, a]

        # For each action, compute and accumulate counterfactual regret
        # refresh only current player regret if it's their move
        if curr_player_n == player:
            _alpha = self.count_param(alpha, iter_n)
            _beta = self.count_param(beta, iter_n)
            # print(_alpha, _beta)
            # if math.isinf(alpha) and alpha > 0:
            #     _alpha = 1
            # else:
            #     _alpha = iter_n ** alpha / (iter_n ** alpha + 1)
            # if math.isinf(beta) and beta > 0:
            #     _beta = 1
            # else:
            #     _beta = iter_n ** beta / (iter_n ** beta + 1)

            for a in range(self.NUM_ACTIONS):
                node.positiveRegretSum *= _alpha
                node.negativeRegretSum *= _beta

                regret = util[:, :, a] - nodeUtil
                if player == 0:
                    r_new = np.dot(regret, p1)
                else:
                    r_new = np.dot(p0, -regret)
                r_new_sign = r_new >= 0

                for i in range(node.NUM_CARDS):
                    if r_new_sign[i]:
                        node.positiveRegretSum[i, a] += r_new[i]
                    else:
                        node.negativeRegretSum[i, a] += r_new[i]
        return nodeUtil

    # train Kuhn poker
    def train(self,
              iterations: int):
        util = np.zeros((3, 3))
        for i in range(1, iterations + 1):
            for player_n in range(2):
                util += self.m_dcfr(i, "", np.array([1] * 3), np.array([1] * 3), player_n,
                                    math.inf, -math.inf, 2)
                # CFR+: math.inf, -math.inf, 2: 1500 // -0.055552097912113935
                # 1.5, 0, 2: 1500                   //-0.055527798514633075
                # 1, 1, 1
        agv = util / 2 / iterations / 6  # average game value
        print(np.sum(agv))
        print("Average game value: ", agv)
        for n in self.nodeMap.values():
            print(n.toString_m())
        return self


if __name__ == '__main__':
    trainer = MKuhnTrainer().train(5000)
