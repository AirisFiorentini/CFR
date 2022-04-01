from pytreemap import TreeMap
from typing import List
import numpy as np

# Kuhn Poker definitions
# PASS = 0
# BET = 1
# NUM_ACTIONS = 2
# NUM_HANDS = 2


class MNode:
    #  TODO: decide the dims of regretSum, strategy, strategySUM
    # Kuhn node definitions for matrices
    def __init__(self,
                 NUM_ACTIONS: int,
                 NUM_CARDS: int,
                 NUM_HANDS: int = 2) -> object:
        # dim: num_hands * num_actions
        self.regretSum = np.zeros((NUM_CARDS, NUM_ACTIONS))  # NUM_HANDS
        # dim of strategy and strategySum?
        self.strategy = np.zeros((NUM_CARDS, NUM_ACTIONS))  # diag(vector  (sigma_i)) NUM_CARDS, NUM_CARDS, NUM_ACTIONS)
        self.strategySum = np.zeros((NUM_CARDS, NUM_ACTIONS))
        self.infoSet = ""
        self.NUM_ACTIONS = NUM_ACTIONS
        self.NUM_CARDS = NUM_CARDS
        self.NUM_HANDS = NUM_HANDS

    # TODO: check getStrategy()
    # Get current information set mixed strategy through regret-matching
    def getStrategy(self,
                    realizationWeight: np.ndarray) -> np.ndarray:
        normalizingSum = np.zeros(self.NUM_CARDS)
        for k in range(self.NUM_CARDS):
            for a in range(self.NUM_ACTIONS):
                self.strategy[k][a] = self.regretSum[k][a] if self.regretSum[k][a] > 0 else 0
                normalizingSum[k] += self.strategy[k][a]
        for k in range(self.NUM_CARDS):
            for a in range(self.NUM_ACTIONS):
                if normalizingSum[k] > 0:
                    self.strategy[a] /= normalizingSum[k]
                else:
                    self.strategy[k][a] = 1.0 / self.NUM_ACTIONS
                self.strategySum[k][a] += realizationWeight[k] * self.strategy[k][a]
                # np.dot(realizationWeight, self.strategy)?
        return self.strategy

    # Get average information set mixed strategy across all training iterations
    def getAverageStrategy(self) -> np.ndarray:
        avgStrategy = np.zeros((self.NUM_CARDS, self.NUM_ACTIONS))
        normalizingSum = np.zeros(self.NUM_CARDS)  # self.NUM_ACTIONS
        for k in range(self.NUM_CARDS):
            normalizingSum[k] += np.sum(self.strategySum[k])
        for k in range(self.NUM_CARDS):
            if normalizingSum[k] > 0:
                avgStrategy[k] = self.strategySum[k] / normalizingSum[k]
            else:
                avgStrategy[k] = 1 / self.NUM_ACTIONS  # np.ones(self.NUM_ACTIONS)
        return avgStrategy

    # TODO: show the results
    def toString(self) -> str:  # Get information set string representation
        return '{:4}: {},\n regret {}'.format(self.infoSet, self.getAverageStrategy(), self.regretSum)


class MKuhnTrainer:
    # Kuhn Poker definitions
    def __init__(self):
        self.nodeMap = TreeMap()
        self.PASS = 0
        self.BET = 1
        self.NUM_ACTIONS = 2
        self.NUM_CARDS = 3

    # Information set node class definition (node class above)
    # Counterfactual regret minimization iteration
    def m_cfr(self,
              history: str,
              p0: np.ndarray,
              p1: np.ndarray) -> np.ndarray:

        plays = len(history)
        player = plays % 2
        # opponent = 1 - player
        # return payoff for terminal states
        if plays > 1:
            terminalPass = history[plays - 1] == 'p'
            doubleBet = history[plays - 2: plays] == "bb"
            U = np.zeros((self.NUM_CARDS, self.NUM_CARDS))
            is_terminal = False
            for i in range(self.NUM_CARDS):
                for j in range(self.NUM_CARDS):
                    if i != j:
                        if player == 0:
                            if terminalPass:
                                is_terminal = True
                                if history == "pp":
                                    U[i][j] = 1 if i > j else -1  # isPlayerCardHigher[i][j]
                                else:
                                    U[i][j] = 1
                            elif doubleBet:
                                is_terminal = True
                                U[i][j] = 2 if i > j else -2
                        else:  # player_1
                            if terminalPass:
                                is_terminal = True
                                if history == "pp":
                                    U[i][j] = 1 if j > i else -1
                                else:
                                    U[i][j] = 1
                            elif doubleBet:
                                is_terminal = True
                                U[i][j] = 2 if j > i else -2
            if is_terminal:
                return U

        infoSet = history
        # print(history)
        # Get information set node or create it if nonexistent
        node = self.nodeMap.get(infoSet)
        if node is None:
            node = MNode(self.NUM_ACTIONS, self.NUM_CARDS)
            node.infoSet = infoSet
            self.nodeMap.put(infoSet, node)

        # For each action, recursively call m_cfr with additional history and probability
        strategy = node.getStrategy(p0 if player == 0 else p1)
        util = np.zeros((self.NUM_CARDS, self.NUM_CARDS, self.NUM_ACTIONS))  # ??? self.NUM_ACTIONS->self.NUM_CARDS
        nodeUtil = np.zeros((self.NUM_CARDS, self.NUM_CARDS))  # self.NUM_CARDS
        # TODO: count NodeUtil
        for a in range(self.NUM_ACTIONS):
            nextHistory = history + ("p" if a == 0 else "b")
            # print(nextHistory)
            if player == 0:
                # util[a] = -self.cfr(cards, nextHistory, p0 * strategy[a], p1)
                util[:, :, a] = -self.m_cfr(nextHistory, p0 * strategy[:, a], p1)
                # nodeUtil += np.dot(util[:, :, a], strategy[:, a])
                for i in range(self.NUM_CARDS):
                    nodeUtil[i, :] += util[i, :, a] * strategy[i, a]
            else:
                # util[a] = -self.cfr(cards, nextHistory, p0, p1 * strategy[a])
                util[:, :, a] = -self.m_cfr(nextHistory, p0, p1 * strategy[:, a])
                # nodeUtil += np.dot(strategy[:, a], -util[:, :, a])
                for j in range(self.NUM_CARDS):
                    nodeUtil[:, j] += strategy[j, a] * util[:, j, a]

        # For each action, compute and accumulate counterfactual regret
        for a in range(self.NUM_ACTIONS):
            regret = util[:, :, a] - nodeUtil
            if player == 0:
                node.regretSum[:, a] += np.dot(p1, regret)
            else:
                node.regretSum[:, a] += np.dot(p0, -regret)
        return nodeUtil

    # TODO: add differentiation of players' regrets
    # train Kuhn poker
    def train(self,
              iterations: int):
        util = np.zeros((3, 3))
        for i in range(iterations):
            util += self.m_cfr("", np.array([1] * 3), np.array([1] * 3))  # params?
        agv = util / iterations / 6  # *1/6 - no need?
        print(np.sum(agv))
        print("Average game value: ", agv)
        for n in self.nodeMap.values():
            print(n.toString())
        return self

    def one_run(self):
        self.m_cfr("", np.array([1] * 3), np.array([1] * 3))
        print()
        for n in self.nodeMap.values():
            print(n.toString())


if __name__ == '__main__':
    # MKuhnTrainer().one_run()
    trainer = MKuhnTrainer().train(1000)
    """
    for i in range(self.NUM_CARDS):
        for j in range(self.NUM_CARDS):
            if i != j:
    """
