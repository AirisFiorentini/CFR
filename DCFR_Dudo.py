import numpy as np
from pytreemap import TreeMap
from typing import List
# from Kuhn_s_poker_matrix_v import MNode
import math
import utils

class MDudoNode:
    # TODO: inheritance
    def __init__(self, NUM_ACTIONS: int, isClaimed: List[bool], NUM_SIDES: int = 6, NUM_HANDS: int = 2):
        self.positiveRegretSum = np.zeros((NUM_SIDES, NUM_ACTIONS))
        self.negativeRegretSum = np.zeros((NUM_SIDES, NUM_ACTIONS))
        self.strategy = np.zeros((NUM_SIDES, NUM_ACTIONS))
        self.strategySum = np.zeros((NUM_SIDES, NUM_ACTIONS))
        self.infoSet = ""
        self.NUM_ACTIONS = NUM_ACTIONS
        self.NUM_SIDES = NUM_SIDES
        self.NUM_HANDS = NUM_HANDS

        self.isClaimed = isClaimed

    # Get current information set mixed strategy through regret-matching
    def getStrategy(self,
                    t: int,
                    gamma: float,
                    realizationWeight: np.ndarray,
                    active_player_n: int,
                    curr_player_n: int) -> np.ndarray:

        _gamma = (t / (t + 1)) ** gamma
        if active_player_n == curr_player_n:
            self.strategySum *= _gamma

        regretSum = self.positiveRegretSum + self.negativeRegretSum
        normalizingSum = np.zeros(self.NUM_SIDES)
        for k in range(self.NUM_SIDES):
            for a in range(self.NUM_ACTIONS):
                self.strategy[k][a] = regretSum[k][a] if regretSum[k][a] > 0 else 0
                normalizingSum[k] += self.strategy[k][a]

        for k in range(self.NUM_SIDES):
            for a in range(self.NUM_ACTIONS):
                if normalizingSum[k] > 0:
                    self.strategy[k][a] /= normalizingSum[k]
                else:
                    self.strategy[k][a] = 1.0 / self.NUM_ACTIONS
                # refresh only for the current player
                if active_player_n == curr_player_n:
                    self.strategySum[k][a] += realizationWeight[k] * self.strategy[k][a]
        return self.strategy

    # Get average information set mixed strategy across all training iterations
    def getAverageStrategy(self) -> np.ndarray:
        avgStrategy = np.zeros((self.NUM_SIDES, self.NUM_ACTIONS))
        normalizingSum = np.zeros(self.NUM_SIDES)
        for k in range(self.NUM_SIDES):
            normalizingSum[k] += np.sum(self.strategySum[k])
        for k in range(self.NUM_SIDES):
            if normalizingSum[k] > 0:
                avgStrategy[k] = self.strategySum[k] / normalizingSum[k]
            else:
                avgStrategy[k] = 1 / self.NUM_ACTIONS
        return avgStrategy


class MDudoTrainer:
    # Dudo definitions
    def __init__(self):
        # Dudo definitions of 2 6-sided dice
        self.nodeMap = TreeMap()
        self.NUM_SIDES = 6
        self.NUM_ACTIONS = (2 * self.NUM_SIDES) + 1
        self.DUDO = self.NUM_ACTIONS - 1

        self.claimNum = ([1] * 6) + ([2] * 6)
        self.claimRank = [2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1]

    # convert Dudo information set to an integer (binary nums)
    def infoSetToInt(self, isClaimed: List[bool]) -> int:
        str_num = ''
        for i in range(len(isClaimed)):
            if isClaimed[i]:
                str_num += '1'
            else:
                str_num += '0'
        return int(str_num, base=2)

    # convert Dudo claim history to a String
    def claimHistoryToString(self, isClaimed: List[bool]) -> str:
        sb = ""
        for a in range(self.NUM_ACTIONS):
            if isClaimed[a]:
                if len(sb) > 0:
                    sb += ','
                sb += str(self.claimNum[a])
                sb += '*'
                sb += str(self.claimRank[a])
        return sb

    def count_param(self,
                    phi: float,
                    iter_n: int) -> float:
        if math.isinf(phi) and phi > 0:
            return 1
        elif math.isinf(phi) and phi < 0:
            return 0
        else:
            return (iter_n ** phi) / (iter_n ** phi + 1)

    # info set node class definitions (m_node class)
    # Counterfactual regret minimization iteration
    def m_dcfr(self,
               iter_n: int,
               isClaimed: List[bool],
               p0: np.ndarray,
               p1: np.ndarray,
               curr_player_n: int,
               alpha: float,
               beta: float,
               gamma: float) -> np.ndarray:  # history -> isClaimed

        plays = isClaimed.count(True)
        player = plays % 2
        # return payoff for terminal states
        if isClaimed[self.DUDO]:
            doubted = self.NUM_ACTIONS - 2 - isClaimed[self.NUM_ACTIONS - 2::-1].index(True)
            cN = self.claimNum[doubted]  # quantity
            cR = self.claimRank[doubted]  # value
            realDoubtedRankQuantity = np.zeros((self.NUM_SIDES, self.NUM_SIDES))
            for i in range(self.NUM_SIDES):
                for j in range(self.NUM_SIDES):
                    dice = [i + 1, j + 1]  # '1' <- 0 in arrays
                    realDoubtedRankQuantity[i][j] = dice.count(cR) + dice.count(1) if cR != 1 else dice.count(cR)

            U = np.zeros((self.NUM_SIDES, self.NUM_SIDES))
            # payoffs: +1 || -1 for the first player (only!)
            for i in range(self.NUM_SIDES):
                for j in range(self.NUM_SIDES):
                    if realDoubtedRankQuantity[i][j] >= cN:

                        U[i][j] = 1
                    else:
                        U[i][j] = -1
            if player == 1:
                return -U
            else:
                return U


        infoSet = str(self.infoSetToInt(isClaimed))
        # <Get information set node or create it if nonexistent>
        node = self.nodeMap.get(infoSet)
        AfterTrueIndex = self.NUM_ACTIONS - isClaimed[self.NUM_ACTIONS - 1::-1].index(True) if True in isClaimed else 0
        if node is None:
            node = MDudoNode(self.NUM_ACTIONS - AfterTrueIndex if AfterTrueIndex > 0 else 12, isClaimed)
            node.infoSet = infoSet
            self.nodeMap.put(infoSet, node)

        # For each action, recursively call cfr with additional history and probability
        strategy = node.getStrategy(iter_n, gamma, p0 if player == 0 else p1, player, curr_player_n)
        util = np.zeros((self.NUM_SIDES, self.NUM_SIDES, self.NUM_ACTIONS))
        nodeUtil = np.zeros((self.NUM_SIDES, self.NUM_SIDES))
        for a in range(node.NUM_ACTIONS):
            nextHistory = isClaimed.copy()
            iter = AfterTrueIndex + a
            nextHistory[iter] = True
            if player == 0:
                util[:, :, a] = self.m_dcfr(iter_n, nextHistory, p0 * strategy[:, a], p1, curr_player_n,
                                            alpha, beta, gamma)
                for i in range(self.NUM_SIDES):
                    nodeUtil[i, :] += util[i, :, a] * strategy[i, a]
            else:
                util[:, :, a] = self.m_dcfr(iter_n, nextHistory, p0, p1 * strategy[:, a], curr_player_n,
                                            alpha, beta, gamma)
                for j in range(self.NUM_SIDES):
                    nodeUtil[:, j] += util[:, j, a] * strategy[j, a]

        # For each action, compute and accumulate counterfactual regret
        # refresh only current player regret if it's their move
        if curr_player_n == player:
            _alpha = self.count_param(alpha, iter_n)
            _beta = self.count_param(beta, iter_n)
            for a in range(node.NUM_ACTIONS):
                node.positiveRegretSum *= _alpha
                node.negativeRegretSum *= _beta
                regret = util[:, :, a] - nodeUtil
                if player == 0:
                    r_new = np.dot(regret, p1)
                else:
                    r_new = np.dot(p0, -regret)
                r_new_sign = r_new >= 0

                for i in range(node.NUM_SIDES):
                    if r_new_sign[i]:
                        node.positiveRegretSum[i, a] += r_new[i]
                    else:
                        node.negativeRegretSum[i, a] += r_new[i]
        return nodeUtil

    def train(self,
              iterations: int = 100):
        results = []
        util = np.zeros((6, 6))
        for i in range(1, iterations + 1):
            print("iteration: ", i)
            for player in range(2):
                startClaims = [False] * self.NUM_ACTIONS
                util += self.m_dcfr(i, startClaims, np.array([1] * 6), np.array([1] * 6), player,
                                    math.inf, -math.inf, 2)
                # TODO: call exploitability counting here for the opponent
                # CFR+: math.inf, -math.inf, 2
                # 1.5, 0, 2: 1500
                # 1, 1, 1
                # util += self.m_dcfr(i, startClaims, np.array([1] * 6), np.array([1] * 6), 1, math.inf, -math.inf, 2)
            cur_res = np.sum(util / i / 36 / 2)
            if i % 10 == 0:
                # cur_res = np.sum(util / iterations / 36)
                results.append(cur_res)
        utils.save_result_to_file(results, "DCFR_Dudo")

        print("The number of iterations: ", iterations)
        agv = util / iterations / 2 / 36
        print(np.sum(agv))
        print(agv)

        return self

    # TODO: describe a node
    def getNode(self, isClaimed: List[bool]) -> str:
        infoSet = str(self.infoSetToInt(isClaimed))
        if infoSet in self.nodeMap:
            return self.nodeMap.get(infoSet).strategy
        else:
            return MDudoNode(self.NUM_ACTIONS, [False] * 13).toString()

    def getNodeStrategy(self, die: int, isClaimed: List[bool]) -> str:
        infoSet = str(self.infoSetToInt(isClaimed))
        if infoSet in self.nodeMap:
            return self.nodeMap.get(infoSet).strategy[die - 1]
        else:
            return "Not found"


if __name__ == '__main__':
    TrainRes = MDudoTrainer().train(200)

    # startClaims = [False] * 13
    # startClaims[2] = True
    # print(TrainRes.getNodeStrategy(2, startClaims))
