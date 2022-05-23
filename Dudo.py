import random
from pytreemap import TreeMap
from typing import List
from Kuhn_s_poker import Node


# TODO: implement DudoNode instead of Node
"""
class DudoNode(Node):
    def __init__(self,
                 NUM_ACTIONS: int,
                 isClaimed: List[bool],
                 die=0
                 ):
        self.regretSum = [0.0] * NUM_ACTIONS
        self.strategy = [0.0] * NUM_ACTIONS
        self.strategySum = [0.0] * NUM_ACTIONS
        self.infoSet = ""
        self.NUM_ACTIONS = NUM_ACTIONS
        # for output Dudo
        self.die = die
        self.isClaimed = isClaimed
"""

class DudoTrainer:
    def __init__(self):  # Dudo definitions
        # Dudo definitions of 2 6-sided dice
        self.nodeMap = TreeMap()
        self.NUM_SIDES = 6
        self.NUM_ACTIONS = (2 * self.NUM_SIDES) + 1
        self.DUDO = self.NUM_ACTIONS - 1

        self.claimNum = ([1] * 6) + ([2] * 6)
        self.claimRank = [2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1]

    def rollDice(self) -> List[int]:
        return [random.randint(1, self.NUM_SIDES), random.randint(1, self.NUM_SIDES)]

    def infoSetToInt(self, playerRoll: int, isClaimed: List[bool]) -> int:  # convert Dudo information set to an integer
        infoSetNum = playerRoll
        for a in range(self.NUM_ACTIONS - 1).__reversed__():
            infoSetNum = 2 * infoSetNum + (1 if isClaimed[a] else 0)
        return infoSetNum

    def claimHistoryToString(self, isClaimed: List[bool]) -> str:  # convert Dudo claim history to a String
        sb = ""
        for a in range(self.NUM_ACTIONS):
            if isClaimed[a]:
                if len(sb) > 0:
                    sb += ','
                sb += str(self.claimNum[a])
                sb += '*'
                sb += str(self.claimRank[a])
        return sb

    # info set node class definitions (node class)
    # Counterfactual regret minimization iteration
    def cfr(self, dice: List[int], isClaimed: List[bool], p0: float, p1: float) -> float:  # history -> isClaimed
        plays = isClaimed.count(True)
        player = plays % 2
        # return payoff for terminal states
        if isClaimed[self.DUDO]:
            doubted = self.NUM_ACTIONS - 2 - isClaimed[self.NUM_ACTIONS - 2::-1].index(True)
            # print("doubted", doubted)
            cN = self.claimNum[doubted]
            cR = self.claimRank[doubted]
            realDoubtedRankQuantity = dice.count(cR) + dice.count(1) if cR != 1 else dice.count(cR)
            if realDoubtedRankQuantity >= cN:
                return 1  # Dudo loses -1,
            else:
                # вычесть
                return -1  # last stake player loses |actual - claimed|

        infoSet = str(self.infoSetToInt(dice[player], isClaimed))
        # <Get information set node or create it if nonexistent>
        node = self.nodeMap.get(infoSet)
        AfterTrueIndex = self.NUM_ACTIONS - isClaimed[self.NUM_ACTIONS - 1::-1].index(True) if True in isClaimed else 0
        if node is None:
            node = Node(self.NUM_ACTIONS - AfterTrueIndex if AfterTrueIndex > 0 else 12, isClaimed, dice[player])
            node.infoSet = infoSet
            # output feint with ears
            # node.die = dice[player]
            # node.isClaimed = isClaimed
            self.nodeMap.put(infoSet, node)

        # For each action, recursively call cfr with additional history and probability
        strategy = node.getStrategy(p0 if player == 0 else p1)
        util = [0.0] * self.NUM_ACTIONS
        nodeUtil = 0
        for i in range(node.NUM_ACTIONS):
            nextHistory = isClaimed.copy()
            iter = AfterTrueIndex + i
            nextHistory[iter] = True
            if player == 0:
                util[i] = -self.cfr(dice, nextHistory, p0 * strategy[i], p1)
            else:
                util[i] = -self.cfr(dice, nextHistory, p0, p1 * strategy[i])
            nodeUtil += strategy[i] * util[i]
        # For each action, compute and accumulate counterfactual regret
        for a in range(node.NUM_ACTIONS):
            regret = util[a] - nodeUtil
            node.regretSum[a] += (p1 if player == 0 else p0) * regret
        return nodeUtil

    # Train Dudo
    def train(self, iterations: int):
        util = 0.0
        for i in range(iterations):
            if i % 100 == 0:
                print(i)
            dice = self.rollDice()
            startClaims = [False] * self.NUM_ACTIONS
            util += self.cfr(dice, startClaims, 1, 1)
        print("The number of iterations: ", iterations)
        print("Average game value: ", util / iterations)
        # for n in self.nodeMap.values():  # print cards + history
        #     print(n.die, self.claimHistoryToString(n.isClaimed), n.toString(), sep='|')
        #     print()
        return self

    def getNode(self, die: int, isClaimed: List[bool]) -> str:
        infoSet = str(self.infoSetToInt(die, isClaimed))
        if infoSet in self.nodeMap:
            return self.nodeMap.get(infoSet).toString()
        else:
            return Node(self.NUM_ACTIONS, [False] * 13).toString()  # change properly -1?


# DudoTrainer main method
if __name__ == '__main__':
    blabla = [False, False, False, False, False, True, True, False, False, False, False, False, False]
    f_bla = [False] * 13
    iterations = 4000
    TrainRes = DudoTrainer().train(iterations)
    # print(TrainRes.getNode(6, [False]*13))
    # print(TrainRes.getNode(3, blabla))
    # for i in range(20):
    #     print(TrainRes.rollDice())
