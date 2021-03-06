import random
from pytreemap import TreeMap
from typing import List

# Kuhn Poker definitions
# PASS = 0
# BET = 1
# NUM_ACTIONS = 2


class Node:  # унаследовать класс
    # Kuhn node definitions
    # TODO: remove dudo defs
    def __init__(self,
                 NUM_ACTIONS: int,
                 isClaimed: List[bool],
                 die=0):
        self.regretSum = [0.0] * NUM_ACTIONS
        self.strategy = [0.0] * NUM_ACTIONS
        self.strategySum = [0.0] * NUM_ACTIONS
        self.infoSet = ""
        self.NUM_ACTIONS = NUM_ACTIONS
        # for output Dudo
        self.die = die
        self.isClaimed = isClaimed

    # Get current information set mixed strategy through regret-matching
    def getStrategy(self,
                    realizationWeight: float) -> List[float]:
        normalizingSum = 0.0
        for a in range(self.NUM_ACTIONS):
            self.strategy[a] = self.regretSum[a] if self.regretSum[a] > 0 else 0
            normalizingSum += self.strategy[a]
        for a in range(self.NUM_ACTIONS):
            if normalizingSum > 0:
                self.strategy[a] /= normalizingSum
            else:
                self.strategy[a] = 1.0 / self.NUM_ACTIONS
            self.strategySum[a] += realizationWeight * self.strategy[a]
        return self.strategy

    # Get average information set mixed strategy across all training iterations
    def getAverageStrategy(self) -> List[float]:
        avgStrategy = [0.0] * self.NUM_ACTIONS
        normalizingSum = 0.0
        for a in range(self.NUM_ACTIONS):
            normalizingSum += self.strategySum[a]
        for a in range(self.NUM_ACTIONS):
            if normalizingSum > 0:
                avgStrategy[a] = self.strategySum[a] / normalizingSum
            else:
                avgStrategy[a] = 1.0 / self.NUM_ACTIONS
        return avgStrategy

    def toString(self) -> str:  # Get information set string representation
        return '{:4}: {}, regret {}'.format(self.infoSet, self.getAverageStrategy(), self.regretSum)  # str(get)


def shuffleCards(cards: List[int]):
    for c1 in range(len(cards) - 1, 0, -1):
        c2 = random.randint(0, c1)
        cards[c1], cards[c2] = cards[c2], cards[c1]


class KuhnTrainer:
    def __init__(self):  # Kuhn Poker definitions
        self.nodeMap = TreeMap()  # (string, node) #public TreeMap<String, Node> nodeMap = new TreeMap<String, Node>();
        self.PASS = 0
        self.BET = 1
        self.NUM_ACTIONS = 2

    # Information set node class definition (node class above)
    # Counterfactual regret minimization iteration
    def cfr(self, cards: List[int], history: str, p0: float, p1: float) -> float:
        plays = len(history)
        player = plays % 2
        opponent = 1 - player
        # return payoff for terminal states
        if plays > 1:
            terminalPass = history[plays - 1] == 'p'
            doubleBet = history[plays - 2: plays] == "bb"
            isPlayerCardHigher = cards[player] > cards[opponent]
            if terminalPass:
                if history == "pp":
                    return 1 if isPlayerCardHigher else -1
                else:
                    return 1
            elif doubleBet:
                return 2 if isPlayerCardHigher else -2
        infoSet = str(cards[player]) + history
        # Get information set node or create it if nonexistent
        node = self.nodeMap.get(infoSet)
        if node is None:
            node = Node(self.NUM_ACTIONS, [False] * 13)  # isClaimed is extra
            node.infoSet = infoSet
            self.nodeMap.put(infoSet, node)

        # For each action, recursively call cfr with additional history and probability
        strategy = node.getStrategy(p0 if player == 0 else p1)
        util = [0.0] * self.NUM_ACTIONS
        nodeUtil = 0
        for a in range(self.NUM_ACTIONS):
            nextHistory = history + ("p" if a == 0 else "b")
            if player == 0:
                util[a] = -self.cfr(cards, nextHistory, p0 * strategy[a], p1)
            else:
                util[a] = -self.cfr(cards, nextHistory, p0, p1 * strategy[a])
            nodeUtil += strategy[a] * util[a]
        # For each action, compute and accumulate counterfactual regret
        for a in range(self.NUM_ACTIONS):
            regret = util[a] - nodeUtil
            node.regretSum[a] += (p1 if player == 0 else p0) * regret
        # print(nodeUtil)
        return nodeUtil

    # Train Kuhn poker
    def train(self, iterations: int):  # train Kuhn poker
        cards = [1, 2, 3]
        util = 0.0
        for i in range(iterations):
            shuffleCards(cards)
            util += self.cfr(cards, "", 1, 1)
        print("Average game value: ", util / iterations)
        for n in self.nodeMap.values():
            print(n.toString())

    def one_run(self, cards):
        self.cfr(cards, "", 1, 1)
        print()
        for n in self.nodeMap.values():
            print(n.toString())

    # KuhnTrainer main method
    def main(self):
        iterations = 2  # 000000
        self.train(iterations)


if __name__ == '__main__':
    KuhnTrainer().train(10000)

