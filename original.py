'''Author: hope_is_dead_'''
from math import exp
 
ratings = {
    "Player1": 1400,
    "Player2": 1000,
    "Player3": 1000,
    "Player4": 1000,
    "Player5": 1000,
    "Player6": 1000,
    "Player7": 1000,
    "Player8": 1000,
}
 
possible_scores = [3, 7/3, 5/3, 1, 0] # from first to last place
k = 40
 
probabilities = {}
 
def calcProbabilityDrawPhase(player, *players):
    return exp(ratings.get(player)/173.717792761) / sum(exp(ratings.get(p)/173.717792761) for p in players)
 
def calcProbability(*players):
    if players in probabilities:
        return probabilities.get(players)
    elif len(players) == 1:
        player, = players
        res = calcProbabilityDrawPhase(player, *ratings.keys())
        probabilities[players] = res
        return res
    else:
        player_set = set(ratings.keys())
        prevPlayers = players[:-1]
        for p in prevPlayers:
            player_set.remove(p)
        prevProb = calcProbability(*prevPlayers)
        res = prevProb * calcProbabilityDrawPhase(players[-1], *tuple(player_set))
        probabilities[players] = res
        return res
    
def calcExpectedValue(player):
    res = 0
    player_set = set(ratings.keys())
    player_set.remove(player)
    res += possible_scores[0] * calcProbability(player)
    res += possible_scores[1] * sum(calcProbability(p, player) for p in player_set)
    for p in player_set:
        next_set = set(player_set)
        next_set.remove(p)
        res += possible_scores[2] * sum(calcProbability(p, q, player) for q in next_set)
        for r in next_set:
            next_next_set = set(next_set)
            next_next_set.remove(r)
            res += possible_scores[3] * sum(calcProbability(p, r, q, player) for q in next_next_set)
    return res
 
# the result is expected to be exactly 4 keys from ratings
# result has to look like this: ("Player3", "Player5", "Player1", "Player2")
def updateRatings(*result):
    player_set = set(ratings.keys())
    expectedValues = {}
    for p in ratings.keys():
        expectedValues[p] = calcExpectedValue(p)
    player_scores = {}
    for i, p in enumerate(result):
        player_scores[p] = possible_scores[i]
        player_set.remove(p)
    for p in player_set:
        player_scores[p] = possible_scores[-1]
    for p in ratings:
        ratings[p] = ratings[p] + k * (player_scores[p] - expectedValues[p]) # same as the Elo rating System
    probabilities.clear()