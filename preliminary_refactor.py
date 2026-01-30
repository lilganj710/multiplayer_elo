'''Multiplayer ELO based on an urn model. A preliminary starting point,
but undesirable due to O(n!) complexity and the introduction of extra
"score" hyperparameters'''
import itertools as it
import numpy as np
import numpy.typing as npt


POSSIBLE_SCORES = np.array([3, 7/3, 5/3, 1] + [0]*4)
'''Note that the last 4 scores don't have to be 0 anymore. Again though,
all these extra hyperparams are undesirable'''
K = 40
'''The K-factor for the rating update'''
SCALE_FACTOR = 173.717792761
'''I'm not quite sure where you got this from (maybe this is 400/ln(10)?)
But I'll use it anyway here'''


def get_ordering_probabilities(
        ordered_ratings: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    '''Given ratings in the order that each player finished, use softmaxes
    to get the probability of each draw'''
    softmaxes = np.exp(ordered_ratings / SCALE_FACTOR)
    softmax_suffixes = np.flip(
        np.cumsum(np.flip(softmaxes, axis=-1), axis=-1),
        axis=-1
    )
    incremental_probabilities = softmaxes / softmax_suffixes
    overall_probabilities = np.prod(incremental_probabilities, axis=-1)
    return overall_probabilities


def get_player_expected_values(
        old_ratings: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    '''Given 0-indexed old_ratings for each player, use them to assign
    probabilities to each possible finish ordering. Recover expected
    scores for each player from this'''
    all_orderings = np.array([
        ordering for ordering in it.permutations(range(len(old_ratings)))
    ])
    ordered_ratings = old_ratings[all_orderings]
    ordering_probabilities = get_ordering_probabilities(ordered_ratings)
    inverse_permutations = np.argsort(all_orderings, axis=-1)
    all_possible_scores = np.zeros_like(all_orderings, dtype=np.float64)
    all_possible_scores = POSSIBLE_SCORES[inverse_permutations]
    expected_values = (
        all_possible_scores * ordering_probabilities[:, None]).sum(axis=0)
    return expected_values


def update_ratings(
        result: npt.NDArray[np.int64], old_ratings: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
    '''Given result representing a finish ordering of 0-indexed players
    (permutation of {0, ..., n-1}), update the old_ratings
    Return the new ratings'''
    expected_values = get_player_expected_values(old_ratings)
    player_scores = POSSIBLE_SCORES[result]
    new_ratings = old_ratings + K * (player_scores - expected_values)
    return new_ratings
