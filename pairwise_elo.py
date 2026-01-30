'''Here, an ordering of n players is treated as (n-1)n/2 pairwise
one-on-one matches, with the vanilla ELO update used for each'''
import numpy as np
import numpy.typing as npt


K = 40
'''The K-factor for the rating update'''
SCALE_FACTOR = 400
'''This rating increase represents a tenfold increase in skill level'''


def sigmoid(r: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    '''The base-10 sigmoid function used in vanilla ELO, applied
    elementwise to r'''
    return 1 / (1 + 10**(-r/SCALE_FACTOR))


def get_winner_matrix(result: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    '''winner_matrix[i, j] = 1 if player i won against player j, 0 otherwise'''
    inverse_permutation = np.argsort(result)
    return (inverse_permutation > inverse_permutation[:, None]).astype(int)


def update_ratings(
        result: npt.NDArray[np.int64], old_ratings: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
    '''Given result representing a finish ordering of 0-indexed players
    (permutation of {0, ..., n-1}), update the old_ratings
    Return the new ratings'''
    rating_diffs = old_ratings[:, None] - old_ratings
    expected_scores = sigmoid(rating_diffs)
    winner_matrix = get_winner_matrix(result)
    update_terms = winner_matrix - expected_scores
    np.fill_diagonal(update_terms, 0)
    average_updates = (
        len(result)/(len(result)-1)
        * np.mean(update_terms, axis=-1)
    )
    new_ratings = old_ratings + K * average_updates
    return new_ratings
