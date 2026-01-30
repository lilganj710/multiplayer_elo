'''Test a rating update function by ensuring that if a bunch of
players of equal strengths play each other, there's no rating
drift in the long run'''
from typing import Callable
import itertools as it
import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


import sys
sys.path.append('./')
import preliminary_refactor  # noqa: E402
import pairwise_elo  # noqa: E402


RATING_UPDATE_SIGNATURE = Callable[
    [npt.NDArray[np.int64], npt.NDArray[np.float64]],
    npt.NDArray[np.float64]
]
'''Type signature of a rating update function:
(result, old ratings) -> new ratings'''


def simulate_ratings_over_time(
        rating_update_func: RATING_UPDATE_SIGNATURE,
        relative_strengths: npt.NDArray[np.float64],
        initial_ratings: npt.NDArray[np.float64],
        num_updates: int) -> npt.NDArray[np.float64]:
    '''Return an array of .shape = (num_updates+1, num players) containing
    ratings for each player over time'''
    num_players = len(relative_strengths)
    ratings_list = [initial_ratings]
    for _ in range(num_updates):
        cur_outcome = rng.choice(
            num_players, size=num_players, replace=False, p=relative_strengths)
        new_ratings = rating_update_func(
            cur_outcome, ratings_list[-1])
        ratings_list.append(new_ratings)
    return np.array(ratings_list)


def get_fig_dimensions(num_players: int) -> tuple[int, int]:
    '''Auxiliary function to get a nice-looking axes layout in the figure'''
    num_rows = num_players
    for num_rows in range(int(np.sqrt(num_players)), 0, -1):
        if num_players % num_rows == 0:
            break
    num_cols = num_players // num_rows
    return num_rows, num_cols


def test_rating_update_function(
        rating_update_func: RATING_UPDATE_SIGNATURE,
        relative_strengths: npt.NDArray[np.float64],
        initial_ratings: npt.NDArray[np.float64],
        num_updates: int = 200):
    '''Given relative_strengths of each player (can be interpreted as
    probabilities that each player will get first place), use the
    rating_update_func repeatedly. Plot the rating trajectories
    for each player on separate axes'''
    ratings_over_time = simulate_ratings_over_time(
        rating_update_func, relative_strengths, initial_ratings, num_updates)
    num_rows, num_cols = get_fig_dimensions(len(relative_strengths))
    fig, axes = plt.subplots(num_rows, num_cols)
    axes: list[list[Axes]]
    for player_idx, ax in enumerate(it.chain.from_iterable(axes)):
        ax.plot(ratings_over_time[:, player_idx])
        ax.set_title(f'{player_idx=}')
    fig.suptitle(f'{rating_update_func.__module__} ratings over time')
    plt.show()


def main():
    rating_funcs = [
        preliminary_refactor.update_ratings,
        pairwise_elo.update_ratings
    ]
    relative_strengths = np.ones(num_players := 8)
    relative_strengths[0] = 3
    relative_strengths /= sum(relative_strengths)
    initial_ratings = np.full(num_players, 1000)
    test_rating_update_function(
        rating_funcs[1],
        relative_strengths, initial_ratings
    )


if __name__ == '__main__':
    rng = np.random.default_rng()
    main()
