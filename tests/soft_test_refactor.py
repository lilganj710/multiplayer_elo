'''Print the outputs of original.py vs refactored versions'''
import numpy as np

import sys
sys.path.append('./')
import original  # noqa: E402
import preliminary_refactor  # noqa: E402
import pairwise_elo  # noqa: E402


def main():
    ordering = [f'Player{i}' for i in range(1, 4+1)]
    original.updateRatings(*ordering)
    print(f'{original.ratings=}')

    num_players = 8
    old_ratings = np.full(num_players, 1000)
    old_ratings[0] = 1400
    result = np.arange(num_players)
    ratings = preliminary_refactor.update_ratings(result, old_ratings)
    print(f'from preliminary {ratings=}')

    ratings = pairwise_elo.update_ratings(result, old_ratings)
    print(f'from pairwise ELO: {ratings=}')


if __name__ == '__main__':
    main()
