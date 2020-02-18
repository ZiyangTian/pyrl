import gomoku
import collections


def main():
    player1 = gomoku.RandomPlayer('player1', sleep=0.5)
    player2 = gomoku.RandomPlayer('player2', sleep=0.5)
    game_setting = gomoku.GameSetting(9, 9, 5, 100)
    game = gomoku.Game(game_setting, player1, player2)
    game.play(display=True)


class Node(collections.namedtuple(
    'Node', (
        'state',
        'prob',
        'parent'))):
    def __init__(self, *args, **kwargs):
        del args
        del kwargs
        super(Node, self).__init__()
        self._children = []
        self._visit_times = 0
        self._quality_value = 0.0


if __name__ == '__main__':
    main()
    # n = Node(1, 2, 3)
