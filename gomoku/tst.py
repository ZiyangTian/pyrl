import collections

from gomoku import *


def main():
    player1 = RandomPlayer('player1', sleep=0.5)
    player2 = RandomPlayer('player2', sleep=0.5)
    game_setting = GameSetting(9, 9, 5, 100)
    game = Game(game_setting, player1, player2)
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


def main1():
    node_0 = MCTNode()
    # node_1 = MCTNode(node_0, 0.5)
    # node_2 = MCTNode(node_0, 0.5)
    node_0.expand(Move(Piece.black, 5, 4), 0.5)
    node_0.expand(Move(Piece.black, 4, 4), 0.5)
    node_1, node_2 = tuple(node_0.children.values())
    node_1.backup(1.)
    node_2.backup(2.)


if __name__ == '__main__':
    main()
    # n = Node(1, 2, 3)
