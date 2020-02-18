import collections
import copy
import sys
import math
import random
import numpy as np

from gomoku import game
from gomoku import model


class MCTNode(object):
    def __init__(self, parent=None, prob=1.):
        self.parent = parent
        self.prob = prob
        self._children = {}  # A dict from Move to Node
        self._visit_times = 0
        self._quality_value = 0.0

    @property
    def children(self):
        return self._children

    def expand(self, move, prob):
        if move in self._children:
            raise ValueError('Move {} is already a child.'.format(str(move)))
        self._children[move] = MCTNode(parent=self, prob=prob)

    @property
    def visit_times(self):
        return self._visit_times

    def get_value(self, c_puct):
        if not self.is_root():
            u = c_puct * self.prob * math.sqrt(self.parent.visit_times)
            u /= 1. + self.visit_times
        else:
            u = 0.
        return self._quality_value + u

    def select(self, c_puct):
        """
        :param c_puct:
        Returns
            A `tuple` of (move, value).
            move: A `Move` instance.
            node: An `MCTNode` instance, child node of branch `move`.
        """
        return max(self._children.items(), key=lambda k, v: v.get_value(c_puct))

    def backup(self, new_value):
        if not self.is_root():
            self.parent.backup(-new_value)  # for piece type is different.
        self._visit_times += 1
        self._quality_value += (new_value - self._quality_value) / self._visit_times

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return self._children == {}


class DeepMCTS(collections.namedtuple(
    'DeepMCTS', (
        'policy_value_fn',
        'c_puct'))):
    """DNN policy_value_fn based Mente Carlo tree searching.

    Arguments
        policy_value_fn: A function which takes a state value as input and output
            a tuple of (`action_probs`, `value`), where `action_probs` as a
            `np.ndarray` of shape (`row_size`, `colums_size`), representing the
            probabilities of each move, and `value` is a `float`, representing the
            value of the state.
        c_puct: A `float`, parameter to control exploration and exploitation.
    """

    def __new__(cls, policy_value_fn, c_puct=5.):
        return super(DeepMCTS, cls).__new__(cls, policy_value_fn, c_puct)

    def __init__(self, *args, **kwargs):
        del args
        del kwargs
        super(DeepMCTS, self).__init__()
        self._root = MCTNode()

    def _playout(self, game_data: game.GameData, policy_value_fn):
        """ use initial game_data """
        node = self._root
        game_data = copy.deepcopy(game_data)

        while not node.is_leaf():
            move, node = node.select(self.c_puct)
            game_data.move(move)

        winner = game_data.winner
        if winner is None:
            turn, available_positions = game_data.turn, game_data.available_positions
            probs, value = policy_value_fn(game_data)
            for r in range(game_data.setting.row_size):
                for c in range(game_data.setting.column_size):
                    if available_positions.astype(np.bool)[r][c]:
                        node.expend(game.Move(turn, r, c), probs[r][c])
        elif winner is game.Piece.none:
            value = 0.
        else:
            value = float(winner is game_data.turn)
        node.update_backup(-value)  # ???

    @staticmethod
    def _stable_softmax(x):
        x = np.exp(x - np.max(x))
        x /= np.sum(x)
        return x

    def get_move_probs(self, game_data: game.GameData, policy_value_fn, times=10000, temperature=1e-3):
        """Run all playouts in turn.  """
        for _ in range(times):
            game_data = copy.deepcopy(game_data)
            self._playout(game_data, policy_value_fn)

        moves, visit_times = zip([(move, node.visit_times) for move, node in self._root.children.items()])
        probs = self._stable_softmax(np.log(np.array(visit_times) / temperature + 1.e-10))
        return moves, list(probs)

    def move(self, move: game.Move = None):
        if move in self._root.children:
            self._root = self._root.children[move]
            del self._root.parent  # ...
            self._root.parent = None
            # self._root.prob = 1.
        else:
            self._root = MCTNode()
