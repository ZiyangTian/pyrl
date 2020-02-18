import collections
import copy
import math
import numpy as np
import tensorflow as tf

from gomoku import game


class State(object):
    """State representation of the game environment.

    Properties
        value: A state value.

    state value: A `float` `np.ndarray` of shape (`row_size`, `column_size`, 4).
        Slice (:, :, 0) is a binary matrix, representing whether a black piece on the position.
        Slice (:, :, 1) is a binary matrix, representing whether a white piece on the position.
        Slice (:, :, 2) is a binary matrix, representing whether the position is empty.
        Slice (:, :, 3) is a single-value matrix, representing whether it's black piece's turn.

    Arguments
        game_data: A `GameData` instance.
    """

    def __init__(self, game_data):
        row_size = game_data.setting.row_size
        column_size = game_data.setting.column_size

        black_pieces = game_data.black_pieces
        white_pieces = game_data.white_pieces
        available_positions = game_data.available_positions
        turn = np.broadcast_to(game_data.turn is game.Piece.black, (row_size, column_size)).astype(np.int)

        self._value = np.stack([black_pieces, white_pieces, available_positions, turn], axis=-1).astype(np.float)

    @property
    def value(self):
        return self._value

    @classmethod
    def initial_value(cls, game_setting):
        """ Create a `State` instance from an initial game (empty board).

        Arguments
            game_setting: A `GameSetting` instance to initiate a new game.

        Returns
            A state value.
        """
        row_size = game_setting.row_size
        column_size = game_setting.column_size
        pieces = np.zeros((row_size, column_size), dtype=np.int)
        available_positions = np.ones((row_size, column_size), dtype=np.int)
        turn = np.broadcast_to(game_setting.black_first, (row_size, column_size)).astype(np.int)

        return np.stack([pieces, pieces, available_positions, turn], axis=-1).astype(np.float)


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
        if self.is_root():
            raise ValueError('Cannot get_value of a root node.')
        u = c_puct * self.prob * math.sqrt(self.parent.visit_times)
        u /= 1. + self.visit_times
        return self._quality_value + u

    def select(self, c_puct):
        """
        :param c_puct:
        Returns
            A `tuple` of (move, value).
            move: A `Move` instance.
            node: An `MCTNode` instance, child node of branch `move`.
        """
        return max(self._children.items(), key=lambda item: item[1].get_value(c_puct))

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
    def __new__(cls, policy_value_fn, c_puct=5.):
        return super(DeepMCTS, cls).__new__(cls, policy_value_fn, c_puct)

    def __init__(self, *args, **kwargs):
        del args
        del kwargs
        super(DeepMCTS, self).__init__()
        self._root = MCTNode()

    def _playout(self, game_data: game.GameData):
        """ use initial game_data """
        node = self._root
        game_data = copy.deepcopy(game_data)

        while not node.is_leaf():
            move, node = node.select(self.c_puct)
            game_data.move(move)

        winner = game_data.winner
        if winner is None:
            turn, available_positions = game_data.turn, game_data.available_positions
            probs, value = self.policy_value_fn(game_data)
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

    def get_move_probs(self, game_data: game.GameData, times=10000, temperature=1e-3):
        """Run all playouts in turn.  """
        for _ in range(times):
            game_data = copy.deepcopy(game_data)
            self._playout(game_data)

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


class PolicyValueNet(tf.keras.Model):
    def __init__(self, game_setting, regularizer_rate=0.001):
        super(PolicyValueNet, self).__init__()
        row_size = game_setting.row_size
        column_size = game_setting.column_size

        def get_regularizer():
            return tf.keras.regularizers.l2(regularizer_rate)

        self._input = tf.keras.layers.InputLayer(input_shape=[column_size, row_size, 4])
        self._conv_1 = tf.keras.layers.Conv2D(
            32, [3, 3],
            padding='same', kernel_regularizer=get_regularizer(), activation='relu', name='conv_1')
        self._conv_2 = tf.keras.layers.Conv2D(
            64, [3, 3],
            padding='same', kernel_regularizer=get_regularizer(), activation='relu', name='conv_2')
        self._conv_3 = tf.keras.layers.Conv2D(
            128, [3, 3],
            padding='same', kernel_regularizer=get_regularizer(), activation='relu', name='conv_3')

        self._action_conv = tf.keras.layers.Conv2D(
            4, [1, 1],
            padding='same', kernel_regularizer=get_regularizer(), activation='relu', name='action_conv')
        self._action_conv_flatten = tf.keras.layers.Flatten(name='action_conv_flatten')
        self._action_dense = tf.keras.layers.Dense(
            row_size * column_size,
            kernel_regularizer=get_regularizer(), activation='softmax', name='action_dense')
        self._action_reshape = tf.keras.layers.Reshape((row_size, column_size))

        self._value_conv = tf.keras.layers.Conv2D(
            2, [1, 1],
            padding='same', kernel_regularizer=get_regularizer(), activation='relu', name='value_conv')
        self._value_flatten = tf.keras.layers.Flatten(name='value_flatten')
        self._value_dense_1 = tf.keras.layers.Dense(
            64,
            kernel_regularizer=get_regularizer(), activation='relu', name='value_dense_1')
        self._value_dense_2 = tf.keras.layers.Dense(
            1,
            kernel_regularizer=get_regularizer(), activation='relu', name='value_dense_2')

    def call(self, inputs):
        outputs = tf.keras.Sequential([
            self._input,
            self._conv_1,
            self._conv_2,
            self._conv_3])(inputs)
        action_probs = tf.keras.Sequential([
            self._action_conv,
            self._action_conv_flatten,
            self._action_dense,
            self._action_reshape])(outputs)
        value = tf.keras.Sequential([
            self._value_conv,
            self._value_flatten,
            self._value_dense_1,
            self._value_dense_2])(outputs)
        return action_probs, value

    def get_policy_value_fn(self):
        def policy_value_fn(state):
            state_value = tf.expand_dims(state.value, axis=0)
            action_probs, value = self.call(state_value)
            return tf.squeeze(action_probs, axis=0).numpy(), tf.squeeze(value).numpy()

        return policy_value_fn
