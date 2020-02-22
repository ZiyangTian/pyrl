import abc
import random
import time
import numpy as np

from gomoku import game
from gomoku import model


class Player(object):
    def __init__(self, name, game_setting: game.GameSetting):
        self._name = name
        self._game_setting = game_setting

    @property
    def name(self):
        return self._name

    @abc.abstractmethod
    def get_action(self, game_data):
        raise NotImplementedError('Player.get_action')


class RandomPlayer(Player):
    """A naive random moving player, who will move at a random available position."""
    def __init__(self, name, game_setting, sleep=None):
        super(RandomPlayer, self).__init__(name, game_setting)
        self._sleep = sleep

    def get_action(self, game_data):
        if game_data.winner is not None:
            raise ValueError('The game has already ended with winner {}.'.format(str(game_data.winner)))
        if self._sleep:
            time.sleep(self._sleep)
        available_positions = []
        for row, pieces_row in enumerate(game_data.available_positions):
            for column, piece in enumerate(pieces_row):
                if piece != game.Piece.none.value:
                    available_positions.append((row, column))
        return random.choice(available_positions)


class HumanPlayer(Player):
    """A human player, whose move is gotten from human-machine interaction.
    Arguments
        name: A `str`, player name.
        piece: An enumeration among `Piece.black` or `Piece.white`,  piece type of the player.
        interface: An `Interface` instance.
    """
    def __init__(self, name, game_setting, interface):
        super(HumanPlayer, self).__init__(name, game_setting)
        self._interface = interface

    def get_action(self, game_data):
        return self._interface.get_move()


class DeepMCTSAgent(Player):
    def __init__(self, name, game_setting, policy_value_fn, **kwargs):
        super(DeepMCTSAgent, self).__init__(name, game_setting)
        self._policy_value_fn = policy_value_fn
        self._mcts_config = kwargs
        self._policy = None
        self.reset()

    def reset(self):
        self._policy = model.DeepMCTS(**self._mcts_config)

    def get_action(self, game_data: game.GameData,
                   return_state=False, return_probs=False,
                   is_self_play=False, playout_times=10000, temperature=1e-3):
        """Get action according to game state.
        Arguments
            game_data: A `GameData` instance.
            return_state: A `bool`, whether to return state.
            return_probs: A `bool`, whether to return action selection probabilities.
            is_self_play: A `bool`, whether the agent is playing with itself.
            temperature: A float, temperature variable to control exploitation and exploration.
        Returns
            A single `Move` instance or a `tuple`.
            move: A `Move` instance.
            state: Optional, A `State` instance.
            probs_array: A `np.ndarray` instance of shape (`row_size`, `column_size`).
        """
        available_positions = game_data.available_positions

        if not available_positions.any():
            raise ValueError('There is no available position in the board.')

        moves, probs = self._policy.get_move_probs(
            game_data, self._policy_value_fn, times=playout_times, temperature=temperature)
        probs = np.array(probs)

        if is_self_play:
            move_i = np.random.choice(
                range(len(moves)),  # `Move` object has `__len__` method, thus can be transformed as a `np.ndarray`.
                p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
            move = moves[move_i]
            self._policy.move(move)
        else:
            move = np.random.choice(moves, p=probs)
            self._policy.move(None)

        state = model.State(game_data) if return_state else None
        returns = (move,)
        if return_state:
            returns += (state,)
        if return_probs:
            probs_array = np.zeros_like(available_positions, dtype=np.float)
            for m, p in zip(moves, probs):
                probs_array[m.row, m.column] = p
            returns += (probs_array,)
        if len(returns) == 0:
            returns = returns[0]
        return returns

    def self_play(self, playout_times=10000, temperature=1.):
        """Complete one self-play and return the history and the final data.
        Returns
            A `tuple` of (`states`, `probs`, `turns`, `winner`, `game_data`):
                states: A `list` of `State` instance.
                probs: A `list` of `np.ndarray` of shape (`row_size`, `column_size`).
                turns: A `list` containint `Piece.black` and `Piece.white`.
                winner: An enumeration of `Piece` object.
                game_data: A `GameData` instance.
        """
        states, probs, turns = [], [], []
        game_data = game.GameData(self._game_setting)
        game_data.reset()

        while True:
            self.reset()
            move, state, prob = self.get_action(
                game_data,
                return_state=True, return_probs=True,
                is_self_play=True, playout_times=playout_times, temperature=temperature)
            states.append(state)
            probs.append(prob)
            turns.append(game_data.turn)

            game_data.move(move.row, move.column)
            winner = game_data.winner
            if winner is not None:
                return states, probs, turns, winner, game_data
