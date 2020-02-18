import abc
import random
import time
import numpy as np

from gomoku import game
from gomoku import model
from gomoku import policy as mcts_policy


class Player(object):
    def __init__(self, name):
        self._name = name

    @abc.abstractmethod
    def get_action(self, game_data):
        raise NotImplementedError('Player.get_action')


class RandomPlayer(Player):
    """A naive random moving player, who will move at a random available position."""
    def __init__(self, name, sleep=None):
        super(RandomPlayer, self).__init__(name)
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
    def __init__(self, name, interface):
        """


        """
        super(HumanPlayer, self).__init__(name)
        self._interface = interface

    def get_action(self, game_data):
        return self._interface.get_move()


class Agent(Player):
    def __init__(self, name, policy: mcts_policy.DeepMCTS, piece=game.Piece.none):
        super(Agent, self).__init__(name)
        self._policy = policy

    def get_action(self, game_data: game.GameData,
                   return_state=False, return_probs=False,
                   is_self_play=False, temperature=1e-3):
        """Get action according to game state.
        Arguments
            game_data: A `GameData` instance.
            return_state: A `bool`, whether to return state.
            return_probs: A `bool`, whether to return action selection probabilities.
            is_self_play: A `bool`, whether the agent is playing with itself.
            temperature: A float, temperature variable to control exploitation and exploration.
        Returns
            A single `Move` instance or a tuple.
            move: A `Move` instance.
            state: Optional, A `State` instance.
            probs_array: A `np.ndarray` instance of shape (`row_size`, `column_size`).
        """
        available_positions = game_data.available_positions

        if not available_positions.any():
            raise ValueError('There is no available position in the board.')

        moves, probs = self._policy.get_move_probs(game_data, temperature)

        if is_self_play:
            move = np.random.choice(
                moves,
                p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
            self._policy.move(move)
        else:
            move = np.random.choice(moves, p=probs)
            self._policy.move(None)

        state = model.State(game_data) if return_state else None
        returns = (move,)
        if return_state:
            returns += state
        if return_probs:
            probs_array = np.zeros_like(available_positions)
            for m in moves:
                probs_array[m.row, m.column] = probs[m.row, m.column]
            returns += probs_array
        if len(returns) == 0:
            returns = returns[0]
        return returns

    def self_play(self, game_setting: game.GameSetting):
        states, probs, turns = [], [], []
        data = game.GameData(game_setting)
        data.reset()

        while True:
            (row, column), state, prob = self.get_action(
                data, return_state=True, return_probs=True)
            states.append(state)
            probs.append(prob)
            turns.append(data.turn)

            data.move(row, column)
            winner = data.winner
            if winner is not None:
                return states, probs, turns, winner




