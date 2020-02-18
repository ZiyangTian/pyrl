import collections
import enum
import numpy as np

from gomoku.visualization import base as visualization_base


class Piece(enum.Enum):
    """Gomoku piece type.
    none: No piece or a tie.
    black: Black piece or black piece as the winner.
    white: White piece or white piece as the winner.
    """
    none = 0
    black = 1
    white = 2


Move = collections.namedtuple('Move', ('piece', 'row', 'column'))


class GameSetting(collections.namedtuple(
    'GameSetting', (
        'row_size',
        'column_size',
        'piece_in_line',
        'black_first',
        'max_num_rounds'))):
    """Game setting parameters.

    Arguments
         row_size: An positive `int`, representing the number of rows in the board.
         column_size: An positive `int`, representing the number of rows in the board.
         piece_in_line: An positive `int`, representing the number of pieces in a line to win the game.
         black_first: A `bool`, whether black piece move first.
         max_num_rounds: An positive `int`, representing the maximum number of rounds in one game,
            default to infinity.

    Raises
        ValueError: If any argument is specified unreasonably.
    """

    def __new__(cls, row_size, column_size, piece_in_line,
                black_first=True, max_num_rounds=None):
        for k, v in {
                'row_size': row_size,
                'column_size': column_size,
                'piece_in_line': piece_in_line}.items():
            if v <= 0:
                raise ValueError('Parameter {} should be positive.'.format(k))
        if row_size < piece_in_line or column_size < piece_in_line:
            raise ValueError('Size is too small to generate a board.')
        if max_num_rounds is not None and max_num_rounds < piece_in_line:
            raise ValueError('The allowed maximum number of rounds is too small.')
        return super(GameSetting, cls).__new__(cls, row_size, column_size, piece_in_line, black_first, max_num_rounds)


class GameData(object):

    def __init__(self, game_setting: GameSetting):
        self._setting = game_setting
        self._num_round = 1
        self._turn = Piece.black if game_setting.black_first else Piece.white
        self._pieces = np.array(np.broadcast_to(
            Piece.none.value,
            (self._setting.row_size, self._setting.column_size)))
        self._history = []
        self._winner = None

    @property
    def setting(self):
        return self._setting

    @property
    def num_round(self):
        return self._num_round

    @property
    def turn(self):
        return self._turn

    @property
    def pieces(self):
        return self._pieces

    def history(self, i):
        """Get the i-th move of game history.
        Arguments
            i: An `int`, index of history.
        Returns
            A `Step` instance.
        """
        return self._history[i]

    @property
    def black_pieces(self):
        return (self._pieces == Piece.black.value).astype(np.int)

    @property
    def white_pieces(self):
        return (self._pieces == Piece.white.value).astype(np.int)

    @property
    def available_positions(self):
        return (self._pieces == Piece.none.value).astype(np.int)

    @property
    def winner(self):
        """
        Returns
            None -> no winner, the game is not over.
            `Piece` enumeration -> the game is over with the corresponding winner or a tie.
        """
        return self._winner

    def reset(self):
        """Reset the game."""
        self._num_round = 1
        self._turn = Piece.black
        self._pieces = np.array(np.broadcast_to(
            Piece.none.value,
            (self._setting.row_size, self._setting.column_size)))
        self._history = []
        self._winner = None

    def move(self, row=None, column=None):
        """Move one piece (in turn) with its position.
        Arguments
            row: An `int`, row number of the position to move.
            column:  An `int`, column number of the position to move.
        Returns:
            The winner checking result.
        """
        if self._winner is not None:
            raise AssertionError('The game has already ended with winner {}.'.format(self.winner))

        if row is None and column is None:
            self._history.append(None)
        elif self.available_positions[row, column] == 1:
            self._pieces[row, column] = self._turn.value
            self._history.append(Move(self._turn, row, column))
        else:
            raise ValueError('Cannot place a piece at position ({x}, {y}).'.format(x=row, y=column))

        if self._turn is Piece.black:
            self._turn = Piece.white
        else:
            self._turn = Piece.black
            self._num_round += 1
        self._check_winner()
        return self

    def _check_winner(self):
        if self._winner is not None:
            return self._winner
        if self._num_round < self._setting.piece_in_line:
            return None

        round_cond = self.setting.max_num_rounds is not None and self._num_round > self.setting.max_num_rounds
        last_step = self._history[-1]
        piece = last_step.piece

        if piece is Piece.none:
            if round_cond:
                self._winner = Piece.none
            return self._winner

        if piece is Piece.black:
            pieces = self.black_pieces
        else:
            pieces = self.white_pieces
        row = last_step.row
        column = last_step.column

        def n_in_line(array1d):
            for i in range(len(array1d) - self._setting.piece_in_line + 1):
                if array1d[i: i + self._setting.piece_in_line].all():
                    return True
            return False

        if n_in_line(pieces[row]) \
                or n_in_line(pieces[:, column]) \
                or n_in_line(np.diag(pieces, k=column - row)) \
                or n_in_line(np.diag(np.fliplr(pieces), k=self._setting.column_size - 1 - column - row)):
            self._winner = piece
            return piece

        if round_cond or self.available_positions.max() == 0:
            self._winner = Piece.none
        return self._winner

    def __repr__(self):
        pieces = self.pieces
        row_size = self.setting.row_size
        column_size = self.setting.column_size

        pieces_str = np.array([['   '] * column_size] * row_size)
        pieces_str[pieces == Piece.black.value] = ' ' + "●" + ' '
        pieces_str[pieces == Piece.white.value] = ' ' + "○" + ' '

        repr_str = 'GameData(row_size={}, column_size={}, piece_in_line={}, {}, max_num_rounds={})\n'.format(
            self.setting.row_size,
            self.setting.column_size,
            self.setting.piece_in_line,
            'black piece first' if self.setting.black_first else 'white piece first',
            self.setting.max_num_rounds)
        repr_str += 'round:{}, turn:{}.\n'.format(self._num_round, self._turn)

        repr_str += "┌" + "─" * (3 * column_size) + "┐\n"
        for row in range(row_size):
            line = "│"
            for column in range(column_size):
                line += pieces_str[row, column]
            repr_str += line + "│\n"
        repr_str += "└" + "─" * (3 * column_size) + "┘\n"

        return repr_str


class Game(GameData):
    """Gomoku game environment. The game starts with two player and ends with one winner or a tie.

    Properties
        setting: A `GameSetting` instance.
        num_round: An `int`, the current number of game rounds.
        turn: `Piece.black` or `Piece.white`, the game turn.
        pieces: A 2-D `int` numpy array of shape (`row_size`, `column_size`), each element of which
            represents the value of piece (see class `Piece`).
        black_pieces: A binary 2-D `int` numpy array of shape (`row_size`, `column_size`), represents
            the positions of black pieces.
        white_pieces: A binary 2-D `int` numpy array of shape (`row_size`, `column_size`), represents
            the positions of white pieces.
        available_positions: A binary 2-D `int` numpy array of shape (`row_size`, `column_size`),
            represents the positions of available positions.
        winner: None or the winning player.

    Arguments
        game_setting: A `GameSetting` instance, the setting used in the game.
        black_piece_player: A `Player` instance, the player to play black pieces.
        white_piece_player: A `Player` instance, the player to play white pieces.
        visualization: A `Visualization` instance, the display mode.
    """
    def __init__(self,
                 game_setting,
                 black_piece_player, white_piece_player,
                 visualization=None):

        """ display_mode: None, terminal, graphic"""
        super(Game, self).__init__(game_setting)
        self._black_piece_player = black_piece_player
        self._white_piece_player = white_piece_player
        if type(visualization) is str:
            self._visualization = ''
        elif type(visualization) is visualization_base.Visualization:
            self._visualization = visualization
        else:
            self._visualization = None

    def play(self, display=False):
        """Play one game until ending.
        Arguments
            display:
        Returns
            The winner.
        """
        self.reset()
        while True:
            if self._turn is Piece.black:
                row, column = self._black_piece_player.get_action(super(Game, self))
            else:
                row, column = self._white_piece_player.get_action(super(Game, self))
            self.move(row, column)

            # TODO: ...
            if display:
                print(self)
                print('row:{}, column:{}'.format(row, column))
                print()

            if self.winner is not None:
                if display:
                    print('winner: {}'.format(self._winner))
                return self.winner
