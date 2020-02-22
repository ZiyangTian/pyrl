import numpy as np

from gomoku import game
from gomoku import players

BLACK_PIECE = "●"
WHITE_PIECE = "○"
LTOP_SINGLE = "┌"
TOP_SINGLE = "┬"
RTOP_SINGLE = "┐"
LEFT_SINGLE = "├"
CENTRAL_SINGLE = "┼"
RIGHT_SINGL = "┤"
LBOTTOM_SINGLE = "└"
BOTTOM_SINGLE = "┴"
RBOTTOM_SINGLE = "┘"
ROW_SINGLE = "─"
LINE_SINGLE = "│"
SELECTED = "╬"
UNSELECTABLE = "╳"


def row_name(i):
    return str(i + 1)


def column_name(i):
    m_str = chr(65 + i % 26)
    n = i // 26
    n_str = '' if n == 0 else str(n)
    return m_str + n_str


def _image_board_render(game_data, selected_point=None):
    row_size = game_data.setting.row_size
    column_size = game_data.setting.column_size

    lines = ''.rjust(6) + ''.join(map(lambda i: column_name(i).ljust(2), range(column_size)))
    lines = '\n' + lines + '\n'
    for r in range(row_size):
        line = row_name(r).rjust(6)
        for c in range(column_size):
            if game_data.pieces[r, c] == game.Piece.black.value:
                line += BLACK_PIECE
            elif game_data.pieces[r, c] == game.Piece.white.value:
                line += WHITE_PIECE
            elif (r, c) == selected_point:
                line += SELECTED
            elif r == c == 0:
                line += LTOP_SINGLE + ROW_SINGLE
            elif r == row_size - 1 and c == 0:
                line += LBOTTOM_SINGLE + ROW_SINGLE
            elif r == 0 and c == column_size - 1:
                line += RTOP_SINGLE
            elif r == row_size - 1 and c == column_size - 1:
                line += RBOTTOM_SINGLE
            elif r == 0:
                line += TOP_SINGLE + ROW_SINGLE
            elif c == 0:
                line += LEFT_SINGLE + ROW_SINGLE
            elif r == row_size - 1:
                line += BOTTOM_SINGLE + ROW_SINGLE
            elif c == column_size - 1:
                line += RIGHT_SINGL
            else:
                line += CENTRAL_SINGLE + ROW_SINGLE
        line += '\n'
        lines += line
    return lines


def _array_board_render(game_data):
    pieces = game_data.pieces
    row_size = game_data.setting.row_size
    column_size = game_data.setting.column_size

    pieces_str = np.array([['   '] * column_size] * row_size)
    pieces_str[pieces == game.Piece.black.value] = ' ' + BLACK_PIECE + ' '
    pieces_str[pieces == game.Piece.white.value] = ' ' + WHITE_PIECE + ' '

    lines = LTOP_SINGLE + ROW_SINGLE * (3 * column_size) + RTOP_SINGLE + '\n'
    for row in range(row_size):
        line = LINE_SINGLE
        for column in range(column_size):
            line += pieces_str[row, column]
        lines += line + LINE_SINGLE + '\n'
    lines += LBOTTOM_SINGLE + ROW_SINGLE * (3 * column_size) + RBOTTOM_SINGLE + '\n'


def board_render(game_data, style='array', selected_point=None):
    if style == 'array':
        return _array_board_render(game_data)
    elif style == 'image':
        return _image_board_render(game_data, selected_point=selected_point)
    else:
        raise ValueError('Unexpected style {}.'.format(style))


class Game(game.GameData):
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
    def __init__(self, game_setting, black_piece_player, white_piece_player):
        """ display_mode: None, terminal, graphic"""
        super(Game, self).__init__(game_setting)
        self._black_piece_player = black_piece_player
        self._white_piece_player = white_piece_player

    def play(self, display=False):
        """Play one game until ending.
        Arguments
            display:
        Returns
            The winner.
        """
        self.reset()
        while True:
            if self._turn is game.Piece.black:
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

