import numpy as np
import sys
sys.path.append(r'E:\projects\pyrl')
from gomoku import game
from gomoku.visualization import base

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


def board_render(game_data, selected_point=None):
    row_size = game_data.setting.row_size
    column_size = game_data.setting.column_size

    lines = ''.rjust(6) + ''.join(map(lambda i: base.column_name(i).ljust(2), range(column_size)))
    lines = '\n' + lines + '\n'
    for r in range(row_size):
        line = base.row_name(r).rjust(6)
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


def render_as_array(game_data):
    pieces = game_data.pieces
    row_size = game_data.setting.row_size
    column_size = game_data.setting.column_size

    pieces_str = np.array([['   '] * column_size] * row_size)
    pieces_str[pieces == game.Piece.black.value] = ' ' + BLACK_PIECE + ' '
    pieces_str[pieces == game.Piece.white.value] = ' ' + WHITE_PIECE + ' '

    print(LTOP_SINGLE + ROW_SINGLE * (3 * column_size) + RTOP_SINGLE)
    for row in range(row_size):
        line = LINE_SINGLE
        for column in range(column_size):
            line += pieces_str[row, column]
        print(line + LINE_SINGLE)
    print(LBOTTOM_SINGLE + ROW_SINGLE * (3 * column_size) + RBOTTOM_SINGLE)


