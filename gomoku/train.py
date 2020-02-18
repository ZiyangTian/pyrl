

from gomoku import game
from gomoku import model
from gomoku import player
from gomoku import policy


class Config(object):
    # game setting
    row_size = 6
    column_size = 6
    piece_in_line = 4
    black_first = True
    max_num_round = 36

    # mcts
    temperature = 1.0
    n_playout = 400  # num of simulations for each move
    c_puct = 5

    # data
    batch_size = 512  # mini-batch size for training
    buffer_size = 10000
    play_batch_size = 1
    epochs = 5  # num of train_steps for each update

    # train
    learning_rate = 1e-3
    lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL

    # data_buffer = deque(maxlen=self.buffer_size)
    # kl_targ = 0.02
    # check_freq = 50
    # game_batch_num = 1500
    # best_win_ratio = 0.0

    def __init__(self, **kwargs):
        for k, v in kwargs:
            self.__setattr__(k, v)


def generate_equivalent_data(data):
    pass


def run():
    config = Config
    game_setting = game.GameSetting()