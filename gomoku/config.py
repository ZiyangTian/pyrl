

class Config(object):
    # game setting
    row_size = 6
    column_size = 6
    piece_in_line = 4
    black_first = True
    max_num_round = 36

    # mcts
    temperature = 1.0
    playout_times = 100  # num of simulations for each move
    c_puct = 5.

    # data
    num_games_per_generation = 10

    batch_size = 32  # mini-batch size for training
    buffer_size = 10000
    data_buffer_size = 10000

    # train
    epoch_per_dataset = 5  # num of train_steps for each update
    max_epochs = 1000
    # learning_rate = 1e-3
    # lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL

    # saved model
    saved_dir = 'saved'

    def __init__(self, **kwargs):
        for k, v in kwargs:
            self.__setattr__(k, v)
