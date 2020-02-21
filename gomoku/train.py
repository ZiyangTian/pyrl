import sys

from gomoku import game
from gomoku import model
from gomoku import player
from gomoku import config
from gomoku import data


class TrainPipeline(config.Config):
    def __init__(self, **kwargs):
        super(TrainPipeline, self).__init__(**kwargs)
        self._game_setting = game.GameSetting(
            self.row_size, self.column_size, self.piece_in_line,
            self.black_first, self.max_num_round)
        self._policy_value_net = model.PolicyValueNet(
            self._game_setting)
        self._agent = player.DeepMCTSAgent(
            'agent', self._game_setting,
            self._policy_value_net.policy_value_fn, c_puct=self.c_puct)
        self._data_generator = data.DataGenerator(self._agent, self.data_buffer_size)

    def policy_update(self):
        dataset = self._data_generator.get_dataset()
        dataset = dataset.repeat(self.epochs).shuffle(self.buffer_size).batch_size(self.batch_size)
        self._policy_value_net.fit(dataset, epochs=1)
        print('After training one epoch, metrics:', self._policy_value_net.metrics)
        self._policy_value_net.save(self.saved_model_file)
        # TODO: add early stopping callbacks and learning rate adjustment.

    def policy_evaluate(self, num_games=10):
        pass

    def run(self):
        for n in self.max_epochs:
            print("=" * 20 + "\nEpoch {}\n".format(n) + '-' * 20)
            self._data_generator.generate_new_data(self.num_games_per_generation)
            if len(self._data_generator) < self.batch_size:
                continue
            self.policy_update()
