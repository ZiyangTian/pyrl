import collections
import numpy as np
import tensorflow as tf

from gomoku import game
from gomoku import model
from gomoku import player
from gomoku import policy


class DataGenerator(object):
    def __init__(self, agent, volume):
        self._agent = agent
        self._data_buffer = collections.deque(maxlen=volume)

    def __len__(self):
        return len(self._data_buffer)

    @staticmethod
    def transform_self_play_data(states, probs, turns, winner):
        return list((state.value, prob, winner is turn) for (state, prob, turn) in zip(states, probs, turns))

    @staticmethod
    def get_equivalent_data(state_value, prob, win):
        state_value_flip = np.fliplr(state_value)
        prob_flip = np.fliplr(prob)
        return [
            (state_value, prob, win),
            (np.rot90(state_value), np.rot90(prob), win),
            (np.rot90(state_value, k=2), np.rot90(prob, k=2), win),
            (np.rot90(state_value, k=3), np.rot90(prob, k=3), win),
            (state_value_flip, prob_flip, win),
            (np.rot90(state_value_flip), np.rot90(prob_flip), win),
            (np.rot90(state_value_flip, k=2), np.rot90(prob_flip, k=2), win),
            (np.rot90(state_value_flip, k=3), np.rot90(prob_flip, k=3), win)]

    def generate_new_data(self, num_games=1):
        for _ in range(num_games):  # TODO: add parallelism.
            states, probs, turns, winner, _ = self._agent.self_play()
            source_data = self.transform_self_play_data(states, probs, turns, winner)
            for state_value, prob, win in source_data:
                self._data_buffer.extend(self.get_equivalent_data(state_value, prob, win))

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(list(self._data_buffer))
        dataset = dataset.map(lambda state_value, prob, win: (state_value, (prob, win)))
        return dataset
