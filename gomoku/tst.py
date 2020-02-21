import collections
import sys
# import gomoku
# from gomoku import *


def main():
    player1 = RandomPlayer('player1', sleep=0.5)
    player2 = RandomPlayer('player2', sleep=0.5)
    game_setting = GameSetting(9, 9, 5, 100)
    game = Game(game_setting, player1, player2)
    game.play(display=True)


def main1():
    game_setting = GameSetting(6, 6, 4)
    game_data = GameData(game_setting)
    state = State(game_data)
    mcts = DeepMCTS()
    result = mcts.get_move_probs(game_data, gomoku.model.random_policy_value_fn, times=100)

    node_0 = MCTNode()
    # node_1 = MCTNode(node_0, 0.5)
    # node_2 = MCTNode(node_0, 0.5)
    node_0.expand(Move(Piece.black, 5, 4), 0.5)
    node_0.expand(Move(Piece.black, 4, 4), 0.5)
    node_1, node_2 = tuple(node_0.children.values())
    node_1.backup(1.)
    node_2.backup(2.)


def main2():
    game_setting = GameSetting(6, 6, 4)
    policy_value_fn = PolicyValueNet(game_setting).policy_value_fn
    agent = DeepMCTSAgent('agent', game_setting, policy_value_fn)
    # states, probs, turns, winner, data = agent.self_play(game_setting)
    data_generator = DataGenerator(agent, 100)
    data_generator.generate_new_data(1)


if __name__ == '__main__':
    # main2()
    # n = Node(1, 2, 3)
    print(sys.argv)
