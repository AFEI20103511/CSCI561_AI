import read
import write
from host import GO
from read import readInput
from my_player3 import get_steps

if __name__ == "__main__":
    piece_type, previous_board, board = readInput(5)
    go = GO(5)
    go.set_board(piece_type, previous_board, board)
    # go.visualize_board()
    step = get_steps(previous_board)
    print(step)
    # print(go.get_score(1))
    # print(go.get_score(2))

    # for i in range(5):
    #     step = read.read_helper()
    #     print(step)
    #     write.write_helper(step + 1)

    # print(go.detect_all_neighbours(0,1))
    # print(go.get_possible_placements())
    # print(go.loc_evaluation(3,1))
    # print(go.piece_type)
    #
    # print(go.get_possible_placements())
    #
    # print(go.heuristic())
    #
    # n_go = GO.new_Go(go, 2, 3, piece_type)
    # n_go.visualize_board()
    # print(n_go.piece_type)
    # print(n_go.heuristic())
    # print(n_go.get_possible_placements())

    # steps = read.readHelper()
    # print(f'current steps is {steps}')
    # write.writeHelper(steps + 1)
    # steps = read.readHelper()
    # print(f'current steps is {steps}')