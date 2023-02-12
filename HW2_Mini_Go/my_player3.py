import random
import time

from read import readInput
from write import writeOutput

from host import GO
from MyGO import GO as myGo


class myPlayer():
    def __init__(self, go, piece_type, steps, start):
        self.type = 'random'
        self.go = go
        self.piece_type = piece_type
        self.steps = steps
        self.start = start
        self.int_max = 2147483647
        self.int_min = -2147483648
        self.size = 4
        self.max_depth = 1

    # return best action
    def action(self):
        # decide depth based on steps

        # try to occupy [2,2] if available
        if (self.steps == 1 or self.steps == 2) and self.go.board[2][2] == 0:
            # print(f'My move is {(2,2)}')
            return (2, 2)

        # get possible placements
        possible_placements = go.get_possible_placements()

        # if no possible_placements, return "PASS"
        if not possible_placements:
            return "PASS"
        # if there are possible placements, do minimax to decide best move
        move = self.minimax(possible_placements)
        # print(f'My move is {move}')
        return move

    # return minimax solution
    def minimax(self, possible_placements):

        best_next_move = random.choice(possible_placements)
        best_score = self.int_min
        # placement reward
        reward = []
        reward.append((-100, 0, 5, 0, -100))
        reward.append((0, 5, 10, 5, 0))
        reward.append((5, 10, 100, 10, 5))
        reward.append((0, 5, 10, 5, 0))
        reward.append((-100, 0, 5, 0, -100))

        # for each possible placements, check their score
        # if self.steps >= 13 and self.steps <= 16:
        # time_limit = 8
        # else:
        # time_limit = 9.5
        for move in possible_placements:
            # time_used = time.time() - start
            # print(f'time has used {time_used} s')
            if (time.time() - start > 9.5):
                # print('Minimax terminated due to one round time limit')
                break
            n_go = myGo.new_Go(go, move[0], move[1], self.piece_type)
            n_score = self.min_val(n_go, 1, self.steps + 1, 3 - self.piece_type, self.int_min, self.int_max)
            enemy_elimination_factor = go.detect_num_nei_enemies(move[0], move[1], piece_type)
            score = n_score + n_go.loc_evaluation(move[0], move[1]) + reward[move[0]][
                move[1]] * 0.1 + enemy_elimination_factor
            if score > best_score:
                best_score = score
                best_next_move = move

        return best_next_move
        # return random.choice(possible_placements)

    def min_val(self, go, depth, step, piece_type, alpha, beta):
        # when depth is over max_depth or step over max_step, return current heuristic val
        if depth > self.max_depth or step > go.max_move:
            return go.heuristic()
        # get all possible placements, if there is none, return current heuristic val
        possible_placements = go.get_possible_placements()
        if not possible_placements:
            return go.heuristic()
        # check all moves
        best_value = self.int_max
        for move in possible_placements:
            n_go = myGo.new_Go(go, move[0], move[1], piece_type)
            n_score = self.max_val(n_go, depth + 1, step + 1, 3 - piece_type, alpha, beta)
            if n_score < best_value:
                best_value = n_score
            if best_value <= alpha:
                return best_value
            if best_value < beta:
                beta = best_value
        return best_value

    def max_val(self, go, depth, step, piece_type, alpha, beta):
        # when depth is over max_depth or step over max_step, return current heuristic val
        if depth > self.max_depth or step > go.max_move:
            return go.heuristic()
        # get all possible placements, if there is none, return current heuristic val
        possible_placements = go.get_possible_placements()
        if not possible_placements:
            return go.heuristic()
        # check all moves
        best_value = self.int_min
        for move in possible_placements:
            n_go = myGo.new_Go(go, move[0], move[1], piece_type)
            n_score = self.min_val(n_go, depth + 1, step + 1, 3 - piece_type, alpha, beta)
            if n_score > best_value:
                best_value = n_score
            if best_value >= beta:
                return best_value
            if best_value > alpha:
                alpha = best_value
        return best_value


# get current steps
def get_steps(board):
    #     check if it is empty
    num_stones = 0
    for i in range(5):
        for j in range(5):
            if board[i][j] != 0:
                num_stones += 1
    path = "steps.txt"
    if num_stones == 0:
        re = 1
        with open(path, 'w') as f:
            f.write(str(re + 2))
        return re
    elif num_stones == 1:
        re = 2
        with open(path, 'w') as f:
            f.write(str(re + 2))
        return re
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            re = int(lines[0])
        with open(path, 'w') as g:
            g.write(str(re + 2))
        return re


if __name__ == "__main__":
    # get start timestamp
    start = time.time()

    # read input and create go object
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = myGo(N)
    go.set_board(piece_type, previous_board, board)

    # take go input and return action
    steps = get_steps(board)
    # print(f'current step is {steps}')

    # steps = read.readHelper()

    player = myPlayer(go, piece_type, steps, start)
    action = player.action()

    # write output action and update steps
    writeOutput(action)

    # print time info
    end = time.time()
    # print(f'This step total time used is: {end - start}')