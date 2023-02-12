import random
import sys
import time

import read
import write
from read import readInput
from write import writeOutput

from host import GO

class myPlayer():
    def __init__(self, go, piece_type, steps, start):
        self.type = 'random'
        self.go = go
        self.piece_type = piece_type
        self.steps = steps
        self.start = start
        self.int_max = 2147483647
        self.int_min = -2147483648
        self.size = 5
        self.max_step = 5 * 5 - 1
        self.max_depth = 4

    # return best action
    def action(self):
        # try to occupy [2,2] if available
        if (self.steps == 1 or self.steps == 2) and self.go.board[2][2] == 0:
            print(f'My move is {(2,2)}')
            return (2,2)

        # get possible placements
        possible_placements = go.get_possible_placements()

        # if no possible_placements, return "PASS"
        if not possible_placements:
            "PASS"
        # if there are possible placements, do minimax to decide best move
        move = self.minimax(possible_placements)
        print(f'My move is {move}')
        return move


    # return minimax solution
    def minimax(self, possible_placements):
        best_next_move = random.choice(possible_placements)
        best_score = self.int_min
        # placement reward
        reward = []
        reward.append((-100, 10, 5, 10, -100))
        reward.append((15, 15, 20, 15, 10))
        reward.append((5, 20, 100, 20, 5))
        reward.append((10, 15, 20, 15, 10))
        reward.append((-100, 10, 5, 10, -100))
        best_next_score = 0
        # for each possible placements, check their score
        for move in possible_placements:
            if(time.time() - start > 9500):
                print('Minimax terminated due to one round time limit')
                break
            n_go = GO.new_Go(go, move[0], move[1], self.piece_type)
            n_score = self.min_val(n_go, 1, self.steps + 1, 3 - self.piece_type, self.int_min, self.int_max)
            score = n_score + n_go.loc_evaluation(move[0],move[1]) + reward[move[0]][move[1]]
            if score > best_score:
                best_score = score
                best_next_move = move
                best_next_score = best_score

        return best_next_move
        # return random.choice(possible_placements)

    def min_val(self, go, depth, step, piece_type, alpha, beta):
        # when depth is over max_depth or step over max_step, return current heuristic val
        if depth > self.max_depth or step > self.max_step:
            return go.heuristic()
        # get all possible placements, if there is none, return current heuristic val
        possible_placements = go.get_possible_placements()
        if not possible_placements:
            return go.heuristic()
        # check all moves
        best_value = self.int_max
        for move in possible_placements:
            n_go = GO.new_Go(go, move[0], move[1], piece_type)
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
        if depth > self.max_depth or step > self.max_step:
            return go.heuristic()
        # get all possible placements, if there is none, return current heuristic val
        possible_placements = go.get_possible_placements()
        if not possible_placements:
            return go.heuristic()
        # check all moves
        best_value = self.int_min
        for move in possible_placements:
            n_go = GO.new_Go(go, move[0], move[1], piece_type)
            n_score = self.min_val(n_go, depth + 1, step + 1, 3 - piece_type, alpha, beta)
            if n_score > best_value:
                best_value = n_score
            if best_value >= beta:
                return best_value
            if best_value > alpha:
                alpha = best_value
        return best_value



if __name__ == "__main__":
    # get start timestamp
    start = time.time()

    # read input and create go object
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)

    # take go input and return action
    steps = 0
    for i in range(5):
        for j in range(5):
            if go.board[i][j] != 0:
                steps += 1
    steps += 1

    # steps = read.readHelper()

    player = myPlayer(go, piece_type, steps, start)
    action = player.action()

    # write output action and update steps
    writeOutput(action)
    write.writeHelper(steps + 1)

    # print time info
    end = time.time()
    print(f'This step total time used is: {end - start}')