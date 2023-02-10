# import math
# import random
import numpy as np


class Config:
    def __init__(self):
        # parameters of genetic algo
        self.generations = 500
        self.population_size = 5
        self.init_heuristic_population_size = 10
        self.init_population_size = 10
        self.elite_size = 1
        self.mutation_rate = 0.002


class GeneticSearching:

    def __init__(self):
        self.config = Config()

        # raw data from input file
        self.num_of_cities = 0
        self.list_of_cities_loca = []

        self.path_scores = []
        self.sum_of_scores = 0

        self.num_of_mutation = 0

    # class run entry
    def run(self, file_address):

        # read input file
        self.file_reader(file_address)
        population = self.get_init_population()

        for i in range(0, self.config.generations):
            population = self.get_new_generation(population)

        # return sorted_population(self.init_population)[0]
        score = self.get_score(population[0])
        bestRoute = population[0]
        for i in range(1, self.config.elite_size):
            newscore = self.get_score(population[i])
            if score > newscore:
                score = newscore
                bestRoute = population[i]

        re = []
        for i in range(-1, len(bestRoute)):
            re.append(self.list_of_cities_loca[bestRoute[i]])

        self.write_to_output(re)

    # read and collection data from input file
    def file_reader(self, file_address):
        inputFile = open(file_address, 'r')
        lines = inputFile.readlines()
        self.num_of_cities = int(lines[0])

        for i in range(1, len(lines)):
            coordinate = []
            for j in lines[i].split(" "):
                coordinate.append(int(j))
            self.list_of_cities_loca.append(coordinate)

    # distance between two cities
    def get_dist(self, city_a_idx, city_b_idx):
        a_loca = self.list_of_cities_loca[int(city_a_idx)]

        b_loca = self.list_of_cities_loca[int(city_b_idx)]

        return np.sqrt((a_loca[0] - b_loca[0]) ** 2 + (a_loca[1] - b_loca[1]) ** 2 + (a_loca[2] - b_loca[2]) ** 2)


    # get heuristic path
    def get_heuristic_path(self):
        path = []
        visited = set()
        pre_city = np.random.randint(0, self.num_of_cities)
        path.append(pre_city)
        visited.add(pre_city)
        while len(visited) != self.num_of_cities:
            if len(visited) == self.num_of_cities - 1:
                for i in range(self.num_of_cities):
                    if i not in visited:
                        path.append(i)
                        visited.add(i)
                        break
            else:
                dist = -1
                for i in range(self.num_of_cities):
                    if i not in visited:
                        temp = self.get_dist(pre_city, i)
                        if dist == -1:
                            dist = temp
                            pre_city = i
                        else:
                            if dist > temp:
                                dist = temp
                                pre_city = i
                path.append(pre_city)
                visited.add(pre_city)
        return path

    # get initial population
    def get_init_population(self):
        init_population = []
        # get initial populations
        while self.config.init_heuristic_population_size > 0:
            temp_path = self.get_heuristic_path()
            init_population.append(temp_path)
            self.config.init_heuristic_population_size -= 1
        print(init_population)
        while self.config.init_population_size > 0:
            temp_path = list(np.random.permutation(self.num_of_cities))
            init_population.append(temp_path)
            self.config.init_population_size -= 1
        print(init_population)
        population = sorted(init_population)
        print(population)
        return population

    # get score of a path
    def get_score(self, temp_path):
        score = 0
        for i in range(0, len(temp_path)):
            city_a_idx = temp_path[i - 1]
            city_b_idx = temp_path[i]
            a_loca = self.list_of_cities_loca[int(city_a_idx)]
            b_loca = self.list_of_cities_loca[int(city_b_idx)]
            cur_distance = np.sqrt(
                (a_loca[0] - b_loca[0]) ** 2 + (a_loca[1] - b_loca[1]) ** 2 + (a_loca[2] - b_loca[2]) ** 2)
            score += cur_distance
        return score

    # rank population by path score
    def population_rank_by_score(self, population):
        pop = []
        self.path_scores = []
        temp_sum_of_scores = 0
        for i in range(len(population)):
            score = self.get_score(population[i])
            temp_sum_of_scores += score
            cur_tuple = (i, score)
            pop.append(cur_tuple)
        self.sum_of_scores = temp_sum_of_scores

        pop = sorted(pop, key=lambda t: t[1])
        sorted_population = []
        for i in range(len(pop)):
            sorted_population.append(pop[i][0])
            self.path_scores.append(pop[i][1])

        return sorted_population

    # create mating_pool
    def create_mating_pool(self, population, ranked_list):
        mating_poll = []
        # select best chromes
        sum_of_scores = 0
        path_score = []
        for i in range(len(ranked_list)):
            cur_score = self.get_score(population[ranked_list[i]])
            path_score.append(cur_score)
            sum_of_scores += cur_score

        path_score_per = []
        roulette_wheel = []

        temp = 0
        for i in range(len(ranked_list)):
            path_score_per.append(self.path_scores[i] / self.sum_of_scores * 100)
            temp += path_score_per[i]
            roulette_wheel.append(temp)

        temp = []
        for i in range(self.config.elite_size):
            mating_poll.append(population[ranked_list[i]])
            temp.append(ranked_list[i])
        for i in range(len(ranked_list) - self.config.elite_size):
            probability = np.random.uniform(0, 1) * 100

            for j in range(len(roulette_wheel)):
                if probability <= roulette_wheel[j]:
                    mating_poll.append(population[ranked_list[j]])
                    temp.append(ranked_list[j])
                    break
        return mating_poll

    # crossover between two path
    @staticmethod
    def crossover(path1, path2):
        index1 = np.random.randint(0, len(path1))
        index2 = np.random.randint(0, len(path2))
        start_idx = min(index1, index2)
        end_idx = max(index1, index2)

        path_mid = []

        for i in range(start_idx, end_idx + 1):
            path_mid.append(path1[i])

        for city in path2:
            if city not in path_mid:
                path_mid.append(city)

        return path_mid

    # hybrid generation
    def hybrid(self, matingpool):
        new_pop1 = []
        for i in range(self.config.elite_size):
            new_pop1.append((matingpool[i]))

        for i in range(len(matingpool) - self.config.elite_size):
            pick_idx1 = np.random.randint(0, len(matingpool))
            pick_idx2 = np.random.randint(0, len(matingpool))

            newpath = self.crossover(matingpool[pick_idx1], matingpool[pick_idx2])
            new_pop1.append(newpath)

        return new_pop1

    # mutate
    def mutate(self, path):
        for city1 in range(len(path)):
            p = np.random.sample()
            # print(p)
            # print(self.config.mutation_rate)
            if p < self.config.mutation_rate:
                # print(self.num_of_mutation)
                self.num_of_mutation += 1
                # print(path)
                city2 = np.random.randint(0, len(path))
                temp = path[city1]
                path[city1] = path[city2]
                path[city2] = temp

                # print(path)


        return path

    # mutation
    def mutation(self, new_pop1):
        mutated_paths = []
        for cur_path in new_pop1:
            mutated_path = self.mutate(cur_path)
            mutated_paths.append(mutated_path)
        return mutated_paths

    # generate new population
    def get_new_generation(self, population):
        ranked_list = self.population_rank_by_score(population)
        mating_pool = self.create_mating_pool(population, ranked_list)
        new_pop1 = self.hybrid(mating_pool)
        new_pop2 = self.mutation(new_pop1)
        return new_pop2

    # write to output file
    @staticmethod
    def write_to_output(re):
        with open(r'./output.txt', 'w') as fp:
            for item in re:
                temp = ''
                for i in item:
                    temp += str(i) + ' '
                # write each item on a new line
                temp = temp[:-1]
                temp += '\n'
                fp.write(temp)


if __name__ == "__main__":
    solution = GeneticSearching()
    solution.run(file_address="./input3.txt")
