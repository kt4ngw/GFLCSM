import random
import pickle
import os
import numpy as np
import copy

np.random.seed(2024)
def kl_banlance(D_distribution, clients_data_distribution):
    D = copy.deepcopy(D_distribution)
    for i in range(len(D)):
        D[i] += clients_data_distribution[i]
    sum_sample = sum(D)
    kl_div = 0
    P = np.array(D) / sum_sample
    Q = np.full(len(D), 1 / len(D))
    kl_div = np.sum(P * np.log(P / Q))
    return kl_div

def kl_banlance_t(D_distribution,):
    D = copy.deepcopy(D_distribution)
    sum_sample = sum(D)
    kl_div = 0
    P = np.array(D) / sum_sample
    Q = np.full(len(D), 1 / len(D))
    kl_div = np.sum(P * np.log(P / Q))
    return kl_div


class Group_KL_Maker():
    def __init__(self, clients_data_distribution, clients_representer, options):
        self.clients_data_distribution = clients_data_distribution
        self.clients_representer = clients_representer
        # print(self.clients_representer)
        self.options = options
        if self.options['method_division'] == 1:
            self.script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'groupfile_kl', 'dirichlet')
        else:
            self.script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'groupfile_kl', 'pathology')
        os.makedirs(self.script_dir, exist_ok=True)
        if self.options['is_real_class'] == True:
            self.suffix = 'dn_{}_noc_{}_dir_{}_'.format(
                                            self.options['dataset_name'],
                                            self.options['num_of_clients'],
                                            self.options['dirichlet'],
                                            )
        else:
            self.suffix = 'dn_{}_noc_{}_dir_{}_{}_'.format(
                                            self.options['dataset_name'],
                                            self.options['num_of_clients'],
                                            self.options['dirichlet'],
                                            "estimate"
                                            )
        self.load_or_generate_group()

    def calculate_fitness(self, population, clients_representer):
        # Calculate the fitness of each group, for example by minimizing the imbalance in data distribution
        fitness_scores = []
        for groups in population:
            fitness_group = []
            for i in range(0, len(groups), int(len(groups) /10)):
                new_list = groups[i:i + int(len(groups) /10)]
                D_distribution = np.sum([clients_representer[client] for client in new_list], axis=0)
                fitness_group.append(-kl_banlance_t(D_distribution))
            fitness = sum(fitness_group) / len(fitness_group)
            fitness_scores.append(fitness)
        return fitness_scores
    def crossover(self, parent1, parent2):
    # Single point crossover
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + [client for client in parent2 if client not in parent1[:point]]
        child2 = parent2[:point] + [client for client in parent1 if client not in parent2[:point]]
        return child1, child2

    def mutate(self, group, mutation_rate):
        if random.random() < mutation_rate:
            swap_idx1, swap_idx2 = random.sample(range(len(group)), 2)
            group[swap_idx1], group[swap_idx2] = group[swap_idx2], group[swap_idx1]
        return group
    def get_group(self, is_real_class, population_size=100, num_generations=100, mutation_rate=0.01):
        if is_real_class == False:
            population = []
            for _ in range(population_size):
                clients_index = random.sample(range(self.options['num_of_clients']), self.options['num_of_clients'])
                population.append(clients_index) 
            for _ in range(num_generations):
                fitness_scores = self.calculate_fitness(population, self.clients_representer)
                sorted_population = [groups for _, groups in sorted(zip(fitness_scores, population), reverse=True)]
                population = sorted_population[:population_size//2]
                next_generation = []
                while len(next_generation) < population_size:   
                    parent1, parent2 = random.sample(population, 2)
                    child1, child2 = self.crossover(parent1, parent2)
                    print(child1, child2)
                    child1 = self.mutate(child1, mutation_rate)
                    child2 = self.mutate(child2, mutation_rate)
                    next_generation.extend([child1, child2])
                population = next_generation
            fitness_scores = self.calculate_fitness(population, self.clients_representer)
            best_groups = population[np.argmax(fitness_scores)]
        G = {}
        for i in range(0, 10):
            new_list = best_groups[i * int(len(best_groups) /10):(i + 1) * int(len(best_groups) / 10)]
            G[i] = new_list
        print(G)
        return G
    # def get_group(self, is_real_class):
    #     if is_real_class == False:
    #         a = []
    #         G = {}
    #         k = 0
    #         clients_index = [_ for _ in range(self.options['num_of_clients'])]
    #         while len(clients_index) != 0:
    #             group = []
    #             D_distribution = [0 for _ in range(len(self.clients_representer[0]))]
    #             g1 = random.choice(clients_index)
    #             for i in range(len(D_distribution)):
    #               D_distribution[i] += self.clients_representer[g1][i]
    #             group.append(g1)
    #             clients_index.remove(g1)
    #             # while (len(group) < (self.options['num_of_clients'] / 10)) and (sum(D_distribution) < 800):
    #             while (len(group) < (self.options['num_of_clients'] / 10)):
    #                 more_banlance_client = clients_index[0]
    #                 for j in clients_index:
    #                     # 将j和group组合, 然后判断最小
    #                     # If group + 第一个客户端的距离 比 group + 第二个客户端的距离 更平衡，那么就选择第一个客户端;
    #                     if kl_banlance(D_distribution, self.clients_representer[j]) \
    #                             < kl_banlance(D_distribution, self.clients_representer[more_banlance_client]):
    #                         more_banlance_client = j
    #                 group.append(more_banlance_client)
    #                 for i in range(len(D_distribution)):
    #                     D_distribution[i] += self.clients_representer[more_banlance_client][i]
    #                 clients_index.remove(more_banlance_client)
    #             a.append(D_distribution)
    #             G[k] = group
    #             k += 1
    #         print(a)
    #         return G
    #     else:
    #         print(1)
    #         a = []
    #         G = {}
    #         k = 0
    #         clients_index = [_ for _ in range(self.options['num_of_clients'])]
    #         while len(clients_index) != 0:
    #             group = []
    #             D_distribution = [0 for _ in range(len(self.clients_data_distribution[0]))]
    #             g1 = random.choice(clients_index)
    #             for i in range(len(D_distribution)):
    #               D_distribution[i] += self.clients_data_distribution[g1][i]
    #             group.append(g1)
    #             clients_index.remove(g1)
    #             # while (len(group) < (self.options['num_of_clients'] / 10)) and (sum(D_distribution) < 800):
    #             while (len(group) < (self.options['num_of_clients'] / 10)):
    #                 more_banlance_client = clients_index[0]
    #                 for j in clients_index:
    #                     # 将j和group组合, 然后判断最小
    #                     # If group + 第一个客户端的距离 比 group + 第二个客户端的距离 更平衡，那么就选择第一个客户端;
    #                     if kl_banlance(D_distribution, self.clients_data_distribution[j]) \
    #                             < kl_banlance(D_distribution, self.clients_data_distribution[more_banlance_client]):
    #                         more_banlance_client = j
    #                 group.append(more_banlance_client)
    #                 for i in range(len(D_distribution)):
    #                     D_distribution[i] += self.clients_data_distribution[more_banlance_client][i]
    #                 clients_index.remove(more_banlance_client)
    #             a.append(D_distribution)
    #             G[k] = group
    #             k += 1
    #         print(a)
    #         return G

    def load_or_generate_group(self, filename='group.pkl'):
        filename = self.suffix + filename
        print("self.script_dir", self.script_dir)
        try:
            # 尝试加载已保存的文件
            self.load_group(os.path.join(self.script_dir, filename))
            print("Group loaded from file.")
        except FileNotFoundError:
            # 如果文件不存在，生成新的group并保存
            self.G = self.get_group(self.options['is_real_class'])
            self.save_group(os.path.join(self.script_dir, filename))
            print("New group generated and saved to file.")


    def save_group(self, filename='group.pkl'):
 
        with open(os.path.join(self.script_dir, filename), 'wb') as file:
            pickle.dump(self.G, file)

    def load_group(self, filename='group.pkl'):

        with open(os.path.join(self.script_dir, filename), 'rb') as file:
            self.G = pickle.load(file)

