import torch
import math
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


import numpy as np


def paraGeneration(options):
    np.random.seed(2030)
    # CPU clock speed f_ 
    cpu_frequency = [np.round(np.random.uniform(0.01, 5), decimals=2) for i in range(options['num_of_clients'])]
    # B = np.round(np.random.randint(1, 20, size = options['num_of_clients']), decimals=1)
    np.random.shuffle(cpu_frequency)
    # print(cpu_frequency)
    transmit_rate = [np.round(np.random.uniform(0.1, 10), decimals=1) for i in range(options['num_of_clients'])]
    transmit_power = [np.round(np.random.uniform(0, 10), decimals=1) for i in range(options['num_of_clients'])]
    # print(len(transmit_power))
    return cpu_frequency, transmit_rate, transmit_power

if __name__ == '__main__':
    options = {
        'round_num': 100,
        'num_of_clients': 3
    }
    print(paraGeneration(options))