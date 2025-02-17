import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from getdata import GetDataSet
import argparse
import torch
from getdata import GetDataSet
from src.utils.dirichlet import dirichlet_split_noniid
import importlib
from src.utils.tool_utils import setup_seed
from src.utils.tool_utils import paraGeneration
from src.utils.imbanlance import split_imbalance, split_imbalance_two
from src.models.model import choose_model
from cost import Clients_System_Attr
from src.utils.data_div import IID, IID_RANDOM, dirichlet

# GLOBAL PARAMETERS
DATASETS = ['mnist', 'fashionmnist', 'cifar10']
TRAINERS = {'fedavg': 'FedAvgTrainer',
            'proposed': 'FedETrainer',
            'opti':'OptiSampling',
            'isfedavg':'ISFedAvgTrainer',
            'poc':'POCTrainer',
            'fedprox':'FedProxTrainer',
            'delta':'Delta',
            'pproposed': 'PFedETrainer',
            'ceopti':'CEOPTI',
            'group_fel':'Group_FEL',
            'groupkl':"Group_KL",
            'fedis': 'FedPracIS',
            }


OPTIMIZERS = TRAINERS.keys()
def input_options():
    parser = argparse.ArgumentParser()
    # iid
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='name of dataset.')
    parser.add_argument('--model_name', type=str, default='cifar10_alexnet', help='the model to train')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id to use(e.g. 0,1,2,3)')
    parser.add_argument('--round_num', type=int, default=200, help='number of round in comm')
    parser.add_argument('--num_of_clients', type=int, default=200, help='numer of the clients')
    parser.add_argument( '--c_fraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
    parser.add_argument('--local_epoch', type=int, default=5, help='local train epoch')
    parser.add_argument( '--batch_size', type=int, default=128, help='local train batch size')
    parser.add_argument( "--lr", type=float, default=0.1, help="learning rate, use value from origin paper as default")
    parser.add_argument('--seed', type=int, default=0, help='seed for randomness;')
    parser.add_argument( '--server', type=str, default='proposed', help='server')
    parser.add_argument('--opti', type=str, default='gd', help='optimize_;')
    parser.add_argument( '--method_division', type=int, default=1, help='Ways to classify Non-IID, 1 for Dirichlet, 2 for Pathological Distribution;')
    parser.add_argument( '--dirichlet', default=0.1, type=float, help='Dirichlet;')
    parser.add_argument( '--C', type=int, default=200000, help='comptu. one sample.',)

    ## proposed and fedavg
    parser.add_argument( '--sample', type=bool, default=False, help='whether the sampling algorithm is enabled in proposed.')
    parser.add_argument('--is_real_class', type=bool, default=False, help='is or is not evaluate class;')
    parser.add_argument('--MaXCoV', type=float, default=1, help=';')

    ## proposed
    parser.add_argument('--pre_epoch', type=int, default=10, help='pre local train epoch')
    parser.add_argument('--I', type=float, default=0.5, help='balance factor.')
    parser.add_argument('--omega3', type=float, default=0.6, help='balance factor.')
    args = parser.parse_args()
    options = args.__dict__
    dataset = GetDataSet(options['dataset_name'][:]) # 拿到数据集 分配完再导入
    options['model_size'] = choose_model(options).get_model_size()
    options['classifier_model_size'] = choose_model(options).get_model_size('classifier')
    print(options['model_size'])
    if options['method_division'] == 1:
        client_label, result = dirichlet_split_noniid(dataset.train_label, options['dirichlet'], options['num_of_clients'])
    elif options['method_division'] == 2:
        client_label = split_imbalance_two(dataset.train_label, options['num_of_clients'])
        # 保存客户端标签到文本文件
    # 将列表元素连接成一个字符串，以逗号分隔
    list_str = ' '.join(map(str, dataset.train_label.tolist()))
    with open('client_labels0.15.txt', 'w') as file:
        for label in result[:20]:
            for i in label:
                file.write(f'{i} ')
            file.write(f'\n')
        for i in list_str:
            file.write(f'{i}')
        file.write(f'\n')
    return client_label, options, dataset


def main():

    client_label, options, dataset = input_options()
    cpu_frequency, transmit_rate, transmit_power = paraGeneration(options)
    clients_system_attr = Clients_System_Attr(cpu_frequency, transmit_rate, transmit_power)
    trainer_path = 'src.server.%s' % options['server']
    mod = importlib.import_module(trainer_path)
    trainer_class = getattr(mod, TRAINERS[options['server']])
    Fed = trainer_class(options, dataset, client_label, clients_system_attr)
    setup_seed(options['seed'])
    Fed.train()

if __name__ == '__main__':
    main()