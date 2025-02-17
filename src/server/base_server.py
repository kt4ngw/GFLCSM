import numpy as np
import torch
import time
from src.client.base_client import Client
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import copy
import torch.nn.functional as F
from src.utils.metrics import Metrics
from cost import Clients_System_Attr, Cost

criterion = F.cross_entropy

from src.utils.torch_utils import *


class BaseFederated(object):

    def __init__(self, options, dataset, clients_label, clients_system_attr, model=None, optimizer=None,
                 name=''):
        if model is not None and optimizer is not None:
            self.model = model
            self.optimizer = optimizer
        self.options = options
        self.dataset = dataset
        self.clients_label = clients_label
        self.gpu = options['gpu']
        self.batch_size = options['batch_size']
        self.num_round = options['round_num']
        self.per_round_c_fraction = options['c_fraction']
        self.client_system_attr = clients_system_attr
        self.clients = self.setup_clients(self.dataset, self.clients_label)
        self.clients_num = len(self.clients)
        self.name = '_'.join([name, f'wn{int(self.per_round_c_fraction * self.clients_num)}',  f'tn{len(self.clients)}'])
        self.clients_own_datavolume = [len(client.local_dataset) for client in self.clients]
        self.metrics = Metrics(options, self.clients, self.name)
        self.label_composition_truth = self.get_clients_label_composition_truth(self.clients, self.dataset,
                                                                                self.clients_label)
        self.latest_global_model = self.get_flat_model_params()
        # print("self.max_iterations", self.max_iterations)
        self.cost_calculation = Cost()


    @staticmethod
    def move_model_to_gpu(model, options):
        if options['gpu'] >= 0:
            device = options['gpu']
            torch.cuda.set_device(device)
            # torch.backends.cudnn.enabled = True
            model.cuda()
            print('>>> Use gpu on device {}'.format(device))
        else:
            print('>>> Don not use gpu')

    def get_model_parameters(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_parameters(self, model_parameters_dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_parameters_dict[key]
        self.model.load_state_dict(state_dict)

    def load_model_parameters(self, file):
        model_params_dict = get_state_dict(file)
        self.set_model_params(model_params_dict)

    # def get_flat_model_params(self):
    #     flat_params = get_flat_params_from(self.model)
    #     return flat_params.detach()

    def get_flat_model_params(self):
        flat_feature_extractor_params = get_flat_params_from(self.model.feature_extractor)
        flat_classifier_params = get_flat_params_from(self.model.classifier)
        return torch.cat((flat_feature_extractor_params, flat_classifier_params)).detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)

    def get_flat_gradients(self, dataloader):
        self.optimizer.zero_grad()
        loss, total_num = 0., 0
        for x, y in dataloader:
            x = self.flatten_data(x)
            if self.gpu >= 0:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x)
            loss += criterion(pred, y) * y.size(0)
            total_num += y.size(0)
        loss /= total_num
        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        return flat_grads

    def train(self):
        """The whole training procedure
        No returns. All results all be saved.
        """
        raise NotImplementedError

    def setup_clients(self, dataset, clients_label):
        train_data = dataset.train_data
        train_label = dataset.train_label
        all_client = []
        for i in range(len(clients_label)):
            local_client = Client(self.options, i, self.model, self.optimizer,
                                  TensorDataset(torch.tensor(train_data[self.clients_label[i]]),
                                                torch.tensor(train_label[self.clients_label[i]])), self.client_system_attr)
            all_client.append(local_client)

        return all_client

    def local_train(self, round_i, select_clients, ):
        local_model_paras_set = []
        stats = []
        for i, client in enumerate(select_clients, start=1):
            client.set_flat_model_params(self.latest_global_model)

            local_model_paras, stat = client.local_train()
            local_model_paras_set.append(local_model_paras)
            stats.append(stat)
            if True:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                    round_i, client.id, i, int(self.per_round_c_fraction * self.clients_num),
                    stat['loss'], stat['acc'] * 100, stat['time'], ))
        return local_model_paras_set, stats

    def aggregate_parameters(self, solns, **kwargs):
        """Aggregate local solutions and output new global parameter

        Args:
            solns: a generator or (list) with element (num_sample, local_solution)

        Returns:
            flat global model parameter
        """

        averaged_solution = torch.zeros_like(self.latest_global_model)
        # averaged_solution = np.zeros(self.latest_model.shape)
        self.simple_average = False
        if self.simple_average:
            num = 0
            for num_sample, local_solution in solns:
                num += 1
                averaged_solution += local_solution
            averaged_solution /= num
        else:
            num = 0
            for num_sample, local_solution in solns:
                # print(local_solution)
                num += num_sample
                averaged_solution += num_sample * local_solution
            averaged_solution /= num

        # averaged_solution = from_numpy(averaged_solution, self.gpu)
        return averaged_solution.detach()

    def test_latest_model_on_testdata(self, round_i):
        # Collect stats from total test data
        begin_time = time.time()
        stats_from_test_data = self.global_test(use_test_data=True)
        end_time = time.time()

        if True:
            print('= Test = round: {} / acc: {:.3%} / '
                  'loss: {:.4f} / Time: {:.2f}s'.format(
                round_i, stats_from_test_data['acc'],
                stats_from_test_data['loss'], end_time - begin_time))
            print('=' * 102 + "\n")

        self.metrics.update_test_stats(round_i, stats_from_test_data)

    def global_test(self, use_test_data=True):
        assert self.latest_global_model is not None
        self.set_flat_model_params(self.latest_global_model)
        testData = self.dataset.test_data
        testLabel = self.dataset.test_label
        testDataLoader = DataLoader(TensorDataset(torch.tensor(testData), torch.tensor(testLabel)), batch_size=100,
                                    shuffle=False)
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for X, y in testDataLoader:
                if self.gpu >= 0:
                    X, y = X.cuda(), y.cuda()
                feature, pred = self.model(X)
                loss = criterion(pred, y)

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum()
                test_acc += correct.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        stats = {'acc': test_acc / test_total,
                 'loss': test_loss / test_total,
                 'num_samples': test_total, }
        # a = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
        return stats

    def get_clients_label_composition_truth(self, clients, dataset, clients_label):
        class_num = 10
        if self.options['dataset_name'] == "emnist":
            class_num = 47
        clients_own_label_turth = []
        train_label = dataset.train_label
        for i, client in enumerate(clients, start=0):
            result = [0 for _ in range(class_num)] # np.zeros(class_num)
            for j in range(len(train_label[clients_label[i]])):
                result[train_label[clients_label[i]][j]] += 1
            clients_own_label_turth.append(result)
        return clients_own_label_turth

    def get_each_class_vloume(self, selected_clients):
        class_num = 10
        if self.options['dataset_name'] == "emnist":
            class_num = 47
        D = [0 for _ in range(class_num)]
        for i in selected_clients:
            for j in range(class_num):
                D[j] += i.class_distribution[j]
        return D
