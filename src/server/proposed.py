from src.server.base_server import BaseFederated
from src.models.model import choose_model
from src.optimizers.gd import GD
import numpy as np
from tqdm import tqdm
from src.optimizers.adam import MyAdam
from torch.optim import SGD, Adam
import copy
import torch
from src.group_module.group_maker import Group_Maker
import os 
import pickle

def ba(D):
    sum_sample = sum(D)
    mathcal_Q = 0
    for i in range(len(D)):
        mathcal_Q += (D[i] / sum_sample - 1 / abs(len(D))) ** 2
    return mathcal_Q

class FedETrainer(BaseFederated):
    def __init__(self, options, dataset, clients_label, clients_system_attr):
        model = choose_model(options)
        print(model)
        self.move_model_to_gpu(model, options)
        self.optimizer = GD(model.parameters(), lr=options['lr']) 
        super(FedETrainer, self).__init__(options, dataset, clients_label, clients_system_attr, model, self.optimizer, )
        self.script_dir = f"./src/group_module/estimate_distribution"
        filename = f"dn_{self.options['dataset_name']}_noc_{self.options['num_of_clients']}_dir_{self.options['dirichlet']}" + '_estimate_distribution.pkl'
        try:
            # 尝试加载已保存的文件
            with open(os.path.join(self.script_dir, filename), 'rb') as file:
                self.estimate_data_distribution = pickle.load(file)
            print("Estimate distribution loaded from file.")
        except FileNotFoundError:
            self.pre_model = self.get_flat_model_params()
            self.estimate_data_distribution = self.pretrain()
        # print("self.estimate_data_distribution", self.estimate_data_distribution)
        self.clients_data_distribution = self.combine_clients_data_distribution(self.clients)
        # print(self.clients_data_distribution)
        self.group = Group_Maker(self.label_composition_truth, self.estimate_data_distribution, self.options)
        # print(self.group.G)
        # self.g = self.group.K_MEANS()
        self.group_num = len(self.group.G)
        self.B_groups = self.get_groups_B(self.estimate_data_distribution, self.group.G)

        self.pre_energy_comsumption = self.get_pre_energy_comsumption()
        # print("self.pre_energy_comsumption", self.pre_energy_comsumption)
        self.metrics.update_pre_compustion(self.pre_energy_comsumption )
    def get_pre_energy_comsumption(self, ):
        pre_energy_comsumption = 0
        pre_energy_comsumption_local = 0
        pre_energy_comsumption_upload = 0
        for client in self.clients:
            pre_energy_comsumption_local += client.getLocalEnergy_pre(1) # * self.options['pre_epoch'] # / self.options['local_epoch']
            pre_energy_comsumption_upload += client.getUploadEnergy_pre(1) 
        pre_energy_comsumption = pre_energy_comsumption_local + pre_energy_comsumption_upload
        print(pre_energy_comsumption_local, pre_energy_comsumption_upload)
        return pre_energy_comsumption
    

    def combine_clients_data_distribution(self, clients):
        clients_data_distribution = []
        for client in clients:
            clients_data_distribution.append(client.class_distribution)

        return clients_data_distribution
    def get_groups_B(self, estimate_data_distribution, Groups):
        B_groups = []
        for i in range(len(Groups)):
            for client in Groups[i]:
                D_dis = [0 for i in range(10)]
                for c in range(10):
                    D_dis[c] += estimate_data_distribution[client][c]
            B = ba(D_dis)
            B_groups.append(B)
        return B_groups
    
    def pretrain(self):
        local_model_paras_set, stats = self.pre_train(-1, self.clients)
        estimate_data_distribution = self.get_estimated_distribution(local_model_paras_set)
        script_dir = f"./src/group_module/estimate_distribution"
        os.makedirs(script_dir, exist_ok=True)
        filename = f"dn_{self.options['dataset_name']}_noc_{self.options['num_of_clients']}_dir_{self.options['dirichlet']}" + '_estimate_distribution.pkl'
        with open(os.path.join(script_dir, filename), 'wb') as f:
            pickle.dump(estimate_data_distribution, f)
        return estimate_data_distribution

    def get_estimated_distribution(self, local_model_paras_set):
        count = 0
        estimate_data_distribution = []
        for i in range(len(local_model_paras_set)):
            self.set_flat_model_params(self.latest_global_model)
            model_weights = self.model.state_dict()
            # for name, weight in model_weights.items():
            #     print(name, weight.size())
            classifier_server_para = copy.deepcopy(model_weights['linear.weight'])
            self.set_flat_model_params(local_model_paras_set[i][1])
            model_weights = self.model.state_dict()
            classifier_client_i_para = model_weights['linear.weight']
            class_weights_server_para = torch.split(classifier_server_para, split_size_or_sections=1, dim=0)
            class_weights_client_i_para = torch.split(classifier_client_i_para, split_size_or_sections=1, dim=0)

            # 计算每个类别的权重大小（L2 范数）
            class_weight_sizes = [torch.norm(class_weight_server_para - class_weight_client_i_para, p=2).item() \
                                  for class_weight_server_para, class_weight_client_i_para in \
                                  zip(class_weights_server_para, class_weights_client_i_para)]
            class_weight_sizes = [i / sum(class_weight_sizes) for i in class_weight_sizes]

            client_i_data_v = sum(self.label_composition_truth[i])
            pred = [client_i_data_v * i for i in class_weight_sizes]

            pred_class_tensor = torch.tensor(pred)
            truth_class_tensor = torch.tensor(self.label_composition_truth[i], dtype=torch.float32)
            # 计算向量的点积
            dot_product = torch.dot(pred_class_tensor, truth_class_tensor)
            # 计算向量的 L2 范数
            norm1 = torch.norm(pred_class_tensor)
            norm2 = torch.norm(truth_class_tensor)
            # 计算余弦相似度
            cosine_similarity = dot_product / (norm1 * norm2)
            # 打印结果
            estimate_data_distribution.append(pred)
            print(cosine_similarity)
        return estimate_data_distribution

    def train(self):
        
        print('=== Select {} clients per round ===\n'.format(int(self.per_round_c_fraction * self.clients_num)))

        for round_i in range(self.num_round):
            self.test_latest_model_on_testdata(round_i)
            if self.options['sample'] == True:
                sampling_probability = self.get_energy_sampling_probability(round_i, self.B_groups )
            else:
                sampling_probability = None
            selected_clients = self.select_group(self.group.G, sampling_probability)
            local_model_paras_set, stats = self.local_train(round_i, selected_clients)

            self.latest_global_model = self.aggregate_parameters(local_model_paras_set)
            
            self.metrics.update_costs(round_i, self.cost_calculation.get_latency_energy(selected_clients, round_i))
            print(self.metrics.accumulation_energy)
            self.optimizer.soft_decay_learning_rate()

        self.test_latest_model_on_testdata(self.num_round)
        self.metrics.write()
    
    
    def pre_train(self, round_i, select_clients):
        local_model_paras_set = []
        stats = []
        for i, client in enumerate(select_clients, start=1):
            client.set_flat_model_params(self.pre_model)

            local_model_paras, stat = client.pre_local_train()
            local_model_paras_set.append(local_model_paras)
            stats.append(stat)
            if True:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                    round_i, client.id, i, int(self.clients_num),
                    stat['loss'], stat['acc'] * 100, stat['time'], ))
        return local_model_paras_set, stats
    
    def select_group(self, G, sampling_probability):
        index = []
        g_num = 1
        # sampling_probability = [tensor.item(),  for tensor in sampling_probability]
        # print(sum(sampling_probability))
        if sampling_probability == None:
            b_index = np.random.choice(len(self.group.G), g_num, replace=False)
        else:
            b_index = np.random.choice(len(self.group.G), g_num, replace=False, p=sampling_probability)
        index.extend(b_index)
        select_clients = []
        for g in index:
            for i in G[g]:
                select_clients.append(self.clients[i])
        return select_clients   
    


    # ----------------------------------------------------------- #
    # def get_sampling_probability(self, round_i):
    #     group_local_latency = []
    #     group_local_energy = []
    #     group_upload_latency = []
    #     group_upload_energy = []
    #     for group in self.group.G:
    #         local_latency_temp = 0
    #         upload_latency_temp = 0
    #         local_energy  = 0
    #         upload_energy = 0
    #         for client in self.group.G[group]:
    #             if self.clients[client].getLocalDelay(round_i) > local_latency_temp:
    #                 local_latency_temp = self.clients[client].getLocalDelay(round_i)
    #             if self.clients[client].getUploadDelay(round_i) > upload_latency_temp:
    #                 upload_latency_temp = self.clients[client].getUploadDelay(round_i)
    #             local_energy = self.clients[client].getLocalEnergy(round_i)
    #             upload_energy = self.clients[client].getUploadEnergy(round_i)
    
    #         group_local_latency.append(local_latency_temp)
    #         group_local_energy.append(local_energy)
    #         group_upload_latency.append(upload_latency_temp)
    #         group_upload_energy.append(upload_energy)
    #         # print("group_local_latency", group_local_latency)
    #     computation_capability = [0 for _ in range(len(self.group.G))]
    #     communication_capability = [0 for _ in range(len(self.group.G))]
    #     sampling_probability = [0 for _ in range(len(self.group.G))]
    #     for i in range(len(self.group.G)):
    #         computation_capability[i] = 1 / (0.5 * group_local_latency[i] / max(group_local_latency) + (1 - 0.5) * group_local_energy[i] / max(group_local_energy))
    #         communication_capability[i] =  1 / (0.5 * group_upload_latency[i] / max(group_upload_latency) + (1 - 0.5) * group_upload_energy[i] / max(group_upload_energy))

    #     for i in range(len(self.group.G)):
    #         sampling_probability[i] = 0.5 * computation_capability[i] + (1 - 0.5) * communication_capability[i]
    #     sampling_probability = [sampling_probability[i] / sum(sampling_probability) for i in range(len(sampling_probability))]
    #     # print(sampling_probability)
    #     return sampling_probability
    # ----------------------------------------------------------- #

    # Gpu
    def get_sampling_probability(self, round_i):
        group_local_latency = []
        group_local_energy = []
        group_upload_latency = []
        group_upload_energy = []
        for group in self.group.G:
            local_latency_temp = 0
            upload_latency_temp = 0
            local_energy  = 0
            upload_energy = 0
            for client in self.group.G[group]:
                if self.clients[client].getLocalDelay(round_i) > local_latency_temp:
                    local_latency_temp = self.clients[client].getLocalDelay(round_i)
                if self.clients[client].getUploadDelay(round_i) > upload_latency_temp:
                    upload_latency_temp = self.clients[client].getUploadDelay(round_i)
                local_energy = self.clients[client].getLocalEnergy(round_i)
                upload_energy = self.clients[client].getUploadEnergy(round_i)

            group_local_latency.append(local_latency_temp * 10)
            group_local_energy.append(local_energy )
            group_upload_latency.append(upload_latency_temp)
            group_upload_energy.append(upload_energy * 10)
        # print("group_local_latency", group_local_latency)
        # print("group_local_energy", group_local_energy)
        # print("group_upload_latency", group_upload_latency)
        # print("group_upload_energy", group_upload_energy)
        group_local_latency = torch.tensor(group_local_latency, dtype=torch.float32).cuda()
        group_local_energy = torch.tensor(group_local_energy, dtype=torch.float32).cuda()
        group_upload_latency = torch.tensor(group_upload_latency, dtype=torch.float32).cuda()
        group_upload_energy = torch.tensor(group_upload_energy, dtype=torch.float32).cuda()

        computation_capability = torch.zeros(len(self.group.G), dtype=torch.float32).cuda()
        communication_capability = torch.zeros(len(self.group.G), dtype=torch.float32).cuda()
        # sampling_probability = torch.zeros(len(self.group.G)).cuda()

        max_local_latency = torch.max(group_local_latency)
        max_local_energy = torch.max(group_local_energy)
        max_upload_latency = torch.max(group_upload_latency)
        max_upload_energy = torch.max(group_upload_energy)
        for i in range(len(self.group.G)):
            computation_capability[i] = 1 / (0.5 * group_local_latency[i] / max_local_latency + (1 - 0.5) * group_local_energy[i] / max_local_energy)
            communication_capability[i] = 1 / (0.5 * group_upload_latency[i] / max_upload_latency + (1 - 0.5) * group_upload_energy[i] / max_upload_energy)
        
        # 归一化 
        computation_capability_N  = torch.zeros(len(self.group.G), dtype=torch.float32).cuda()
        communication_capability_N = torch.zeros(len(self.group.G), dtype=torch.float32).cuda()

        for i in range(len(self.group.G)):
            computation_capability_N[i] = computation_capability[i] / torch.sum(computation_capability)
            communication_capability_N[i] = communication_capability[i] / torch.sum(communication_capability)

        sampling_probability = [0 for _ in range(len(self.group.G))]
        for i in range(len(self.group.G)):
            sampling_probability[i] = (0.5 * computation_capability_N[i].item() + (1 - 0.5) * communication_capability_N[i].item())
        sampling_probability_adjusted = sampling_probability.copy()
        sampling_probability_adjusted[-1] += 1 - sum(sampling_probability)
        return sampling_probability_adjusted
    

    def get_energy_sampling_probability(self, round_i, groups_B):
        group_latency = []
        group_energy = []
        for group in self.group.G:
            latency_temp = 0
            energy = 0
            for client in self.group.G[group]:
                if self.clients[client].getLocalDelay(round_i) + self.clients[client].getUploadDelay(round_i) > latency_temp:
                    latency_temp = self.clients[client].getLocalDelay(round_i) + self.clients[client].getUploadDelay(round_i)
                energy += self.clients[client].getLocalEnergy(round_i) + self.clients[client].getUploadEnergy(round_i)
                 
            group_latency.append(latency_temp)
            group_energy.append(energy)

        group_latency = torch.tensor(group_latency, dtype=torch.float32).cuda()
        group_energy = torch.tensor(group_energy, dtype=torch.float32).cuda()

        latency_capability = torch.zeros(len(self.group.G), dtype=torch.float32).cuda()
        energy_capability = torch.zeros(len(self.group.G), dtype=torch.float32).cuda()
        data_equlity_capability =  torch.zeros(len(self.group.G), dtype=torch.float32).cuda()
        # sampling_probability = torch.zeros(len(self.group.G)).cuda()

        # print("group_latency", group_latency)
        # print("group_energy", group_energy)
        max_latency = torch.max(group_latency)
        max_energy = torch.max(group_energy)
        for i in range(len(self.group.G)):
            latency_capability[i] = 1 / (group_latency[i] / max_latency)
            energy_capability[i] = 1 / (group_energy[i] / max_energy)
            data_equlity_capability[i] = 1 / groups_B[i]
        # print(latency_capability)
        # print(energy_capability)
        # 归一化 
        latency_capability_N  = torch.zeros(len(self.group.G), dtype=torch.float32).cuda()
        energy_capability_N = torch.zeros(len(self.group.G), dtype=torch.float32).cuda()
        data_equlity_capability_N = torch.zeros(len(self.group.G), dtype=torch.float32).cuda()

        for i in range(len(self.group.G)):
            latency_capability_N[i] = latency_capability[i] / torch.sum(latency_capability)
            energy_capability_N[i] =  energy_capability[i] / torch.sum(energy_capability)
            data_equlity_capability_N[i] = data_equlity_capability[i] / sum(data_equlity_capability)
        # print(latency_capability_N)
        # print(energy_capability_N)
        sampling_probability = [0 for _ in range(len(self.group.G))]
        for i in range(len(self.group.G)):
            sampling_probability[i] = ((1 - self.options['omega3']) * self.options['I'] * latency_capability_N[i].item() + (1 - self.options['omega3']) * (1 - self.options['I']) * energy_capability_N[i].item() + self.options['omega3'] * data_equlity_capability_N[i].item())
        sampling_probability_adjusted = sampling_probability.copy() 
        sampling_probability_adjusted[-1] += 1 - sum(sampling_probability)
        print(sampling_probability_adjusted)
        return sampling_probability_adjusted
    

 