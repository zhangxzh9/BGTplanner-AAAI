import torch
import copy
import random
import numpy as np
import dgl
from utils.model import model

class user():
    def __init__(self, id_self, raw_feature, items, ratings, neighbors, online_data_user, embed_size, clip, total_budget, negative_sample, selected_item,all_items_num,args,action_map_epsilon=None):
        self.negative_sample = negative_sample # 虚假item总数量？
        self.clip = clip
        self.total_budget = total_budget
        self.id_self = id_self
        self.items = items # 实际交互的item？
        self.embed_size = embed_size
        self.ratings = ratings
        self.neighbors = neighbors
        self.model = model(embed_size, 1)
        self.graph = self.build_local_graph(self.id_self, self.items, self.neighbors)
        self.graph = dgl.add_self_loop(self.graph)
        # self.epsilon = 1
        self.raw_feature = raw_feature
        self.feature_embedding = torch.tensor(self.raw_feature,dtype=torch.float32)
        # self.self_embedding = self.user_feature_to_embedding(self,user_feature)
        self.user_feature = torch.randn(self.embed_size)
        self.selected_item = selected_item
        self.all_items_num = all_items_num # 总共的item数量
        self.negative_sample_items = self.sample_pseudo_items(negative_sample) #虚假交互的item list
        self.args = args
        self.action_map_epsilon = action_map_epsilon
        self.remaining_budget = total_budget
        self.model_grad_history =[]
        # self.adap_sigma = 2
        self.alpha = 1
        self.online_data = online_data_user

        # 计算每轮需要增加的新样本数量
        if len(self.online_data) == 0:
            self.samples_per_sample_or_round = 0
            self.extra_samples_or_rounds  = 0
            self.online_flag = False
        elif len(self.online_data) >= self.args.T:
            self.samples_per_sample_or_round = len(self.online_data) // self.args.T
            self.extra_samples_or_rounds = len(self.online_data) % self.args.T
            self.online_flag = True
        else:
            self.samples_per_sample_or_round = self.args.T // len(self.online_data)
            self.extra_samples_or_rounds = self.args.T % len(self.online_data)
            self.online_flag = True

        # 初始化一个未被选中的样本index集合
        self.remaining_samples = list(range(len(self.online_data)))

    def sample_pseudo_items(self,negative_sample): #
        negative_items = []
        while len(negative_items) < negative_sample:
            item = random.randint(0, self.all_items_num-1)
            if item not in self.items and item not in negative_items:
                negative_items.append(item)
        return negative_items


    def upload_item_ids(self): #上传sample的item
        return self.items+self.negative_sample_items

    def update_training_graph_and_data(self,round):
        # 确定当前轮次需要增加的样本数量
        if self.online_flag:
            if len(self.online_data) >= self.args.T:
                num_samples_to_add = self.samples_per_sample_or_round + (1 if round < self.extra_samples_or_rounds else 0)
            elif round < self.args.T - self.extra_samples_or_rounds:
                num_samples_to_add = 1 if (round) % self.samples_per_sample_or_round == 0 else 0
            else:
                num_samples_to_add = 0

            # 从未被选中的样本中随机选择 num_samples_to_add 个样本
            new_samples = random.sample( self.remaining_samples, num_samples_to_add)

            # 将选择的新样本加入到训练数据中，并从未被选中的online data中移除
            for idx in new_samples:
                item, rating, _ = self.online_data[idx]
                self.items.append(item)
                self.ratings.append(rating)
                self.remaining_samples.remove(idx)

            # update local graph
            self.graph = self.build_local_graph(self.id_self, self.items, self.neighbors) 
            self.graph = dgl.add_self_loop(self.graph)

        # update pseudo items
        self.negative_sample_items = self.sample_pseudo_items(self.negative_sample)

    def build_local_graph(self, id_self, items, neighbors):
        G = dgl.DGLGraph()
        dic_user = {self.id_self: 0}
        dic_item = {}
        count = 1
        for n in neighbors:
            dic_user[n] =  count
            count += 1
        for item in items:
            dic_item[item] = count
            count += 1
        G.add_edges([i for i in range(1, len(dic_user))], 0)
        G.add_edges(list(dic_item.values()), 0)
        G.add_edges(0, 0)
        return G

    def get_user_embedding(self, embedding):
        if len(self.neighbors)==0:
            return embedding[torch.tensor(self.id_self)]
        else:
            return embedding[torch.tensor(self.neighbors)], embedding[torch.tensor(self.id_self)]

    def get_item_embedding(self, embedding):
        return embedding[torch.tensor(self.items)]

    def GNN(self, embedding_user, embedding_item):
        if len(self.neighbors)==0:
            self_embedding = self.get_user_embedding(embedding_user)
            items_embedding = self.get_item_embedding(embedding_item)

            # features =  torch.cat((self.feature_embedding.unsqueeze(0),  items_embedding), 0)
            features =  torch.cat((self_embedding.unsqueeze(0),  items_embedding), 0)

        else:
            neighbor_embedding, self_embedding = self.get_user_embedding(embedding_user)
            items_embedding = self.get_user_embedding(embedding_item)

            # features =  torch.cat((self.feature_embedding.unsqueeze(0), neighbor_embedding, items_embedding), 0)
            features =  torch.cat((self_embedding.unsqueeze(0), neighbor_embedding, items_embedding), 0)

        user_feature = self.model(self.graph, features, len(self.neighbors) + 1)
        self.user_feature = user_feature.detach()

        # predicted = torch.matmul(user_feature, items_embedding.t()) #只计算交互过的item的预测结果
        # predicted = torch.matmul(user_feature, embedding_item.t()) #计算所有item的预测结果
        if len(self.negative_sample_items)!=0:
            sampled_items = self.negative_sample_items
            sampled_items_embedding = embedding_item[torch.tensor(sampled_items)]
            items_embedding_with_sampled = torch.cat((items_embedding, sampled_items_embedding), dim = 0)
            predicted = torch.matmul(user_feature, items_embedding_with_sampled.t())#计算所有item的预测结果
        else:
            predicted = torch.matmul(user_feature, items_embedding.t())#计算真实的交互item的预测结果
        return torch.sigmoid(predicted) #* 5

    def update_local_GNN(self, global_model):
        self.model = copy.deepcopy(global_model)

    def loss_with_nosample(self, predicted):
        return torch.sqrt(torch.mean((predicted - torch.tensor(self.ratings))**2))
    
    def predict(self, item_id, embedding_item): #主函数测试用到了
        item_embedding = embedding_item[item_id]
        predicted_score = torch.sigmoid(torch.matmul(self.user_feature, item_embedding.t())) # * 5
        return predicted_score.detach().numpy()


    def predict_all(self, item_ids, embedding_item):
        selected_items_embedding = embedding_item[item_ids]
        predicted_scores = torch.sigmoid(torch.matmul(self.user_feature, selected_items_embedding.t()))  # * 5
        return predicted_scores.detach().numpy()

    def add_noise_to_pusedo_item_embedding(self, grad):
        grad_value = torch.masked_select(grad, grad != 0)
        mean = torch.mean(grad_value)
        var = torch.std(grad_value)
        if self.negative_sample==0:
            pass
        else:
            grad[torch.tensor(self.negative_sample_items)] += torch.randn((len(self.negative_sample_items), self.embed_size)) * var + mean
        return grad

    def LDP(self, tensor,round,action=None):
        if self.args.allocation == 'BGTplanner':
            epsilon_single_query = self.action_map_epsilon[action]
        #     if self.remaining_budget >= self.action_map_epsilon[action]:
        #         epsilon_single_query = self.action_map_epsilon[action]
        #     else:
        #         epsilon_single_query = self.remaining_budget
                # self.remaining_budget = 0
        else:
            raise ValueError("no such allocation {}".format(self.args.allocation))

        proportion = len(self.items+self.negative_sample_items) / self.all_items_num
        sensitivity = 2 * self.clip 
        if self.args.dp_mechnisim == 'Laplace':
            tensor /= max(1,torch.norm(tensor,p=1)/self.clip)
            # tensor = torch.clamp(tensor, min=-self.clip, max=self.clip)
            # scale = np.log( ( np.exp( epsilon_single_query ) - proportion ) / ( 1 - proportion ) )
            noise_scale = sensitivity / epsilon_single_query
            tensor += torch.from_numpy(np.random.laplace(0, scale= noise_scale, size = tensor.shape))
        elif self.args.dp_mechnisim == 'Gaussian':
            tensor /= max(1,torch.norm(tensor,p=2)/self.clip)
            noise_scale = np.sqrt((sensitivity**2 * self.args.rdp_alpha) / (2 * epsilon_single_query))
            tensor += torch.from_numpy(np.random.normal(0, scale=noise_scale,size = tensor.shape))
        elif self.args.dp_mechnisim == 'Gaussian_basic':
            tensor /= max(1,torch.norm(tensor,p=2)/self.clip)
            scale = np.sqrt( 2 * np.log( ( 1.25 * ( 1 - proportion ) ) / self.args.dp_delta ) ) / np.log( ( np.exp( epsilon_single_query ) - proportion ) / ( 1 - proportion ) ) 
            noise_scale = sensitivity * scale
            tensor += torch.from_numpy(np.random.normal(0, scale=noise_scale,size = tensor.shape))
        else:
            raise ValueError("no such dp mechanism {}".format(self.args.dp_mechnisim))
        # tensor /= max(1,torch.norm(tensor,p=1)/self.clip)
        # sensitivity = 2*self.clip
        # noise_scale = sensitivity /self.total_budget
        # tensor += torch.from_numpy(np.random.laplace(0,scale=noise_scale,size = tensor.shape))
        return tensor

    def train(self, embedding_user, embedding_item,round,action=None):
        embedding_user = torch.clone(embedding_user).detach()
        embedding_item = torch.clone(embedding_item).detach()
        embedding_user.requires_grad = True
        embedding_item.requires_grad = True
        self.feature_embedding.requires_grad = True
        if len(self.negative_sample_items)!=0:
            predicted = self.GNN(embedding_user, embedding_item)
            loss = self.loss_with_nosample(predicted[:len(self.items)])
        else:
            predicted = self.GNN(embedding_user, embedding_item)
            loss = self.loss_with_nosample(predicted)
        
        self.model.zero_grad()
        loss.backward()

        # Add noise to model parameter grads
        model_grad = []
        origin_grad = []
        for index,param in enumerate(list(self.model.parameters())):
            grad = self.LDP(param.grad,round,action) 
            model_grad.append(grad)

        # # 获取模型的所有梯度并拼接成一个向量
        # for param in self.model.parameters():
        #     origin_grad.append(param.grad.view(-1))
        # all_grads = torch.cat(origin_grad)

        # # 对整个梯度向量加噪声
        # noisy_grads_vector = self.LDP(all_grads, round, action)

        # # 将加噪后的梯度重新分配回各个参数
        # offset = 0
        # for param in self.model.parameters():
        #     param_grad_shape = param.grad.shape
        #     param_grad_size = param.grad.numel()
            
        #     # 取出对应的加噪梯度
        #     model_grad.append(noisy_grads_vector[offset:offset + param_grad_size].view(param_grad_shape))
            
        #     # 更新偏移量
        #     offset += param_grad_size


        #Add noise to item embedding grads
        embedding_item.grad = self.add_noise_to_pusedo_item_embedding(embedding_item.grad)
        returned_items = self.items + self.negative_sample_items 
        item_grad = []
        for i in returned_items:
            item = embedding_item.grad[i, :]
            if i in self.items:
                grad= self.LDP(item,round,action)
            else:
                grad = item
            item_grad.append(grad)
        item_grad = torch.stack(item_grad)

        # It can be trained in local. To ensure the consistency of the performance, we don't add noise.
        returned_users = self.neighbors + [self.id_self]
        user_grad = embedding_user.grad[returned_users, :] 

        res = (model_grad, item_grad, user_grad, returned_items, returned_users)
        return res , loss.detach()
