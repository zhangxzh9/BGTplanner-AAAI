import torch
import numpy as np
from random import sample
from utils.model import model
from scipy.sparse.linalg import svds 
torch.multiprocessing.set_sharing_strategy('file_system')


class SERVER():
    def __init__(self, user_list, user_batch, users, items, embed_size, lr, users_feature,args):
        self.all_user_list = user_list  # 所有的users
        self.generate_train_user_list()  # 有train item的users
        self.batch_size = user_batch
        self.users_feature = users_feature
        self.user_embedding = torch.randn(len(users), embed_size).share_memory_()
        self.item_embedding = torch.randn(len(items), embed_size).share_memory_()
        self.item_list = list(items)
        self.model = model(embed_size, 1)
        self.lr = lr
        self.args = args
        self.distribute(self.all_user_list)

    # def users_feature_to_embedding(self, users_feature):

    def generate_train_user_list(self):
        self.train_user_list = []
        for user in self.all_user_list:
            if len(user.items) > 0:
                self.train_user_list.append(user)
        return self.train_user_list

    def aggregator(self, parameter_list):
        flag = False
        number = 0
        gradient_item = torch.zeros_like(self.item_embedding)
        gradient_user = torch.zeros_like(self.user_embedding)
        for parameter in parameter_list:
            [model_grad, item_grad, user_grad, returned_items, returned_users] = parameter
            num = len(returned_items)
            number += num
            if not flag:
                flag = True
                gradient_model = []
                gradient_item[returned_items, :] += item_grad * num
                # gradient_user[returned_users, :] += user_grad * num
                gradient_user[returned_users, :] += user_grad
                for i in range(len(model_grad)):
                    gradient_model.append(model_grad[i] * num)
            else:
                gradient_item[returned_items, :] += item_grad * num
                # gradient_user[returned_users, :] += user_grad * num
                gradient_user[returned_users, :] += user_grad
                for i in range(len(model_grad)):
                    gradient_model[i] += model_grad[i] * num
        gradient_item /= number
        # gradient_user /= number
        for i in range(len(gradient_model)):
            gradient_model[i] = gradient_model[i] / number
        return gradient_model, gradient_item, gradient_user

    def distribute(self, users):
        for user in users:
            user.update_local_GNN(self.model)

    def update_adap_delta(self,beta,delta):
        for user in self.train_user_list:
            user.delta = beta * delta
        return beta * delta


    def distribute_one(self, user):
        user.update_local_GNN(self.model)

    def predict(self, valid_data, top_k):
        # print('predict')
        valid_user = valid_data[:, 0].astype(int)
        valid_items = valid_data[:, 1].astype(int)
        valid_label = valid_data[:, -1]
        
        predicted_ratings = np.zeros(valid_label.shape)

        self.distribute([self.all_user_list[u] for u in set(valid_user)])

        # hit =  np.zeros_like(top_k,dtype=np.float64)
        # recall_deno = np.zeros_like(top_k,dtype=np.float64)
        # precision_deno =np.zeros_like(top_k,dtype=np.float64)
        TP = 0
        FP = 0
        FN = 0
        for user_id in set(valid_user):
            selected_indices = np.where(valid_user == user_id)
            user_items_indices = valid_items[selected_indices]
            user_label_ratings = valid_label[selected_indices]

            #一次性预测user所有items的分数
            user_predicted_scores=self.all_user_list[user_id].predict_all(user_items_indices,self.item_embedding)
            #将预测结果对应放置到predicted_ratings数组中
            predicted_ratings[selected_indices] = user_predicted_scores

            # 将标签评分进行二值化，3及其以上为1，其余为0
            binary_label_ratings = np.where(user_label_ratings >= 0.5, 1, 0)

            # 将模型预测分数进行二值化，2.5以上为1，其余为0
            binary_predicted_scores = np.where(user_predicted_scores >= 0.5, 1, 0)  
            
            # 计算 TP, FP, FN
            TP += np.sum((binary_predicted_scores == 1) & (binary_label_ratings == 1))
            FP += np.sum((binary_predicted_scores == 1) & (binary_label_ratings == 0))
            FN += np.sum((binary_predicted_scores == 0) & (binary_label_ratings == 1))
        
        # 计算指标
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            # 计算指标
            # relevant_items = set(user_items_indices)
            # sorted_indices = np.argsort(user_predicted_scores)[::-1]
            # sorted_items = [x for _, x in sorted(zip(user_predicted_scores, user_items_indices), reverse=True)]

            # for k in range(len(top_k)):
            #     top_k_items = set(sorted_indices[:top_k[k]])
            #     hit[k] += len(relevant_items.intersection(top_k_items)) 
            #     recall_deno[k] += len(relevant_items)
            #     precision_deno[k] += len(top_k_items)

        # recall = hit / recall_deno
        # precision = hit / precision_deno
        # f_1_score = (2 * recall * precision) / (recall + precision)

        mae = sum(abs(valid_label - predicted_ratings)) / len(valid_label) if len(valid_label) > 0 else 0
        rmse = np.sqrt(np.sum((valid_label - predicted_ratings) ** 2) / len(valid_label)) if len(valid_label) > 0 else 0

        return mae, rmse, f1_score, precision, recall
        # for i in range(len(users)):
        #     res_temp = self.all_user_list[users[i]].predict(items[i], self.item_embedding)
        #     res.append(float(res_temp))

    def receive_context(self):
        # item =len(self.item_list)
        user_item_matrix  = np.zeros((len(self.train_user_list),len(self.item_list)),dtype="int8")
        users = sample(self.train_user_list, len(self.train_user_list))
        for user in range(len(self.train_user_list)):
            item_ids = users[user].upload_item_ids()
            for item in item_ids:
                user_item_matrix[user][item] = 1
        k=5
        user_item_matrix = user_item_matrix.astype('float64')
        U, S, Vt = svds(user_item_matrix, k=k)
        num_items = Vt.shape[1]  # 物品数
        context =  np.zeros(k * num_items)
        for i in range(num_items):
            item_vector = Vt[:, i]
            context[i * k:(i + 1) * k] = item_vector
        # return context

        return S



    def train(self, round,action=None):
        parameter_list = []
        loss_list = []
        users = sample(self.train_user_list, len(self.train_user_list))
        # print('distribute')
        self.distribute(users)


        for user in users:
            parameter,loss = user.train(self.user_embedding, self.item_embedding,round+1,action)
            # parameter,loss = user.train_bpr(self.user_embedding, self.item_embedding)
            parameter_list.append(parameter)
            loss_list.append(loss)

        # print('aggregate')
        gradient_model, gradient_item, gradient_user = self.aggregator(parameter_list)

        ls_model_param = list(self.model.parameters())

        # print('Mean loss:',np.mean(loss_list))
        model_optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=0.0001)
        item_optimizer = torch.optim.Adam([self.item_embedding],lr= self.lr,weight_decay=0.0001)
        user_optimizer = torch.optim.Adam([self.user_embedding],lr= self.lr,weight_decay=0.0001)

        for i,param in enumerate(list(self.model.parameters())):
            # grad = self.LDP(param.grad)
            param.requires_grad= True
            param.grad = gradient_model[i]
        model_optimizer.step()
        # print('renew')

        # for i in range(len(ls_model_param)):

        #     ls_model_param[i].data = ls_model_param[i].data - self.lr * gradient_model[i]

        # self.item_embedding = self.item_embedding -  self.lr * gradient_item
        self.item_embedding.requires_grad = True
        self.user_embedding.requires_grad = True
        self.item_embedding.grad = gradient_item
        self.user_embedding.grad = gradient_user

        item_optimizer.step()
        user_optimizer.step()
        # if round%10==0:
        # if round<10:
        # self.user_embedding = self.user_embedding -  self.lr * gradient_user
        # self.user_embedding = self.user_embedding -  0.001 * gradient_user
        # return np.mean(loss_list)
