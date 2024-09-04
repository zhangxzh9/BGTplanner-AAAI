import pickle
import numpy as np
from utils.user import user
from utils.Server import SERVER
import math
from utils.logger import *
import logging
import os
import time
from utils.GaussianProcess import GaussianProcessRegressor
from utils.squarecbwk import *
from utils.linear_programming import solve_linear_programming
from utils.set_seed import set_seed
import gc

def processing_valid_data(valid_data,args):
    res = []
    for key in valid_data.keys():
        if len(valid_data[key]) > 0:
            for ratings in valid_data[key]:
                item, rate, _ = ratings
                if args.data=="ciao":
                    res.append((int(key), int(item), rate))
                else:
                    res.append((int(key), int(item), rate))
    return np.array(res)

def loss(server, valid_data, top_k):
    # group_rmse = 
    # ungroup_rmse = 0
    # label = valid_data[:, -1]W
    # mae = sum(abs(label - predicted)) / len(label)
    # rmse = math.sqrt(sum((label - predicted) ** 2) / len(label))

    return server.predict(valid_data,top_k)

def test(server,valid_data):
    user_rmse, hr_top1, hr_top5, hr_top10  = server.test(valid_data)
    values = user_rmse.values()
    rmse_2 = sum(values) / len(values)
    return rmse_2, hr_top1, hr_top5, hr_top10

def main_bgt(args):
    dp_original_budget = args.total_budget
    # if args.dp_mechnisim == "Gaussian":
    #     args.total_budget = args.total_budget - np.log(1 / (args.dp_delta * args.T)) / (args.rdp_alpha - 1)
    # elif args.dp_mechnisim == 'Laplace' or "Gaussian_basic":
    #     args.total_budget = args.total_budget
    # else:
    #     raise ValueError("no such dp mechanism {}".format(args.dp_mechnisim))
    log_path_name = ('./logs/strategy_{}_epsilon_{}_T_{}_DP_{}_dataset_{}_userbatch_{}_nega_{}_T_MIN_{}.log'.format(args.allocation,dp_original_budget,args.T,args.dp_mechnisim,args.data,args.user_batch,args.negative_sample,args.min_training_rounds_frac))
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    logger = LoggerCreator.create_logger(log_path = log_path_name,  level=logging.INFO)
    logger.info(' '.join(f' \'{k}\': {v}, ' for k, v in vars(args).items()))

    # read data
    if args.data == 'filmtrust' or args.data == 'ciao' or args.data == 'epinions':
        data_file = open('./data/' + args.data + '_FedMF.pkl', 'rb')
        users_feature = {}
        [train_data, valid_data, online_data, user_id_list, item_id_list, social] = pickle.load(data_file)
        for u in user_id_list:
            users_feature[u] = np.array([1])
        data_file.close()
    elif args.data in ['ml-1m','ml-100k','ml-1m-605','ml-100k_online','ml-1m-605_online']:
        data_file = open('./data/' + args.data + '.pkl', 'rb')
        [train_data, valid_data, online_data, user_id_list, item_id_list, social, users_feature] = pickle.load(data_file)
        data_file.close()

    valid_data = processing_valid_data(valid_data,args)
    # test_data = processing_valid_data(test_data,args)
    top_k = args.top_k #[1,3,5,10,20,25,50] #top-1-20

    for run_times in range(args.total_run_times):
        
        seed = 123 + run_times  
        set_seed(seed)
        logger.info(f"Run times {run_times} with seed {seed}:")
        # logger.info(f"Random number from NumPy:{np.random.rand()}")
        # logger.info(f"Random number from PyTorch:{torch.randn(1)}")

        # 格式化日期和时间
        save_path_name = ('./results/strategy_{}_epsilon_{}_T_{}_DP_{}_dataset_{}_userbatch_{}_nega_{}_T_MIN_{}_run_times_{}.json'.
                        format(args.allocation,dp_original_budget,args.T,args.dp_mechnisim,args.data,args.user_batch,args.negative_sample,args.min_training_rounds_frac,run_times))
        if not os.path.exists('./results'):
            os.makedirs('./results')

        # build user_list
        K = 5
        action_map_epsilon = {}
        eps_min = args.total_budget /  args.T  
        min_training_rounds_frac = args.min_training_rounds_frac # min_training_rounds_frac = 4 / 5
        eps_max = args.total_budget /  (min_training_rounds_frac * args.T)
        eps_interval = (eps_max - eps_min) / K
        for i in range(0,K):
            action_map_epsilon[int(i)] = eps_min + eps_interval * i
        logger.info("action_map_epsilon: {}".format(action_map_epsilon))

        d = args.user_batch
        cost_value = np.zeros((d, K))
        for i in range(d):
            cost_value[i,: ] = np.array([ action_map_epsilon[key] for key in action_map_epsilon.keys()])
        # print(cost_value[0,:],cost_value[1,:])

        # for i in range(d):
        #     cost_value[i,: ] = np.arange(0.25, 1.5, 0.25)

        gpr = GaussianProcessRegressor()
        m=d+2
        T=args.T
        T_0 = 5
        Z = args.T/args.total_budget
        B=args.total_budget
        Resource = args.total_budget
        gamma = 2 * np.sqrt(K * T / (m*np.log(T)))
        lamb = np.ones((d+1,1))/(d+1) # numpy array with size (d,1) 
        step_size = 1.5
        dual_step_size = 2 * np.log(d) / np.sqrt(T)

        user_list = []
        for u in user_id_list:
            ratings = train_data[u]
            items = []
            rating = []
            for i in range(len(ratings)):
                item, rate, _  = ratings[i]
                items.append(item)
                rating.append(rate)

            user_list.append(user(u, [], items, rating, [], online_data[u], args.embed_size, args.clip, args.total_budget, args.negative_sample,args.selected_item,len(item_id_list),args,action_map_epsilon))
        
        # build server
        server = SERVER(user_list, args.user_batch, user_id_list, item_id_list, args.embed_size, args.lr, users_feature,args)

        # train and evaluate
        start_time = time.time()
        training_time_list = []
        rmse_list = []
        f1_score_list = []
        precision_list = []
        recall_list = []
        x =[]

        x.append(0)
        mae, rmse, f1_score, precision, recall= loss(server, valid_data, top_k)
        rmse_list.append(rmse)

        his_records = []

        iter_t=0

        gc.collect()
        # while 1:
        while iter_t < args.T:

            if Resource <= 0:             
                break
            f_0 = []
            while iter_t <  K* T_0-1:
                if Resource <= 0:
                    break
                for a in range(K-1,-1,-1):
                    for t in range(T_0):
                        logger.info('Round:{}'.format(iter_t))
                        for u in user_list:
                            u.update_training_graph_and_data(iter_t)
                        server.generate_train_user_list()
                        context = server.receive_context()

                        server.train(round=iter_t,action=a)

                        # 计算reward
                        mae, rmse, f1_score, precision, recall= loss(server, valid_data, top_k)
                        
                        if np.isnan(rmse):
                            rmse = rmse_list[-1]  
                        rmse_list.append(rmse)

                        reward = rmse_list[-2] - rmse_list[-1]
                        his_records.append((a,context,reward))
                        Resource -= action_map_epsilon[a]

                        f1_score_list.append(f1_score)
                        precision_list.append(precision)
                        recall_list.append(recall)
                        logger.info('Round:{}, stage:1, action:{}, eps: {}, mae: {}, rmse: {}'.format(iter_t, a, action_map_epsilon[a], mae, rmse))
                        # logger.info('Round:{}, top_k:     {}'.format(iter_t, top_k))
                        logger.info('Round:{}, stage:1, f1_score: {}, precision: {}, recall: {}'.format(iter_t, f1_score, precision, recall))
                        # logger.info('Round:{}'.format(iter_t, precision))
                        # logger.info('Round:{}'.format(iter_t, recall))
                        
                        now_time = time.time()
                        elapsed_time = now_time - start_time
                        training_time_list.append(elapsed_time)
                        iter_t+=1
                        x.append(iter_t) # 迭代轮次列表？
                        
                        reward_value = gp_ucb(gpr,his_records,epoch=iter_t,context=context)
                        # for i in range(len(reward_value)):
                        f_0.append(reward_value[a])
                        logger.info('Round:{}, stage:1, reward:{}, predict_reawrd: {}'.format(iter_t, reward, reward_value[a]))

                # reward_value = gp_ucb(gpr,his_records,epoch=iter_t,context=context)
                # for i in range(len(reward_value)):
                #     f_0.append(reward_value[i])

            error = 0
            while iter_t < (K+1) * T_0 and iter_t >= K* T_0:
                # 在(K)*T0轮次之后，随机选择a进行训练
                if Resource <= 0:
                    break
                logger.info('Round:{}'.format(iter_t))
                a = np.random.choice(K)
                for u in user_list:
                    u.update_training_graph_and_data(iter_t)
                server.generate_train_user_list()
                context = server.receive_context()

                server.train(round=iter_t,action=a)
                mae, rmse, f1_score, precision, recall= loss(server, valid_data, top_k)
                if np.isnan(rmse):
                    rmse = rmse_list[-1]  
                rmse_list.append(rmse)

                reward = rmse_list[-2] - rmse_list[-1]
                his_records.append((a,context,reward))

                reward_value = gp_ucb(gpr,his_records,epoch=iter_t,context=context)
                error += pow(reward-reward_value[a],2)
                if iter_t == (K+1) * T_0 -1:
                    M = math.sqrt( K*(error / T_0) + 4* math.log(T*d)/T_0)
                    print('predict_reward:',f_0)
                    OPT  = solve_linear_programming(T_0, K, f_0, B, d, T, M, action_map_epsilon)                     
                    Z = T/B * (OPT + M)
                Resource -= action_map_epsilon[a]

                f1_score_list.append(f1_score)
                precision_list.append(precision)
                recall_list.append(recall)
                # logger.info('Round:{}, stage:2, action:{}, mae: {}, rmse: {}'.format(iter_t, a, mae, rmse))
                logger.info('Round:{}, stage:2, action:{}, eps: {}, mae: {}, rmse: {}'.format(iter_t, a, action_map_epsilon[a], mae, rmse))
                # logger.info('Round:{}, top_k:     {}'.format(iter_t, top_k))
                logger.info('Round:{}, stage:2, f1_score: {}, precision: {}, recall: {}'.format(iter_t, f1_score, precision, recall))
                # logger.info('Round:{}'.format(iter_t, precision))
                # logger.info('Round:{}'.format(iter_t, recall))
                logger.info('Round:{}, stage:2, reward:{}, predict_reawrd: {}'.format(iter_t, reward, reward_value[a]))
                
                now_time = time.time()
                elapsed_time = now_time - start_time
                training_time_list.append(elapsed_time)
                iter_t+=1
                x.append(iter_t) # 迭代轮次列表？

            B_2 = Resource
            while iter_t >=(K+1) * T_0 and iter_t < args.T:
                logger.info('Round:{}'.format(iter_t))
                if Resource <= 0:
                    break
                for u in user_list:
                    u.update_training_graph_and_data(iter_t)
                server.generate_train_user_list()
                context = server.receive_context()

                reward_value = gp_ucb(gpr,his_records,epoch=iter_t,context=context)
                reward_value = np.array(reward_value)
                reward_value = reward_value[None,:]
                a = np.argmax(Lagrangian_sampling(reward_value.T, cost_value.T, Z, lamb[0:d,:], gamma,K,B_2,(T-(K+1)*T_0)))

                server.train(round=iter_t,action=a)

                mae, rmse, f1_score, precision, recall= loss(server, valid_data, top_k)
                if np.isnan(rmse):
                    mae, rmse, f1_score, precision, recall= loss(server, valid_data, top_k)
                    rmse = rmse_list[-1]  
                rmse_list.append(rmse)

                reward = rmse_list[-2] - rmse_list[-1]
                Resource -= action_map_epsilon[a]

                dual_grad = ( B_2/(T-(K+1)*T_0) * np.ones((d,1)) - cost_value[:,a].reshape(-1,1))
                lamb = OMD_update(lamb, dual_grad, dual_step_size, d)
                his_records.append((a,context,reward))
                
                f1_score_list.append(f1_score)
                precision_list.append(precision)
                recall_list.append(recall)
                # logger.info('Round:{}, stage:3, action:{}, mae: {}, rmse: {}'.format(iter_t, a, mae, rmse))
                logger.info('Round:{}, stage:3, action:{}, eps: {}, mae: {}, rmse: {}'.format(iter_t, a, action_map_epsilon[a], mae, rmse))
                # logger.info('Round:{}, top_k:     {}'.format(iter_t, top_k))
                logger.info('Round:{}, stage:3, f1_score: {}, precision: {}, recall: {}'.format(iter_t, f1_score, precision, recall))
                logger.info('Round:{}, stage:3, reward:{}, predict_reawrd: {}'.format(iter_t, reward, reward_value[0][a]))
                # logger.info('Round:{}'.format(iter_t, precision))
                # logger.info('Round:{}'.format(iter_t, recall))
                
                now_time = time.time()
                elapsed_time = now_time - start_time
                training_time_list.append(elapsed_time)
                iter_t+=1
                x.append(iter_t) # 迭代轮次列表？

        logger.info('run_times {} finish'.format(run_times))
        logger.info('run_times {}, min rmse: {}'.format(run_times, min(rmse_list) ))
        logger.info('run_times {}, f1_score: {}, precision: {}, recall: {}'.format(run_times, max(f1_score_list), max(precision_list), max(recall_list)))
        data = {'args':args,'x':x,'rmse_list':rmse_list,'training_time_list':training_time_list,
                'f1_score_list':f1_score_list,'precision_list':precision_list,'recall_list':recall_list,'OPT':OPT} #保存文件

        f = open(save_path_name, 'wb')
        pickle.dump(data,f)
        f.close()