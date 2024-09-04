import numpy as np
import math
import scipy.optimize as optimize
from utils.GaussianProcess import GaussianProcessRegressor

#dim = m,num = K in X, d+1 in Theta
def Data_Generation_X_deter(num,dim):
    # num <= dim -1 
    X = np.eye(num, M = dim - 1)
    X = 1/np.sqrt(2) * np.concatenate((np.ones((num,1)),X),axis = 1)
    #X[2:,0] = 0.0
    return X # shape (K,m)

def Data_Generation_Theta_deter(num,dim):
    Theta = np.zeros((num,dim))
    Theta[0,0:2] = 1/np.sqrt(2)  * np.ones((1,2))
    Theta[1,0] = 1/np.sqrt(2)
    Theta[1,2] = 1/np.sqrt(2)
    Theta[2,1:5] = 0.5 * np.ones((1,4))
    Theta[3:,:] = np.concatenate((np.zeros((num-3,3 )), np.eye(num - 3, M = dim - 3)),axis = 1)

    
    return Theta # shape(d+1,m)

def Simulate_Data(m,d,K,B,T):
    contexts = Data_Generation_X_deter(K,m)
    Theta = Data_Generation_Theta_deter(d+1,m)
    D = Theta @ contexts.T
    reward_vector = D[0,:]
    cost_matrix = D[1:,:]
    simplex_constriant = np.ones_like(reward_vector)
    A_ub = np.concatenate([cost_matrix,simplex_constriant[None,:]],axis = 0)
    #print(np.shape(A_ub))
    #print(np.size(reward_vector))

    b_ub = B/T * np.ones_like(A_ub[:,0])
    b_ub[d] = 1
    LP_solution = optimize.linprog( -reward_vector, A_ub, b_ub.T )
 #   print(LP_solution)
    optimal_allocation = LP_solution.x
    OPT = -1 *LP_solution.fun
    #print('OPT',OPT * T)
    return contexts,Theta,OPT,np.array(optimal_allocation)


# T = 1000
# m = 8
# d = 5
# K = 5
# B = T/4
# Z = T/B
# optimal_allocation = 0


def Simulation_CBwK( T,m,d,K,B, simulation_round = 10, online_oracle = 'Newton '):
    np.random.seed(0)
    
    Est_list = []
    Reg_list = []
    OPT_list = []
    Reward_list = []
    allocation_list = []
    gamma = 2 * np.sqrt(K * T / (m*np.log(T)))
    contexts,Theta,OPT,optimal_allocation =  Simulate_Data(m,d,K,B,T)
    if online_oracle == 'OGD':
        gamma = 2 *  np.sqrt(K) * (T ** 0.25)
#     for t in range(simulation_round):
# #         contexts_set.append(contexts)
# #         Theta_set.append(Theta)
# #         OPT_set.append(OPT)
# #         optimal_allocation_set.append(optimal_allocation)


    for t in range(simulation_round):
        Resource = B * np.ones((d,1))
        hat_Theta = np.ones((d+1,m))/np.sqrt(m) 


        D_hat = hat_Theta @ contexts.T
        reward_value = D_hat[0,:] # numpy array with size (K,)
        reward_value = reward_value[None,:] # numpy array with size (1,K)
        cost_value = D_hat[1:,:] # numpy array with size (d,K)
        action_allocation = np.zeros(K,)
        lamb = np.ones((d+1,1))/(d+1) # numpy array with size (d,1)
        step_size = 1.5
        dual_step_size = 2 * np.log(d) / np.sqrt(T)
        #gamma = np.sqrt(K*T)
        Z = T/B
        total_reward = 0
        empirical_cov = (step_size ** 2) * np.eye(m) #  numpy array with size (m,m)
        noise_std = 0.2
        for t in range(T):
            if np.min(Resource) < 2 or t == (T - 2): 
#                 print('K = ',K, 'd = ',d, 'm = ', m)
#                print('Resource consumed, Algorithm Stop at step ',t + 1,'/',T)
#                print('Total reward: ',total_reward.item() )
#                 print('Regret = ', T*OPT - total_reward.item() )
#                 print('Normalized Regret = ', (T*OPT - total_reward.item())/np.sqrt(T))
                Reward_list.append(total_reward.item())
                OPT_list.append(T*OPT)
                Reg_list.append(T*OPT - total_reward.item())
                Est_list.append(sum(abs(hat_Theta - Theta)) ** 2)
                allocation_list.append(action_allocation)
               # print('Stopped at ',t)
                break
            
            # Make action
            a = np.argmax(Lagrangian_sampling(reward_value.T, cost_value.T, Z, lamb[0:d,:], gamma))
            action_allocation[a] = action_allocation[a] + 1
            # Get feedback
            #print(a)
    
#             if a == 0:
#                 continue
#             else: 
            #print('non-null!')
            x = contexts[a,:,None] #numpy array, (m,1)
            observation =  Theta @ x
            observation = observation + noise_std * np.random.randn(*np.shape(observation)) 
            #print('shape:',observation.shape)
            grad = compute_gradient(x.T , observation, hat_Theta) # compute the gradient in linear setting.

            if online_oracle == 'OGD':
                #OGD update
                hat_Theta = OGD_update(hat_Theta,grad,step_size)
                step_size = 1/np.sqrt(T+1)
            else:
                #Newton update 

                empirical_cov = empirical_cov + x @ x.T
                inv_Hessian = np.linalg.inv(empirical_cov)
                hat_Theta = NGD_update(hat_Theta, grad, inv_Hessian, step_size)

            total_reward =  total_reward + (Theta @ x)[0]
            Resource = Resource - observation[1:,:]
            #update reward value
            D_hat = hat_Theta @ contexts.T
            reward_value = D_hat[0,:] # numpy array with size (K,)
            reward_value = reward_value[None,:] # numpy array with size (1,K)
            cost_value = D_hat[1:,:] # numpy array with size (d,K)
            #update lambda
            dual_grad = ( B/T * np.ones((d,1)) - observation[1:,:])
            lamb = OMD_update(lamb, dual_grad, dual_step_size)

            
    
    return Reg_list,OPT_list,Reward_list,Est_list,allocation_list




def Lagrangian_sampling(reward_value, cost_value , Z, lamb, gamma,K,B,T):
    # reward_value: numpy array with shape (K,1)
    # cost_value: numpy array with shape (K,d)  
    # lamb: numpy array with shape (d,1)
    # gamma: scalar
    # Return a K+1 dimensional distribution, where the 0-th componment is the probability of pulling null arm.
    
    # NonNull_value =  (reward_value + Z * ( B/T - cost_value) @ lamb).reshape(-1) # non-null values numpy array with shape (K,)
    NonNull_value =  (reward_value +  ( B/T - cost_value) @ lamb).reshape(-1) # non-null values numpy array with shape (K,)
    #print('NonNull_value',NonNull_value)
    #Null_value = Z * (B/T) # scalar
    #print('Null_value',Null_value)

    #Lagrangian_value = np.insert(NonNull_value,0,Null_value)
    Lagrangian_value = NonNull_value
    #print('Lagrangian_value',Lagrangian_value)
    
    Optimal_Index = np.argmax(Lagrangian_value) #  index of optimal empirical arm
    #print('Optimal_Index:',Optimal_Index)

    Lagrangian_gap = Lagrangian_value[Optimal_Index] - Lagrangian_value # numpy array with shape (K,)

    Lagrangian_score = 1/(K + gamma * Lagrangian_gap) # numpy array with shape (K,)
    Lagrangian_score[Optimal_Index] = (
        1.0 - np.sum(Lagrangian_score[:Optimal_Index]) -  np.sum(Lagrangian_score[Optimal_Index+1:]))
    # numpy array with shape (K,)
    #print('Lagrangian_score:',Lagrangian_score)
    #print(np.sum(Lagrangian_score))
    return np.random.multinomial(1,Lagrangian_score)


def OMD_update(old_lamb, gradient, stepsize, d):
    # old_lamb: numpy array with shape (d,1)
    # gradient: numpy array with shape (d,1)
    # stepsize: scalar
    padded_gradient = np.insert(gradient,d,0.0)
    # padded_gradient = gradient
#     print(gradient)
#     print(padded_gradient)
#     print(old_lamb)
    new_lamb = old_lamb * np.exp(-stepsize * padded_gradient[:,None])
    new_lamb = new_lamb / np.sum(new_lamb)
    return new_lamb


def OGD_update(theta, grad, step_size):
    #theta:  numpy array with shape (d+1,m)
    #gradient: numpy arrary with shape (d+1,m)
    #step_size: scalar
    theta_new = theta - step_size * grad
    
    # add projection step
    if (np.sum(theta_new ** 2) > 1):
        theta_new = theta_new/np.sqrt(np.sum(theta_new ** 2))
    theta_new[theta_new<0] = -theta_new[theta_new<0]
    return theta_new

def NGD_update(theta, grad, inv_Hessian, step_size):
    # theta:  numpy array with shape (m,d+1)
    # gradient: numpy arrary with shape (d+1,m)
    # Hessian: numpy arrary with shape (d+1,m,m)
    # step_size: scalar
    
    theta_new = theta - (step_size * inv_Hessian @ grad[:,:,None])[:,:,0]
    
    if (np.sum(theta_new ** 2) > 1):
        theta_new = theta_new/np.sqrt(np.sum(theta_new ** 2))
    theta_new[theta_new<0] = -theta_new[theta_new<0]
    return theta_new    


def compute_gradient(x, y, Theta): # compute the gradient in linear setting.
    #x: numpy array with shape (1,m)
    #y:  numpy array with shape (d+1,1)
    #Theta:  numpy array with shape (d+1,m)
    D = Theta @ x.T
    grad = 2 * (D - y) @ x
    #print(np.shape(grad))
    return grad

def gp_ucb(gpr,his_records,epoch,context=None):
    # x_train_list = []
    # y_train_list = []
    beta = gpr.cal_beta(epoch)        
    # actions = list(self.action_history.keys())
    # loss_dict, net_with_noise = self.evaluate_noise_levels(net)
    # actions = list(loss_dict.keys())  # 动作集合
    # rewards = {}  # 奖励字典
    # his_records = []

    actions = [0,1,2,3,4]

    action_context =[]
    # print("his_records:{}".format(his_records))
    for t in his_records:     
        action_context.append(np.concatenate((t[0:2]),axis=None))
        # print("his_records^t:{}".format(t))

    # his_records = np.array(his_records)
    # x_train = his_records[:, :-1]
    x_train = action_context
    # his_records = np.array(his_records)
    # y_train = his_records[:, -1]
    y_train = [row[-1] for row in his_records]
    upper_bound_list = []
    mean_list = []
    std_list = []
    for action in actions:
        # x_train = x_train_list[action]
        # y_train = y_train_list[action]
        x_predict = np.concatenate((np.array(action),context),axis=None)
        x_predict = np.reshape(x_predict, (1,context.shape[0]+1))
        K, K_star, K_2stars = gpr.fit(epoch, np.array(x_train), x_predict)
        mean, std = gpr.predict(K, K_star, K_2stars, np.array(y_train))
        mean = round(mean[0], 4)
        std = round(std[0], 4)
        mean_list.append(mean)
        std_list.append(std)
        upper_bound_list.append(mean + math.sqrt(beta) * std)


    # best_action = actions[np.argmax(upper_bound_list)]  # 获取置信上界最高的动作

    return mean_list
