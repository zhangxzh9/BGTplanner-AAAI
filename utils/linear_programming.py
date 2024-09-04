


import pulp
import math

def solve_linear_programming(T0, K, hat_f, B, d, T,M,action_map_epsilon):
    
    # 创建线性规划问题
    problem = pulp.LpProblem("MaximizeObjective", pulp.LpMaximize)

    # M = math.sqrt(4* math.log(T*d)/T0)

    # 定义变量 p_{t, a}，这些变量是二进制的
    # T0 =  # 设置T0的值，即时间步的数量
    # K =  # 设置K的值，即行动的数量

    p = pulp.LpVariable.dicts("p", [(t, a) for t in range(T0) for a in range(K)], 0, 1, pulp.LpBinary)

    # 定义目标函数
    hat_f_values = {}  # 存储 \hat{f}_0\left(x_t, a\right) 的值，以字典形式存储
    hat_g_values = {}  # 存储 \hat{\boldsymbol{g}}_0\left(x_t, a\right) 的值，以字典形式存储

    for t in range(T0):
        for a in range(K):
            hat_f_values[(t, a)] =  hat_f[a] # 设置 \hat{f}_0\left(x_t, a\right) 的值
            hat_g_values[(t, a)] =  action_map_epsilon[a] # 设置 \hat{\boldsymbol{g}}_0\left(x_t, a\right) 的值

    problem += pulp.lpSum(hat_f_values[(t, a)] * p[(t, a)] for t in range(T0) for a in range(K)) / T0

    # 定义约束条件
    # B =  # 设置B的值
    # T =  # 设置T的值
    # M =  # 设置M的值

    problem += pulp.lpSum(hat_g_values[(t, a)] * p[(t, a)] for t in range(T0) for a in range(K)) <= (B / T + 2 * M)

    # 求解线性规划问题
    problem.solve()

    # 输出结果
    if pulp.LpStatus[problem.status] == "Optimal":
        optimal_solution = {}
        for t in range(T0):
            for a in range(K):
                optimal_solution[(t, a)] = int(p[(t, a)].value())
        print("Optimal Solution:")
        print(optimal_solution)
        print("Objective Value:", pulp.value(problem.objective))
        return  pulp.value(problem.objective)
    else:
        print("No optimal solution found.")
        return 0

    


