"""
Implements maximum entropy inverse reinforcement learning (Ziebart et al., 2008)
Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

from itertools import product#Python的内建模块itertools提供了非常有用的用于操作迭代对象的函数
import numpy as np
import numpy.random as rn
from irl import value_iteration

def irl(feature_matrix, n_actions, discount, transition_probability,
        trajectories, epochs, learning_rate):
    '''
    找到给定轨迹的奖励函数
    :param feature_matrix:第n行表示第n种状态的矩阵。
            形状为(N, D)的NumPy数组，其中N为状态数，D为状态的维数
    :param n_actions:动作数量，int
    :param discount:MDP的折扣因子,float
    :param transition_probability:动作action下状态从state_i转换到state_k的概率分布,
            Numpy array(state_i,action,state_k),尺寸：(N,A,N)
    :param trajectories:状态-动作的三位数组。状态为int型，action为int型。
            这是一个尺寸为(T,L,2)的Numpy数组，其中T为轨迹数量，L为轨迹长度
            [
            T1:[(s1,a1),(s2,a2),(s3,a3),(s4,a4),(s5,a5),...(sL,aL)],
            T2:[(s1,a1),(s2,a2),(s3,a3),(s4,a4),(s5,a5),...(sL,aL)],
            ...
            TT:[(s1,a1),(s2,a2),(s3,a3),(s4,a4),(s5,a5),...(sL,aL)]
            ]
    :param epochs:梯度下降步数
    :param learning_rate:学习率
    :return:尺寸为(N,)的奖励向量
    '''
    n_states, d_states = feature_matrix.shape
    #n_state:状态数
    #d_state:状态维数
    #初始化权重
    alpha = rn.uniform(size=(d_states,))
    '''
        函数原型:numpy.random.uniform(low,high,size)
        功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
        size=(d_states,):生成一个一维数组，数组为d_state列
    '''
    # Calculate the feature expectations \tilde{phi}.
    #计算特征期望，累加特征向量中特征并求均值
    feature_expectations = find_feature_expectations(feature_matrix,
                                                     trajectories)#第n中状态对应的轨迹？？
    # Gradient descent on alpha.
    #alpha上的梯度下降
    for i in range(epochs):#epochs:梯度下降次数，对于每一次梯度下降：
        # print("i: {}".format(i))
        r = feature_matrix.dot(alpha)#返回两个矩阵内积,r=特征向量和系数的内积,r为N*1的向量
        expected_svf = find_expected_svf(n_states, r, n_actions, discount,
                                         transition_probability, trajectories)
        grad = feature_expectations - feature_matrix.T.dot(expected_svf)

        alpha += learning_rate * grad

    return feature_matrix.dot(alpha).reshape((n_states,))

def find_svf(n_states, trajectories):
    '''
    从轨迹中找出州的访问频率
    :param n_states:
    :param trajectories:
    :return:
    '''
    """
    Find the state visitation frequency from trajectories.

    n_states: Number of states. int.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> State visitation frequencies vector with shape (N,).
    """

    svf = np.zeros(n_states)

    for trajectory in trajectories:
        for state, _, _ in trajectory:
            svf[state] += 1

    svf /= trajectories.shape[0]

    return svf

def find_feature_expectations(feature_matrix, trajectories):
    '''
    找出给定轨迹的特征期望值,这是平均路径特征向量
    :param feature_matrix:状态矩阵
    :param trajectories:轨迹
    :return: 返回特征矩阵按行叠加后处以行数的一维向量,(1,feature_matrix.shape[1])
    '''
    """
    Find the feature expectations for the given trajectories. This is the
    average path feature vector.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Feature expectations vector with shape (D,).
    """

    feature_expectations = np.zeros(feature_matrix.shape[1])#初始化，全0

    for trajectory in trajectories:#对于每一条轨迹：
        for state, _, _ in trajectory:#对于每一条轨迹：
            feature_expectations += feature_matrix[state]#[0,0,0]+=[1,2,3]=[1,2,3]

    feature_expectations /= trajectories.shape[0]#求每个特征均值

    return feature_expectations#返回特征期望

def find_expected_svf(n_states, r, n_actions, discount,
                      transition_probability, trajectories):
    '''
    从轨迹中找出期望状态的访问频率，svf是啥——state visitation frequencies
    :param n_states:
    :param r:
    :param n_actions:
    :param discount:
    :param transition_probability:
    :param trajectories:
    :return:
    '''
    """
    Find the expected state visitation frequencies using algorithm 1 from
    Ziebart et al. 2008.

    n_states: Number of states N. int.
    alpha: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Expected state visitation frequencies vector with shape (N,).
    """

    n_trajectories = trajectories.shape[0]#轨迹数量
    trajectory_length = trajectories.shape[1]#轨迹长度

    # policy = find_policy(n_states, r, n_actions, discount,
    #                                 transition_probability)
    policy = value_iteration.find_policy(n_states, n_actions,
                                         transition_probability, r, discount)
    #输出为每个状态的价值
    start_state_count = np.zeros(n_states)
    for trajectory in trajectories:
        start_state_count[trajectory[0, 0]] += 1
    p_start_state = start_state_count/n_trajectories

    expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[k, t] += (expected_svf[i, t-1] *
                                  policy[i, j] * # Stochastic policy
                                  transition_probability[i, j, k])

    return expected_svf.sum(axis=1)

def softmax(x1, x2):
    """
    Soft-maximum calculation, from algorithm 9.2 in Ziebart's PhD thesis.

    x1: float.
    x2: float.
    -> softmax(x1, x2)
    """

    max_x = max(x1, x2)
    min_x = min(x1, x2)
    return max_x + np.log(1 + np.exp(min_x - max_x))

def find_policy(n_states, r, n_actions, discount,
                           transition_probability):
    """
    Find a policy with linear value iteration. Based on the code accompanying
    the Levine et al. GPIRL paper and on Ziebart's PhD thesis (algorithm 9.1).

    n_states: Number of states N. int.
    r: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    -> NumPy array of states and the probability of taking each action in that
        state, with shape (N, A).
    """

    # V = value_iteration.value(n_states, transition_probability, r, discount)

    # NumPy's dot really dislikes using inf, so I'm making everything finite
    # using nan_to_num.
    V = np.nan_to_num(np.ones((n_states, 1)) * float("-inf"))

    diff = np.ones((n_states,))
    while (diff > 1e-4).all():  # Iterate until convergence.
        new_V = r.copy()
        for j in range(n_actions):
            for i in range(n_states):
                new_V[i] = softmax(new_V[i], r[i] + discount*
                    np.sum(transition_probability[i, j, k] * V[k]
                           for k in range(n_states)))

        # # This seems to diverge, so we z-score it (engineering hack).
        new_V = (new_V - new_V.mean())/new_V.std()

        diff = abs(V - new_V)
        V = new_V

    # We really want Q, not V, so grab that using equation 9.2 from the thesis.
    Q = np.zeros((n_states, n_actions))
    for i in range(n_states):
        for j in range(n_actions):
            p = np.array([transition_probability[i, j, k]
                          for k in range(n_states)])
            Q[i, j] = p.dot(r + discount*V)

    # Softmax by row to interpret these values as probabilities.
    Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
    Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
    return Q

def expected_value_difference(n_states, n_actions, transition_probability,
    reward, discount, p_start_state, optimal_value, true_reward):
    """
    Calculate the expected value difference, which is a proxy to how good a
    recovered reward function is.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    reward: Reward vector mapping state int to reward. Shape (N,).
    discount: Discount factor. float.
    p_start_state: Probability vector with the ith component as the probability
        that the ith state is the start state. Shape (N,).
    optimal_value: Value vector for the ground reward with optimal policy.
        The ith component is the value of the ith state. Shape (N,).
    true_reward: True reward vector. Shape (N,).
    -> Expected value difference. float.
    """

    policy = value_iteration.find_policy(n_states, n_actions,
        transition_probability, reward, discount)
    value = value_iteration.value(policy.argmax(axis=1), n_states,
        transition_probability, true_reward, discount)

    evd = optimal_value.dot(p_start_state) - value.dot(p_start_state)
    return evd
