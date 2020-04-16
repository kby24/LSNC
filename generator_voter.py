# used for networkx version 1.11 and python 2.7
import numpy as np
import networkx as nx
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import random
import math
import pickle

# global config
# set seed
np.random.seed(2051)
# random / all : randomly init for each sample / init all 2^N_Node states
Init_Way = 'random'
# Reinit_Time = 1024
# Config of network topology
N_Node = 1000
Goal_Data_Num = 30000  #
G_Type = 'renyi'
Average_Degree = 4
Ws_Nei = 3
Ws_P = 0.3
renyi_p = 0.04

# Config of Dyn:table/prob
DYN_Type = 'prob'
DYN = 'voter'
# Desginage_Dyn = {0:0,1:0,2:0,3:0}

# config of way to detect inherent character of this network

Draw_Grow_Step = 50
# Derrida_Re_Time = 100
# if dyn type is table , the goal data num means we have to explore that much different state
# elif dyn type is prob, means we have to let net spread through that much state and dont require they are different
Draw_Hm_Time = 20
# mark
mark = random.randint(0, 100000)
print('random mark' + str(mark))
# store folder
Store_Folder = '/data/zhangyan/voter'


# 一些函数

# generate the network
def generate_network(g_type='random', n_node=5, average_degree=3):
    # generate random network
    if g_type == 'random':
        dg = nx.DiGraph();
        # add nodes
        for i in range(n_node):
            dg.add_node(i, value=random.randint(0, 1))
        # num of edges
        edge_num = n_node * average_degree;
        edges = []
        while len(edges) < edge_num:
            start_node = random.randint(0, n_node - 1);
            end_node = random.randint(0, n_node - 1);
            if start_node != end_node and [start_node, end_node] not in edges:
                edges.append([start_node, end_node])
        # add those num
        for edge in edges:
            dg.add_edge(edge[0], edge[1]);
        return dg
    # generate n-k automata
    elif g_type == 'nkautomata':
        dg = nx.DiGraph();
        # add nodes with a random value of 0 or 1
        for i in range(N_Node):
            dg.add_node(i, value=random.randint(0, 1))
        # add edge : every node with k edge directing to it
        for i in range(N_Node):
            edges_to_this_node = []
            for j in range(Average_Degree):
                # choose a starter and direct to this node
                add = False
                while not add:
                    starter = random.randint(0, n_node - 1)
                    if starter not in edges_to_this_node:
                        edges_to_this_node.append(starter)
                        dg.add_edge(starter, i)
                        add = True
        return dg
    # generate ba scale free network
    elif g_type == 'ba':
        BA = nx.random_graphs.barabasi_albert_graph(N_Node, 2)
        return BA
    elif g_type == 'ws':
        WS = nx.random_graphs.watts_strogatz_graph(N_Node, Ws_Nei, Ws_P)
        return WS
    elif g_type == 'renyi':
        renyi = nx.random_graphs.erdos_renyi_graph(N_Node, renyi_p)
        return renyi


# get the innode of each node
# return:{0:[1,2,3],1:[0,4]...}
def get_innode(adj):
    innodes = {}
    for i in range(adj.shape[0]):
        innode = []
        for j in range(adj.shape[0]):
            if adj[j][i] == 1:
                innode.append(j)
        innodes[i] = innode
    return innodes


# init node data randomly
def init_node(dg):
    for i in range(dg.number_of_nodes()):
        dg.nodes[i]['value'] = random.randint(0, 1)


# let the net spread by probility
def spread_prob(dg, DYN, step=100):
    node_num = dg.number_of_nodes()
    # data to be returned
    data = []
    # add initial value to data
    origin_val = []
    for i in range(node_num):
        origin_val.append(dg.nodes[i]['value'])
    data.append(origin_val)

    # control the circulates
    run = 0
    # step is the only limitation because there is no conception like attractor and so on...
    while run < step:
        run += 1
        # each step
        next_val = []
        # if DYN is voter
        if DYN == 'voter':
            for i in range(node_num):
                # num for neighbors who vote for agree
                k = 0.
                # num for all neighbors
                m = len(innodes[i])
                for iter, val in enumerate(innodes[i]):
                    if dg.nodes[val]['value'] == 1:
                        k += 1.
                if random.random() < k / m:
                    next_val.append(1)
                else:
                    next_val.append(0)

        # print(next_val)
        # set value to the net
        for i in range(node_num):
            dg.nodes[i]['value'] = next_val[i]

        # just add to data to record
        data.append(next_val)
    return np.array(data)


dg = generate_network(g_type='ws', n_node=N_Node, average_degree=Average_Degree)
# dg = nx.random_graphs.barabasi_albert_graph(N_Node, 4)
# adj mat
print(dg)
adj = nx.adjacency_matrix(dg).toarray()
print(adj)
# innode of each node

innodes = get_innode(adj);

print("analyze of the graph")

# generate data
all_data = np.array([[-1]])
# has been explored
has_explored = []
datas = []

# in this way we aim to generate data until final number of all data reachs goal data num\
i = 0
while len(has_explored) < Goal_Data_Num:
    print('how many has we explored')
    # if i % 10 == 0:
    print(len(has_explored))

    # print('initial time-----'+str(i))
    # init node with a perticular num,if net is too big then use random init instead
    if Init_Way == 'random':
        init_node(dg)

    # init state
    init_state = []
    for j in range(N_Node):
        init_state.append(dg.nodes[j]['value'])
    # if this state has been explored
    if init_state in has_explored:
        continue
    else:
        has_explored.append(init_state)

    # spread
    if DYN_Type == 'prob':
        data = spread_prob(dg, DYN, step=50)
        datas.append(data)

    # make each [a,b,c,d] to [a,b,b,c,c,d]
    # (2,3)means(2step,3node)

    # if only one point ,that means it is a eden state and a fix point
    if data.shape[0] == 1:
        temp = np.zeros((2, data.shape[1]))
        temp[0] = data
        temp[1] = data
        data = temp
    expand_data = np.zeros((2 * data.shape[0] - 2, data.shape[1]))
    for j in range(data.shape[0]):
        # add to has explored
        if j < data.shape[0] - 1:
            cur_state = data[j].tolist()
            # if dyn type is table, then we have to make sure weather the state is explored or not
            if DYN_Type == 'table':
                if cur_state not in has_explored:
                    has_explored.append(cur_state)
            # dyn type is prob means we just have to record how many state we have visited
            elif DYN_Type == 'prob':
                has_explored.append(cur_state)

        # generate data to use
        if j == 0:
            expand_data[0] = data[0]
        elif j == data.shape[0] - 1:
            expand_data[expand_data.shape[0] - 1] = data[j]
        else:
            # j between first and last
            expand_data[2 * j - 1] = data[j]
            expand_data[2 * j] = data[j]
    # print(expand_data)
    # concat data in every step
    if all_data[0][0] == -1:
        all_data = expand_data
    else:
        all_data = np.concatenate((all_data, expand_data), axis=0)

print(all_data)
print(all_data.shape)
# change the shape from(step,node_num) => (step,node_num,1)
all_data = all_data[:, :, np.newaxis]
print(all_data.shape)

# save the data
# save time series data
serise_address = Store_Folder + 'mark-' + str(G_Type) + str(N_Node) + str(Goal_Data_Num) + '-series.pickle'
with open(serise_address, 'wb') as f:
    pickle.dump(all_data, f)

# save adj mat
adj_address = Store_Folder + 'mark-' + str(G_Type) + str(N_Node) + str(Goal_Data_Num) + '-adjmat.pickle'
with open(adj_address, 'wb') as f:
    pickle.dump(adj, f)
