# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea # import geatpy
from python.routing.ea_kmeans.my_problem import MyProblem # 导入自定义问题接口
from scipy.spatial.distance import cdist

from python.network.node import *
from python.network.network import Network
from python.routing.routing_protocol import *
from python.sleep_scheduling.sleep_scheduler import *
import config as cf

class EA_KMEANS(RoutingProtocol):
    def _setup_phase(self, network):
        """===============================生成數據文件==========================="""
        data = np.array([[node.pos_x, node.pos_y] for node in network[0:-1]])
        np.savetxt('data.csv', data, delimiter = ',')
        """===============================实例化问题对象==========================="""
        problem = MyProblem() # 生成问题对象
        """=================================种群设置==============================="""
        Encoding = 'RI'       # 编码方式
        NIND = 2              # 种群规模
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
        population = ea.Population(Encoding, Field, NIND) # 实例化种群对象
        """===============================算法参数设置============================="""
        myAlgorithm = ea.soea_EGA_templet(problem, population)
        myAlgorithm.MAXGEN = 30 # 最大进化代数
        myAlgorithm.trappedValue = 1e-4 # “进化停滞”判断阈值
        myAlgorithm.maxTrappedCount = 10 # 进化停滞计数器最大上限值，如果连续maxTrappedCount代被判定进化陷入停滞，则终止进化
        """==========================调用算法模板进行种群进化======================="""
        [population, obj_trace, var_trace] = myAlgorithm.run() # 执行算法模板
        # population.save() # 把最后一代种群的信息保存到文件中
        """=================================輸出结果==============================="""
        best_gen = np.argmin(problem.maxormins * obj_trace[:, 1]) # 记录最优种群个体是在哪一代
        # best_ObjV = obj_trace[best_gen, 1]
        # print('最优的目标函数值为：%s'%(best_ObjV))
        # print('最优的聚类中心为：')
        Phen = var_trace[best_gen, :]
        cluster_centroids = Phen.reshape(problem.k, int(len(Phen) / problem.k)) # 得到最优的聚类中心
        # print(centers)
        # print('有效进化代数：%s'%(obj_trace.shape[0]))
        # print('最优的一代是第 %s 代'%(best_gen + 1))
        # print('评价次数：%s'%(myAlgorithm.evalsNum))
        # print('时间已过 %s 秒'%(myAlgorithm.passTime))
        # problem.draw(centers)
        dis = cdist(cluster_centroids, data, 'euclidean')
        dis_split = dis.reshape(1, cf.NB_CLUSTERS, data.shape[0])
        labels = np.argmin(dis_split, 1)[0]
        n_clusters_ = cf.NB_CLUSTERS

        heads = []
        network.centroids = []
        for cluster_id, centroid in enumerate(cluster_centroids):
            tmp_centroid = Node(0)
            tmp_centroid.pos_x = centroid[0]
            tmp_centroid.pos_y = centroid[1]
            network.centroids.append(tmp_centroid)  # FCM确定的聚类中心
            nearest_node = None
            shortest_distance = cf.INFINITY
            for node in network[0:-1]:
                distance = calculate_distance(node, tmp_centroid)
                # 计算与聚类中心最近的节点
                if distance < shortest_distance:
                    nearest_node = node
                    shortest_distance = distance
                # 选择最近聚类中心节点为簇头
            nearest_node.next_hop = cf.BSID
            nearest_node.membership = cluster_id  # 标记为该类群
            heads.append(nearest_node)

        for i, node in enumerate(network[0:-1]):
            if node in heads:  # node is already a cluster head 过滤簇头节点
                continue
            cluster_id = labels[i]
            node.membership = cluster_id
            head = [x for x in heads if x.membership == cluster_id][0]  # 普通节点选择该类群的簇头作为下一跳
            node.next_hop = head.id

        self.head_rotation(network, n_clusters_)
        # logging.DEBUG('Estimated number of clusters: %d' % n_clusters_)

    def head_rotation(self, network, n_clusters_):
        logging.debug('AP: head rotation')
        # head rotation
        # current cluster heads choose next cluster head with the most
        # residual energy and nearest to the cluster centroid
        # 当前簇头选举：选择剩余能量最大的节点
        for cluster_id in range(0, n_clusters_):
            cluster = network.get_nodes_by_membership(cluster_id)
            # check if there is someone alive in this cluster
            if len(cluster) == 0:
                continue

                # someone is alive, find node with highest energy in the cluster
                # to be the next cluster head
            highest_energy = cf.MINUS_INFINITY
            next_head = None
            for node in cluster:
                if node.energy_source.energy > highest_energy:
                    highest_energy = node.energy_source.energy
                    next_head = node

            for node in cluster:
                node.next_hop = next_head.id
            next_head.next_hop = cf.BSID