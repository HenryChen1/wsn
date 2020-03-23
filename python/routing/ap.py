from sklearn.cluster import AffinityPropagation
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import logging, sys

from python.routing.mte import *
from python.utils.utils import *
from python.network.node import *
from python.network.network import Network
from python.routing.routing_protocol import *
from python.sleep_scheduling.sleep_scheduler import *
import config as cf


class AP(RoutingProtocol):
  def _setup_phase(self, network):
        
    logging.debug('Affinity Propagation: setup phase')

    #   计算聚类数
    #   sensor_nodes = network.get_sensor_nodes()
    #   # calculate the average distance to the BS
    #   transform = lambda node: calculate_distance(node, network.get_BS())
    #   distances_to_BS = [transform(node) for node in sensor_nodes]
    #   avg_distance_to_BS = np.average(distances_to_BS)
    #   nb_clusters = cf.NB_CLUSTERS

    # 格式化数据集，调用AP算法API
    data = [[node.pos_x, node.pos_y] for node in network[0:-1]]
    data = np.array(data)
    af = AffinityPropagation(preference=-150000).fit(data)
    cluster_centers_indices = af.cluster_centers_indices_
    cluster_centroids = af.cluster_centers_
    labels = af.labels_
    # labels = labels.T
    n_clusters_ = len(cluster_centers_indices)

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





