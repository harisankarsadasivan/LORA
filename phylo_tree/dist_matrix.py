import numpy as np
import scipy.cluster.hierarchy as hier

import matplotlib.pyplot as plt

def dist_matrix(link):
	'''
	Compute distance of each pair of nodes in a tree
	Input:
		- linkage matrix of hierarchical clustering
	Return:
		- numpy array where A[i,j] indicates distance between node i and node j
	'''
	rootnode = hier.to_tree(link)  # convert to a tree object
	n = rootnode.get_count()

	tree_dist = np.zeros(shape=(n, n))
	for i in range(n):
		for j in range(i+1, n):
			tree_dist[i, j] = calc_node_dist(rootnode, i, j)
			tree_dist[j, i] = tree_dist[i, j]

	return tree_dist


def find_lca(root, n1, n2):
	'''
	Find lowest common ancestor of two given nodes
	Input:
		- root node
		- two nodes
	Return:
		- lowest common ancestor
	'''
	if root is None:
		return None

	if root.get_id() == n1 or root.get_id() == n2:
		return root

	left = find_lca(root.get_left(), n1, n2)
	right = find_lca(root.get_right(), n1, n2)

	if left is not None and right is not None:
		return root

	if left:
		return left
	else:
		return right


def find_level(root, idx, d, lvl):
	'''
	Get distance of a node from root
	'''
	if root is None:
		return

	if root.get_id() == idx:
		d.append(lvl)
		return

	find_level(root.get_left(), idx, d, lvl+1)
	find_level(root.get_right(), idx, d, lvl+1)



def calc_node_dist(root, n1, n2):
	'''
	Calculate distance between two nodes in a tree
	Input:
		- root node
		- two nodes
	Return:
		- distance
	'''
	d1 = []  # distance of n1 from lowest common ancestor
	d2 = []  # distance of n2 from lowest common ancestor

	lca = find_lca(root, n1, n2)
	if lca:
		find_level(lca, n1, d1, 0)
		find_level(lca, n2, d2, 0)
		return d1[0] + d2[0]
	else:
		return -1
