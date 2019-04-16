import numpy as np
import scipy.cluster.hierarchy as hier
from construct_dendrogram import square_to_condensed

# Compute the cophenetic correlation between constructed dendrogram and ground truth
# NOTE: cophenetic corr is just the pearson correlation coefficient of the constructed dendrogrammatic
# distances with the ground truth dendrogrammatic distances
def coph_corr(constructed_dend, ind2bac):
    n = hier.to_tree(constructed_dend).get_count() # number of original elements
    constructed_dist_mat = dist_matrix(constructed_dend)
    ground_truth_dict = get_gt()

    # Get condensed vector form of dendro dists for constructed dendro
    constructed_dists = np.zeros(int(n*(n-1)/2)) # n choose 2 indices
    for i in range(n):
        for j in range(i+1, n):
            constructed_dists[square_to_condensed(i, j, n)] = constructed_dist_mat[i, j]

    # Get condensed vector form of dendro dists for ground truth dendro
    gt_dists = np.zeros(int(n*(n-1)/2)) # n choose 2 indices
    for i in range(n):
        bac1 = ind2bac[i]
        for j in range(i+1, n):
            bac2 = ind2bac[j]
            gt_dists[square_to_condensed(i, j, n)] = ground_truth_dict[bac1][bac2]

    return np.corrcoef(constructed_dists, gt_dists)[0,1]

def get_gt():
	gt = {
    	'faecalis' : {'faecalis':0,'pneumonia':1, 'monocytogenes':3, 'subtilis':3, 'bovis':6,'tuberculosis':6,'paratuberculosis':6,'avermitilis':5,'aeruginosa':6,'ecoli':6,'enterica':7,'typhi':7,'meningitis':8,'pertussis':8,'violaceum':7,'sicca':6,'pylori':8,'hepaticus':8,'succinogenus':7,'jejuni':6},
    	'pneumonia' : {'faecalis':1,'pneumonia':0, 'monocytogenes':3, 'subtilis':3, 'bovis':6,'tuberculosis':6,'paratuberculosis':6,'avermitilis':5,'aeruginosa':6,'ecoli':6,'enterica':7,'typhi':7,'meningitis':8,'pertussis':8,'violaceum':7,'sicca':6,'pylori':8,'hepaticus':8,'succinogenus':7,'jejuni':6},    
		'monocytogenes' : {'faecalis':3,'pneumonia':3, 'monocytogenes':0, 'subtilis':1, 'bovis':6,'tuberculosis':6,'paratuberculosis':6,'avermitilis':5,'aeruginosa':6,'ecoli':6,'enterica':7,'typhi':7,'meningitis':8,'pertussis':8,'violaceum':7,'sicca':6,'pylori':8,'hepaticus':8,'succinogenus':7,'jejuni':6},
		'subtilis' : {'faecalis':3,'pneumonia':3, 'monocytogenes':1, 'subtilis':0, 'bovis':6,'tuberculosis':6,'paratuberculosis':6,'avermitilis':5,'aeruginosa':6,'ecoli':6,'enterica':7,'typhi':7,'meningitis':8,'pertussis':8,'violaceum':7,'sicca':6,'pylori':8,'hepaticus':8,'succinogenus':7,'jejuni':6},
		'bovis' : {'faecalis':6,'pneumonia':6, 'monocytogenes':6, 'subtilis':6, 'bovis':0,'tuberculosis':1,'paratuberculosis':1,'avermitilis':2,'aeruginosa':5,'ecoli':5,'enterica':6,'typhi':6,'meningitis':7,'pertussis':7,'violaceum':6,'sicca':5,'pylori':7,'hepaticus':7,'succinogenus':6,'jejuni':5},
		'tuberculosis' : {'faecalis':6,'pneumonia':6, 'monocytogenes':6, 'subtilis':6, 'bovis':1,'tuberculosis':0,'paratuberculosis':1,'avermitilis':2,'aeruginosa':5,'ecoli':5,'enterica':6,'typhi':6,'meningitis':7,'pertussis':7,'violaceum':6,'sicca':5,'pylori':7,'hepaticus':7,'succinogenus':6,'jejuni':5},
		'paratuberculosis' : {'faecalis':6,'pneumonia':6, 'monocytogenes':6, 'subtilis':6, 'bovis':1,'tuberculosis':1,'paratuberculosis':0,'avermitilis':2,'aeruginosa':5,'ecoli':5,'enterica':6,'typhi':6,'meningitis':7,'pertussis':7,'violaceum':6,'sicca':5,'pylori':7,'hepaticus':7,'succinogenus':6,'jejuni':5},
		'avermitilis' : {'faecalis':5,'pneumonia':5, 'monocytogenes':5, 'subtilis':5, 'bovis':1,'tuberculosis':1,'paratuberculosis':0,'avermitilis':2,'aeruginosa':5,'ecoli':5,'enterica':6,'typhi':6,'meningitis':7,'pertussis':7,'violaceum':6,'sicca':5,'pylori':7,'hepaticus':7,'succinogenus':6,'jejuni':5},
		'aeruginosa' : {'faecalis':6,'pneumonia':6, 'monocytogenes':6, 'subtilis':6, 'bovis':5,'tuberculosis':5,'paratuberculosis':5,'avermitilis':4,'aeruginosa':0,'ecoli':1,'enterica':2,'typhi':2,'meningitis':5,'pertussis':5,'violaceum':4,'sicca':3,'pylori':5,'hepaticus':5,'succinogenus':4,'jejuni':3},
		'ecoli' : {'faecalis':6,'pneumonia':6, 'monocytogenes':6, 'subtilis':6, 'bovis':5,'tuberculosis':5,'paratuberculosis':5,'avermitilis':4,'aeruginosa':1,'ecoli':0,'enterica':2,'typhi':2,'meningitis':5,'pertussis':5,'violaceum':4,'sicca':3,'pylori':5,'hepaticus':5,'succinogenus':4,'jejuni':3},
		'enterica' : {'faecalis':7,'pneumonia':7, 'monocytogenes':7, 'subtilis':7, 'bovis':6,'tuberculosis':6,'paratuberculosis':6,'avermitilis':5,'aeruginosa':2,'ecoli':2,'enterica':0,'typhi':1,'meningitis':6,'pertussis':6,'violaceum':5,'sicca':4,'pylori':6,'hepaticus':6,'succinogenus':5,'jejuni':4},
		'typhi' : {'faecalis':7,'pneumonia':7, 'monocytogenes':7, 'subtilis':7, 'bovis':6,'tuberculosis':6,'paratuberculosis':6,'avermitilis':5,'aeruginosa':2,'ecoli':2,'enterica':1,'typhi':0,'meningitis':6,'pertussis':6,'violaceum':5,'sicca':4,'pylori':6,'hepaticus':6,'succinogenus':5,'jejuni':4},
		'meningitis' : {'faecalis':8,'pneumonia':8, 'monocytogenes':8, 'subtilis':8, 'bovis':7,'tuberculosis':7,'paratuberculosis':7,'avermitilis':6,'aeruginosa':5,'ecoli':5,'enterica':6,'typhi':6,'meningitis':0,'pertussis':1,'violaceum':2,'sicca':3,'pylori':7,'hepaticus':7,'succinogenus':6,'jejuni':5},
		'pertussis' : {'faecalis':8,'pneumonia':8, 'monocytogenes':8, 'subtilis':8, 'bovis':7,'tuberculosis':7,'paratuberculosis':7,'avermitilis':6,'aeruginosa':5,'ecoli':5,'enterica':6,'typhi':6,'meningitis':1,'pertussis':0,'violaceum':2,'sicca':3,'pylori':7,'hepaticus':7,'succinogenus':6,'jejuni':5},
		'violaceum' : {'faecalis':7,'pneumonia':7, 'monocytogenes':7, 'subtilis':7, 'bovis':6,'tuberculosis':6,'paratuberculosis':6,'avermitilis':5,'aeruginosa':4,'ecoli':4,'enterica':5,'typhi':5,'meningitis':2,'pertussis':2,'violaceum':0,'sicca':2,'pylori':6,'hepaticus':6,'succinogenus':5,'jejuni':4},
		'sicca' : {'faecalis':6,'pneumonia':6, 'monocytogenes':6, 'subtilis':6, 'bovis':5,'tuberculosis':5,'paratuberculosis':5,'avermitilis':4,'aeruginosa':3,'ecoli':3,'enterica':4,'typhi':4,'meningitis':3,'pertussis':3,'violaceum':2,'sicca':0,'pylori':5,'hepaticus':5,'succinogenus':4,'jejuni':3},
		'pylori' :  {'faecalis':8,'pneumonia':8, 'monocytogenes':8, 'subtilis':8, 'bovis':7,'tuberculosis':7,'paratuberculosis':7,'avermitilis':6,'aeruginosa':5,'ecoli':5,'enterica':6,'typhi':6,'meningitis':7,'pertussis':7,'violaceum':6,'sicca':5,'pylori':0,'hepaticus':1,'succinogenus':2,'jejuni':3},
		'hepaticus' : {'faecalis':8,'pneumonia':8, 'monocytogenes':8, 'subtilis':8, 'bovis':7,'tuberculosis':7,'paratuberculosis':7,'avermitilis':6,'aeruginosa':5,'ecoli':5,'enterica':6,'typhi':6,'meningitis':7,'pertussis':7,'violaceum':6,'sicca':5,'pylori':1,'hepaticus':0,'succinogenus':2,'jejuni':3},
		'succinogenus' :{'faecalis':7,'pneumonia':7, 'monocytogenes':7, 'subtilis':7, 'bovis':6,'tuberculosis':6,'paratuberculosis':6,'avermitilis':5,'aeruginosa':4,'ecoli':4,'enterica':5,'typhi':5,'meningitis':6,'pertussis':6,'violaceum':5,'sicca':4,'pylori':2,'hepaticus':2,'succinogenus':0,'jejuni':2},
		'jejuni' : {'faecalis':6,'pneumonia':6, 'monocytogenes':6, 'subtilis':6, 'bovis':5,'tuberculosis':5,'paratuberculosis':5,'avermitilis':4,'aeruginosa':3,'ecoli':3,'enterica':4,'typhi':4,'meningitis':5,'pertussis':5,'violaceum':4,'sicca':3,'pylori':3,'hepaticus':3,'succinogenus':2,'jejuni':0},
	}
	return gt


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
