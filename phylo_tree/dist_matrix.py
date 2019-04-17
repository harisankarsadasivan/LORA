import numpy as np
import scipy.cluster.hierarchy as hier
from construct_dendrogram import square_to_condensed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

    corr = np.corrcoef(constructed_dists, gt_dists)[0,1]
    plt.figure()
    plt.hist2d(gt_dists, constructed_dists, (8,12), cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel('Ground Truth Distance')
    plt.ylabel('Constructed Distance')
    plt.title('Cophenetic Correlation = ' + str(corr)[:5])
    plt.savefig('corr.png', bbox_inches='tight')
    return corr

def get_gt():
        
        gt = {
    	'faecalis' : {'faecalis':0,'pneumonia':2, 'monocytogenes':3, 'subtilis':3, 'bovis':8,'tuberculosis':8,'paratuberculosis':7,'avermitilis':6,'aeruginosa':8,'ecoli':9,'enterica':10,'typhi':10,'meningitis':10,'pertussis':10,'violaceum':9,'sicca':8,'pylori':9,'hepaticus':9,'succinogenus':8,'jejuni':7},
    	'pneumonia' : {'faecalis':2,'pneumonia':0, 'monocytogenes':3, 'subtilis':3, 'bovis':8,'tuberculosis':8,'paratuberculosis':7,'avermitilis':6,'aeruginosa':8,'ecoli':9,'enterica':10,'typhi':10,'meningitis':10,'pertussis':10,'violaceum':9,'sicca':8,'pylori':9,'hepaticus':9,'succinogenus':8,'jejuni':7},
    	'monocytogenes' : {'faecalis':3,'pneumonia':3, 'monocytogenes':0, 'subtilis':2, 'bovis':8,'tuberculosis':8,'paratuberculosis':7,'avermitilis':6,'aeruginosa':8,'ecoli':9,'enterica':10,'typhi':10,'meningitis':10,'pertussis':10,'violaceum':9,'sicca':8,'pylori':9,'hepaticus':9,'succinogenus':8,'jejuni':7},
    	'subtilis' : {'faecalis':3,'pneumonia':3, 'monocytogenes':2, 'subtilis':0, 'bovis':8,'tuberculosis':8,'paratuberculosis':7,'avermitilis':6,'aeruginosa':8,'ecoli':9,'enterica':10,'typhi':10,'meningitis':10,'pertussis':10,'violaceum':9,'sicca':8,'pylori':9,'hepaticus':9,'succinogenus':8,'jejuni':7},
    	'bovis' : {'faecalis':8,'pneumonia':8, 'monocytogenes':8, 'subtilis':8, 'bovis':0,'tuberculosis':2,'paratuberculosis':3,'avermitilis':4,'aeruginosa':8,'ecoli':9,'enterica':10,'typhi':10,'meningitis':10,'pertussis':10,'violaceum':9,'sicca':8,'pylori':9,'hepaticus':9,'succinogenus':8,'jejuni':7},
    	'tuberculosis' : {'faecalis':8,'pneumonia':8, 'monocytogenes':8, 'subtilis':8, 'bovis':2,'tuberculosis':0,'paratuberculosis':3,'avermitilis':4,'aeruginosa':8,'ecoli':9,'enterica':10,'typhi':10,'meningitis':10,'pertussis':10,'violaceum':9,'sicca':8,'pylori':9,'hepaticus':9,'succinogenus':8,'jejuni':7},
		'paratuberculosis' : {'faecalis':7,'pneumonia':7, 'monocytogenes':7, 'subtilis':7, 'bovis':3,'tuberculosis':3,'paratuberculosis':0,'avermitilis':2,'aeruginosa':7,'ecoli':8,'enterica':9,'typhi':9,'meningitis':9,'pertussis':9,'violaceum':8,'sicca':7,'pylori':8,'hepaticus':8,'succinogenus':7,'jejuni':6},
		'avermitilis' : {'faecalis':6,'pneumonia':6, 'monocytogenes':6, 'subtilis':6, 'bovis':3,'tuberculosis':3,'paratuberculosis':2,'avermitilis':0,'aeruginosa':6,'ecoli':7,'enterica':8,'typhi':8,'meningitis':8,'pertussis':8,'violaceum':7,'sicca':6,'pylori':7,'hepaticus':7,'succinogenus':6,'jejuni':5},
		'aeruginosa' : {'faecalis':8,'pneumonia':8, 'monocytogenes':8, 'subtilis':8, 'bovis':8,'tuberculosis':8,'paratuberculosis':7,'avermitilis':6,'aeruginosa':0,'ecoli':3,'enterica':4,'typhi':4,'meningitis':6,'pertussis':6,'violaceum':5,'sicca':4,'pylori':7,'hepaticus':7,'succinogenus':6,'jejuni':5},
		'ecoli' : {'faecalis':9,'pneumonia':9, 'monocytogenes':9, 'subtilis':9, 'bovis':9,'tuberculosis':9,'paratuberculosis':8,'avermitilis':7,'aeruginosa':3,'ecoli':0,'enterica':3,'typhi':3,'meningitis':7,'pertussis':7,'violaceum':6,'sicca':5,'pylori':8,'hepaticus':8,'succinogenus':7,'jejuni':6},
		'enterica' : {'faecalis':10,'pneumonia':10, 'monocytogenes':10, 'subtilis':10, 'bovis':10,'tuberculosis':10,'paratuberculosis':9,'avermitilis':8,'aeruginosa':4,'ecoli':3,'enterica':0,'typhi':2,'meningitis':8,'pertussis':8,'violaceum':7,'sicca':6,'pylori':9,'hepaticus':9,'succinogenus':8,'jejuni':7},
		'typhi' : {'faecalis':10,'pneumonia':10, 'monocytogenes':10, 'subtilis':10, 'bovis':10,'tuberculosis':10,'paratuberculosis':9,'avermitilis':8,'aeruginosa':4,'ecoli':3,'enterica':2,'typhi':0,'meningitis':8,'pertussis':8,'violaceum':7,'sicca':6,'pylori':9,'hepaticus':9,'succinogenus':8,'jejuni':7},
		'meningitis' : {'faecalis':10,'pneumonia':10, 'monocytogenes':10, 'subtilis':10, 'bovis':10,'tuberculosis':10,'paratuberculosis':9,'avermitilis':8,'aeruginosa':6,'ecoli':7,'enterica':8,'typhi':8,'meningitis':0,'pertussis':2,'violaceum':3,'sicca':4,'pylori':9,'hepaticus':9,'succinogenus':8,'jejuni':7},
		'pertussis' : {'faecalis':10,'pneumonia':10, 'monocytogenes':10, 'subtilis':10, 'bovis':10,'tuberculosis':10,'paratuberculosis':9,'avermitilis':8,'aeruginosa':6,'ecoli':7,'enterica':8,'typhi':8,'meningitis':2,'pertussis':0,'violaceum':3,'sicca':4,'pylori':9,'hepaticus':9,'succinogenus':8,'jejuni':7},
		'violaceum' : {'faecalis':9,'pneumonia':9, 'monocytogenes':9, 'subtilis':9, 'bovis':9,'tuberculosis':9,'paratuberculosis':8,'avermitilis':7,'aeruginosa':5,'ecoli':6,'enterica':7,'typhi':7,'meningitis':3,'pertussis':3,'violaceum':0,'sicca':3,'pylori':9,'hepaticus':9,'succinogenus':8,'jejuni':7},
		'sicca' : {'faecalis':8,'pneumonia':8, 'monocytogenes':8, 'subtilis':8, 'bovis':8,'tuberculosis':8,'paratuberculosis':7,'avermitilis':6,'aeruginosa':4,'ecoli':5,'enterica':6,'typhi':6,'meningitis':4,'pertussis':4,'violaceum':3,'sicca':0,'pylori':7,'hepaticus':7,'succinogenus':6,'jejuni':5},
		'pylori' :  {'faecalis':9,'pneumonia':9, 'monocytogenes':9, 'subtilis':9, 'bovis':9,'tuberculosis':9,'paratuberculosis':8,'avermitilis':7,'aeruginosa':7,'ecoli':8,'enterica':9,'typhi':9,'meningitis':9,'pertussis':9,'violaceum':8,'sicca':7,'pylori':0,'hepaticus':2,'succinogenus':3,'jejuni':4},
		'hepaticus' :  {'faecalis':9,'pneumonia':9, 'monocytogenes':9, 'subtilis':9, 'bovis':9,'tuberculosis':9,'paratuberculosis':8,'avermitilis':7,'aeruginosa':7,'ecoli':8,'enterica':9,'typhi':9,'meningitis':9,'pertussis':9,'violaceum':8,'sicca':7,'pylori':2,'hepaticus':0,'succinogenus':3,'jejuni':4},
		'succinogenus' :{'faecalis':8,'pneumonia':8, 'monocytogenes':8, 'subtilis':8, 'bovis':8,'tuberculosis':8,'paratuberculosis':7,'avermitilis':6,'aeruginosa':6,'ecoli':7,'enterica':8,'typhi':8,'meningitis':8,'pertussis':8,'violaceum':7,'sicca':6,'pylori':3,'hepaticus':3,'succinogenus':0,'jejuni':3},
		'jejuni' : {'faecalis':7,'pneumonia':7, 'monocytogenes':7, 'subtilis':7, 'bovis':7,'tuberculosis':7,'paratuberculosis':6,'avermitilis':5,'aeruginosa':5,'ecoli':6,'enterica':7,'typhi':7,'meningitis':7,'pertussis':7,'violaceum':6,'sicca':5,'pylori':4,'hepaticus':4,'succinogenus':3,'jejuni':0},
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
