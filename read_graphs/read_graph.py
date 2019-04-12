import re
import argparse
import textwrap
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from collections import defaultdict


def seqs_to_edges(fasta):
	'''
	Map contigs to corresponding edges
	Input:
		- .fasta file
	Return:
		- a mapping between contigs and edges
	'''
	seq_edge_maps = defaultdict(list)
	current_id = None
	current_pt = None

	with open(fasta, 'r') as f:
		for line in f:
			# remove special characters of each line
			line = line.strip()
			if (line.startswith('>')):
				infos = line.split('_')
				current_id = int(infos[1])
				current_pt = int(infos[2])
				seq_edge_maps[current_id].append('')
			else:
				seq_edge_maps[current_id][current_pt] += line

	return seq_edge_maps


def extract_info(edges, seq):
	'''
	Get everything embedded within an edge
	Input:
		- edge data
		- sequence data
	Return:
		- mappings of edge and all data
	'''
	node_data = defaultdict(list)

	for edge in edges:
		src = str(edge[0])
		dst = str(edge[1])
		attr = edge[2]

		nums = re.findall(r'-?\d+', attr['label'])
		color = attr['color']
		idx = int(nums[0])                    # index
		cover = int(nums[2])                  # coverage
		reverse = False if idx > 0 else True  # orientation

		if color == '"red"':
			repeat = int(attr['penwidth'])
		else:
			repeat = 1  # no repeat by default

		seqs = seq[abs(idx)][0]  # actual sequence
		data = {'reverse': reverse, 'cover': cover, 'repeat': repeat, 'length': len(seqs), 'seq': seqs}
		node_data[(src, dst)].append(data)

	return node_data


def embed_nodes(graph, attrs):
	'''
	Embed nodes of the line graph
	Input:
		- line graph
		- original edge information
	'''
	nodes = graph.nodes()
	for node in nodes:
		attr = {node: attrs[(node[0], node[1])][node[2]]}
		nx.set_node_attributes(graph, attr)



def write_nodedata(nodes, file):
	'''
	Write nodes and corresponding attributes to file
	Input:
		- node data
		- output file path
	'''
	with open(file, 'w') as f:
		f.write('# >[ID]\t[reverse]\t[coverage]\t[repeat]\t[length]\n')
		f.write('# [sequence]\n')
		
		for node in nodes:
			f.write('>' + str(node[0]) + '\t')
			f.write(str(node[1]['reverse']) + '\t')
			f.write(str(node[1]['cover']) + '\t')
			f.write(str(node[1]['repeat']) + '\t')
			f.write(str(node[1]['length']) + '\n')
			# formatting the sequence
			seq = textwrap.fill(node[1]['seq'], width=100)
			f.write(seq + '\n')



def main(args):
	name_in  = args.input
	name_out = args.output


	dot_file = name_in[0]  # [0]: graph_after_rr.gv
	seq_file = name_in[1]  # [1]: repeat_graph_edges.fasta
	# dmp_file = name_in[2]  # [2]: repeat_graph_dump

	node_file = name_out[0]
	edge_file = name_out[1]

	# read the repeat graph
	# rep_graph = nx.drawing.nx_agraph.read_dot(name_in)
	rep_graph = nx.drawing.nx_pydot.read_dot(dot_file)

	# visualization
	# nx.draw_networkx(rep_graph)
	# plt.show()

	# {ID: sequence}
	seq_data = seqs_to_edges(seq_file)

	rep_line_graph = nx.line_graph(rep_graph)

	# original edge attributes
	# {'color' = [c], 'label' = [ID][length][coverage], 'penwidth' = [w]}
	ori_edge_attr = rep_graph.edges.data()
	new_node_attr = extract_info(ori_edge_attr, seq_data)

	embed_nodes(rep_line_graph, new_node_attr)

	rep_line_graph = nx.convert_node_labels_to_integers(rep_line_graph)

	# write results to files
	write_nodedata(rep_line_graph.nodes.data(), node_file)
	nx.write_edgelist(rep_line_graph, edge_file, data=False)


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--input', nargs=2, help='path to the input file')
	ap.add_argument('-o', '--output', nargs=2, help='path to the output file')
	args = ap.parse_args()
	main(args)