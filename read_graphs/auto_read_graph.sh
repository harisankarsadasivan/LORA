#!/bin/bash

# This is just a simple script to read all available repeat graphs

# Always make sure the path is correctly specified before execution
for d in ../data/flye_output/*;
do
        f=$(basename $d)
        dot_file=$d/20-repeat/graph_after_rr.gv
        seq_file=$d/20-repeat/repeat_graph_edges.fasta
        node_file=$d/20-repeat/node_list
        edge_file=$d/20-repeat/edge_list

        echo "Now reading graph from $f..."
        python3 read_graph.py -i $dot_file $seq_file -o $node_file $edge_file
done
