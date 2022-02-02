# LabeledSparseStruct

This repository provides a reference implementation of **LabeledSparseStruct** as described in the paper:

The Geometrical Shapes of Violence: Predicting and Explaining Terrorist Operations Through Graph Embeddings

Gian Maria Campedelli, Janet Layne, Jack Herzoff, and Edoardo Serra.

The LabeledSparseStruct algorithm generates structural node representations for an undirected, weighted graph with a single label for each node. The algorithm will be amended shortly to accept directed graphs along with an arbitrary number of node labels. An option for graph classification will also be added.

## Usage

### Input

The supported input format is an edgelist with weights and labels:<br/>

    node1_id  node2_id  node1_label  node2_label  weight   

### Example

To run *LabeledSparseStruct* on Mutag dataset, execute the following command from the project home directory:<br/>
	``python src/main.py --input graph/mutag.edgelist --output emb/mutag.csv``

### Help

usage:<br/>
	``python src/main.py --help``

### Output

Output will be of length *n* for a graph with *n* vertices and of the format:

    node_id dim1 dim2 ... dimn
