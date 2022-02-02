# SparseStruct

# LabeledSparseStruct

This repository provides a reference implementation of **LabeledSparseStruct** as described in the paper:

The Geometrical Shapes of Violence: Predicting and Explaining Terrorist Operations Through Graph Embeddings

Gian Maria Campedelli, Janet Layne, Jack Herzoff, and Edoardo Serra.

The LabeledSparseStruct algorithm generates structural node representations for an undirected, weighted graph with a single label for each node. The algorithm currently only works
for a dataset formatted exactly as the dataset used for the paper. A general program is in development.
The algorithm will also be amended shortly to accept directed graphs along with an arbitrary number of node labels. 

## Usage

### Input

The supported input format is currently only the dataset used for the publication. A version for general input is in development.

### Example

To run *LabeledSparseStruct* :<br/>
	``python LabeledSparseStruct.py``



### Output

Output will be of length *n x m* for a graph with *n* vertices and *m* dimensions of the format:

    dim1 dim2 ... dimn
