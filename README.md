# SparseStruct



This repository provides a general implementation of SparseStruct as described in the paper:
[Large-scale Sparse Structural Node Representation](https://ieeexplore.ieee.org/document/9377854)

Edoardo Serra, Mikel Joaristi, Alfredo Cuzzocrea

 but includes adaptations which allow for labeled nodes and weighted edges, a reference implementation of **LabeledSparseStruct**. Additionally, **ExplanationSparseStruct** is included, which generates importance scores for specific graph edges.  These are described in the paper:

[The geometrical shapes of violence: predicting and explaining terrorist operations through graph embeddings](https://academic.oup.com/comnet/article-abstract/10/2/cnac008/6564024)

Gian Maria Campedelli, Janet Layne, Jack Herzoff, and Edoardo Serra.



The SparseStruct algorithm generates node representations based upon the structural characteristics of node's neighborhoods. 
An implementation for directed graphs is in development. 

## Citing
If you make use of the SparseStruct or Explanation algorithm in your research, please cite the following paper:
```bibtex
@article{10.1093/comnet/cnac008,
    author = {Campedelli, Gian Maria and Layne, Janet and Herzoff, Jack and Serra, Edoardo},
    title = "{The geometrical shapes of violence: predicting and explaining terrorist operations through graph embeddings}",
    journal = {Journal of Complex Networks},
    volume = {10},
    number = {2},
    year = {2022},
    month = {04},
    issn = {2051-1329},
    doi = {10.1093/comnet/cnac008},
    url = {https://doi.org/10.1093/comnet/cnac008},
    eprint = {https://academic.oup.com/comnet/article-pdf/10/2/cnac008/43290087/cnac008.pdf},
}
```


## Undirected

This implementation works for undirected graphs. An implementation for directed graphs is in development.

### Usage
From the command line:
```bash
python SparseStruct.py --input --output --labels --weighted --stop --depth --rep  
```
where:
* --weighted is a Boolean indicator whether the input graph contains weighted edges
* --rep is the desired final dimensionality of the node representations
* --depth is the number of iterations, which corresponds to the depth of neighborhood exploration upon which each node's representation is based (note that if stop = True then the algorithm will run to convergence regardless of depth input)
For example:
```bash
python SparseStruct.py --input fileneame --output filename --labels filename --weighted False --stop False --depth 10 --rep 20  
```

#### Input

SparseStruct expects in input comma separated edgelist with headers in the form of: <br>
	noeID1, nodeID2, weight


#### Example

To run *LabeledSparseStruct* :<br/>
	``python LabeledSparseStruct.py``



#### Output

Output will be of length *n x m* for a graph with *n* vertices and *m* dimensions of the format:

    dim1 dim2 ... dimn

## DirectedSparseStruct

## ExplanationSparseStruct

This algorithm generates the explanation for classification of a SparseStruct node representation. This current implementation runs an Extra Trees Classifier on a graph classification task and returns importance values for each correctly classified edge in the graph.

## Usage

### Input

The supported input format is currently only the dataset used for the publication. A version for general input is in development. This algorithm runs LabeledSparseStruct and as the neighborhood is explored for each node, collects information about which features represent which edges being explored. As such, this is run as a standalone program, not as an addon to the LabeledSparseStruct algorithm. If no explanation is required, and only an embedding, LabeledSparseStruct is much faster due to the use of the Lime explainer for each node.

### Example

To run *SparseStructExplanation* :<br/>
	``python ExplanationSparseStruct.py``



### Output

Output is a nested dictionary with the group and timestamp as a tuple outer key, inner key is (node1_id, node2_id) and value is importance score.
