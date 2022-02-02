"""
    Reference implementation for WeightedSparseStruct

    Author: Janet Layne, Edoardo Serra

"""
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import labeledsparsestruct
import sparsestructloader

def parse_args():
	# Parses SparseStruct arguments

	parser = argparse.ArgumentParser(description="Run WeightedSparseStruct.")

	parser.add_argument('--input', nargs='?', default='graph/mutag.csv',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/mutag.csv',
	                    help='Output embedding path')

	parser.add_argument('--svd', type=int, default=20,
	                    help='Number of SVD components. Default = 20')
        




	return parser.parse_args()


def main(args):
    l = sparsestructloader.loader()
    l.read(args.input)
    G = l.G
    labs = l.l
    emb = labeledsparsestruct.fast(G, labs)
    svd = TruncatedSVD(n_components=min(args.svd,emb.shape[1]-1),n_iter=10,random_state=1)
    emb1=svd.fit_transform(emb)
    l.storeEmb(args.output, emb1)


if __name__ == "__main__":
	args = parse_args()
	main(args)