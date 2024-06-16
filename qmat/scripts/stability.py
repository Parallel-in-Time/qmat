#!/usr/bin/env python
from qmat import genQCoeffs

def main():
    nodes, weights, Q = genQCoeffs("coll", nNodes=3, nodeType="LEGENDRE", quadType="GAUSS")
    print("node :", nodes)
    print("weights :", weights)
    print("Q :")
    print(Q)

if __name__ == "__main__":
    main()