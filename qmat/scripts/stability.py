#!/usr/bin/env python
from qmat import genQCoeffs

def main():
    nodes, weights, Q = genQCoeffs("coll", nNodes=4, nodeType="LEGENDRE", quadType="RADAU-RIGHT")
    print("node :", nodes)
    print("weights :", weights)
    print("Q :")
    print(Q)

if __name__ == "__main__":
    main()