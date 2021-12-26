# Implementation of a decision tree from scratch using classes in Python.

class Tree:
    def __init__(self):
        self.root = Node()

class Node:
    def __init__(self):
        self.data = None             # dataset rows held at that node
        self.unusedAttrs = None      # indices of attrs we haven't split by
        self.splitAttr = None        # the attr that giving the best split
        self.splitPt = None          # the split point used with the splitAttr
        self.left = None             # left child
        self.right = None            # right child
        self.label = None            # IF LEAF: the categorisation of rows here 
                                     # - true or false HeartDisease?
