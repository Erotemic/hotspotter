import numpy as np
from hotspotter.other.ConcretePrintable import DynStruct
def L2_squared(vec1, vec2):
    'Returns squared distance between two vectors'
    return np.sum((vec1 - vec2)**2)

class PivotElement(DynStruct):
    def __init__(self, element):
        super(PivotElement, self).__init__()
        self.right_child = None
        self.left_child  = None
        self.element     = element

def printDBG(msg):
    pass

def ProximityTree(S, tau=10, delta=L2_squared):
    ''' Recursive construction of a Proximity Tree S:
    a set of data elements from which to construct the tree tau, the number
    of data elements to sample before splitting delta, a distance function'''
    # Base case for recursion
    printDBG('Input size: len(S)=%r < tau=%r' % (len(S), tau))
    if len(S) < tau: return PivotElement('Leaf')
    # Otherwise pick a tau random elements
    np.random.shuffle(S)
    S_hat = S[0:tau]
    # And then choose one of those as a pivot. 
    np.random.shuffle(S_hat)
    P = S[0]
    new_node = PivotElement(P)
    # Find the median distance from the selection to the pivot
    D  = [delta(x, P) for x in S_hat]
    dt = np.median(D)
    # Partition S into left and right subsets based on the median distance
    S_le = [x for x in S if delta(x, P) <= dt]
    S_gt = [x for x in S if delta(x, P) >  dt]
    printDBG('Partitioning into len(S_le)=%r ' % (len(S_le)))
    printDBG('Partitioning into len(S_gt)=%r ' % (len(S_gt)))
    # Recursive Step
    new_node.left_child  = ProximityTree(S_le, tau, delta)
    new_node.right_child = ProximityTree(S_gt, tau, delta)
    return new_node

if __name__ == '__main__':
    data = np.random.rand(1000,128)
    S     = data
    tau   = 10
    delta = L2_squared
    root  = ProximityTree(S, tau, delta)
    print_exclude_aug=['element']
    #print_exclude_aug=[]
    root.printme2(type_bit=False, print_exclude_aug=print_exclude_aug)
