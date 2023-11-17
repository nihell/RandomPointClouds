import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

def prune_contributions(contributions):

    total_ECP = dict()

    for a in contributions:
        total_ECP[a[0]] = total_ECP.get(a[0], 0) + a[1]

    # remove the contributions that are 0
    to_del = []
    for key in total_ECP:
        if total_ECP[key] == 0:
            to_del.append(key)
    for key in to_del:
        del total_ECP[key]
        
    return sorted(list(total_ECP.items()), key = lambda x: x[0])


def EC_at_bifiltration(contributions, f1, f2):
    return sum([c[1] for c in contributions if (c[0][0] <= f1) and
                                               (c[0][1] <= f2)])


def difference_ECP(ecp_1, ecp_2, dims, return_contributions = False):
    f1min, f1max, f2min, f2max = dims
    
    contributions = [((f1min, f2min), 0), ((f1max, f2max), 0)]
    
    contributions += ecp_1
    contributions += [(c[0], -1*c[1]) for c in ecp_2]
    
    contributions = [((f1min, f2min), 0)]+prune_contributions(contributions)+[((f1max, f2max), 0)]
    
    f1_list = sorted(set([c[0][0] for c in contributions]))
    f2_list = sorted(set([c[0][1] for c in contributions]))
    
    difference = 0
    
    for i, f1 in enumerate(f1_list[:-1]):
        delta_i = f1_list[i+1] - f1_list[i]
        for j, f2 in enumerate(f2_list[:-1]):
            delta_j = f2_list[j+1] - f2_list[j]
            
            difference += EC_at_bifiltration(contributions, f1, f2) * delta_i * delta_j
    
    if return_contributions:
        return difference, contributions
    else:
        return difference

def wasserstein_distance_ECP(contrib1, contrib2, internal_p=1):
    positive = []
    negative = []
    for c in contrib1:
        val = np.array(c[0])
        if len(val.shape)==0:           #compatibility with ecc
            val=val.reshape(-1)
        for i in range(np.abs(c[1])):

            if c[1]>0:
                positive.append(val)
            elif c[1]<0:
                negative.append(val)
    for c in contrib2:
        val = np.array(c[0])
        if len(val.shape)==0:           #compatibility with ess
            val=val.reshape(-1)
        for i in range(np.abs(c[1])):
            if c[1]>0:
                negative.append(val)
            elif c[1]<0:
                positive.append(val)

    cost = distance_matrix(positive,negative, p = internal_p)
    row_ind, col_ind = linear_sum_assignment(cost)

    return cost[row_ind, col_ind].sum()


def bottleneck_distance_ECP(contrib1, contrib2, internal_p=1, eps = 0.00001):
    positive = []
    negative = []
    for c in contrib1:
        val = np.array(c[0])
        if len(val.shape)==0:
            val=val.reshape(-1)
        for i in range(np.abs(c[1])):

            if c[1]>0:
                positive.append(val)
            elif c[1]<0:
                negative.append(val)
    for c in contrib2:
        val = np.array(c[0])
        if len(val.shape)==0:
            val=val.reshape(-1)
        for i in range(np.abs(c[1])):
            if c[1]>0:
                negative.append(val)
            elif c[1]<0:
                positive.append(val)

    cost = distance_matrix(positive,negative, p = internal_p)
    # binary search
    A = np.sort(cost.flatten())
    L = 0 
    R = len(A)-1
    while L < R:
        m = (L + R) // 2
        csr = csr_matrix(np.where(cost>A[m],0,1))
        matching = maximum_bipartite_matching(csr)
        matched = np.sum(matching!=-1)
        if matched < len(positive):
            L = m + 1
        else:
            R = m
    return A[L]



def plot_ECP(contributions, dims,
             this_ax=None, 
             colorbar=False, **kwargs):
    
    f1min, f1max, f2min, f2max = dims
    
    if this_ax == None:
        this_ax = plt.gca()
    
    f1_list = [f1min] + sorted(set([c[0][0] for c in contributions])) + [f1max]
    f2_list = [f2min] + sorted(set([c[0][1] for c in contributions])) + [f2max]
    
    Z = np.zeros((len(f2_list)-1, len(f1_list)-1))

    for i, f1 in enumerate(f1_list[:-1]):
        for j, f2 in enumerate(f2_list[:-1]):
            Z[j,i] = EC_at_bifiltration(contributions, f1, f2)
    
    # Plotting
    im = this_ax.pcolormesh(f1_list, f2_list, Z)
    
    if colorbar:
        plt.colorbar(im, ax=this_ax)
    
    return this_ax


