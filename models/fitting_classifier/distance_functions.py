
""" Distance functions """

def euclidean(x, y, v, m, mp=True):
    if v: print('SE'); plot_im(abs(x - y)**2, cmap=m)
    if mp: return abs(x - y)**2
    else: return np.sqrt(np.sum((x - y)**2))
              
def manhattan(x, y, v, m, mp=True):
    if v: print('AE'); plot_im(abs(x - y), cmap=m)
    if mp: return abs(x - y)
    else: return np.sum(abs(x - y))
              
def KL(x, y, v, m, mp=True):
    eps = 0.0000001
    if v: print('KL'); plot_im((x + eps) * np.log((x + eps) / (y + eps)), cmap=m)
    if mp: return (x + eps) * np.log((x + eps) / (y + eps))
    else: return np.sum((x + eps) * np.log((x + eps) / (y + eps)))

def minkowski(x, y, v, m, nroot=3, mp=True):
    def nth_root(x, n_root): return x ** (1/float(n_root))
    if v: print('ME%d' %nroot); plot_im(abs(x - y) ** nroot, cmap=m)
    if mp: return abs(x - y) ** nroot
    else: return nth_root(np.sum(abs(x - y) ** nroot), nroot)

def ncc(x, y, v, m, mp=True): 
    if v: print('NCC'); plot_im(((x - np.mean(x)) * (y - np.mean(y))) / ((x.size - 1) * np.std(x) * np.std(y)), cmap=m)
    if mp: return ((x - np.mean(x)) * (y - np.mean(y))) / ((x.size - 1) * np.std(x) * np.std(y))
    else: return np.sum((x - np.mean(x)) * (y - np.mean(y))) / ((x.size - 1) * np.std(x) * np.std(y))

def cosine(x, y, v=None, m=None):
    def square_rooted(x): return np.sqrt(np.sum([a*a for a in x]))
    num = np.sum(x * y)
    den = np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2))
    return num/den
 
def cosine_(x, y, v=None, m=None):
    import scipy
    return scipy.spatial.distance.cosine(x, y, w=None)
 
def jaccard(x, y, v=None, m=None):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality / union_cardinality

def norm(x, y): return np.linalg.norm(x - y)

def hamming(x, y, v=None, m=None):
    assert len(x) == len(y)
    return np.sum(x != y)
