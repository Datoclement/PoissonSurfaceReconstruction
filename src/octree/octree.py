import numpy as np
from tqdm import tqdm

# Base Functions
def base(vec, func):
    return np.prod(func(vec))

def f1(vec):
    return np.logical_and(vec<+.5, vec>-.5)

def f2(vec):
    return np.maximum(1.-np.abs(vec), 0.)

def f3(x):
    abx = np.abs(x)
    b1 = abx < 0.5
    b2 = abx < 1.5
    vals = (1.5-abx)**2./2. * b2 * (1.-b1) \
        + (-x**2. + 3./4.) * b1
    return vals

def ddf3(x):
    abx = np.abs(x)
    b1 = abx < 0.5
    b2 = abx < 1.5
    return -2. * b1 + b2 * (1.-b1)

octant_bits = np.array(
        [[(i&(1<<j))>0 for j in range(3)] for i in range(8)],
        dtype=int)

def test():

    # base function test
    import matplotlib.pyplot as plt
    l, h = intv = (-2, +2)
    prec = 1000
    x = np.linspace(l,h,prec*(h-l)+1)
    plt.plot(x,f1(x),'b',label='0th-order spline')
    plt.plot(x,f2(x),'g',label='1st-order spline')
    plt.plot(x,f3(x),'r',label='2nd-order spline')
    plt.legend()
    plt.show()

def poisson_surface_reconstruction(depth, tempt):

    num_voxels = 2**depth

    def reconstructor(points, normals):

        def init_smoothing_filter():

            def smoothing_filter(scaler):
                def scaled_filter(x):
                    return np.prod(f3(scaler(x)))
                def dd_scaled_filter(x):
                    return np.prod(ddf3(scaler(x)))
                return scaled_filter, dd_scaled_filter

            radius = 1.5

            return smoothing_filter, radius

        filter, radius = init_smoothing_filter()

        def build_octree():

            # adjust the bounding box
            lows = np.amin(points, axis=0)
            highs = np.amax(points, axis=0)
            size = np.amax(highs-lows) * (1.+3./num_voxels)
            voxel_size = size / num_voxels
            means = (lows + highs)/2.
            lows = means - size/2.
            highs = means + size/2.
            print('lows',lows)
            print('highs',highs)
            print('voxel size',voxel_size)

            nodes = list()
            index_to_leaf = dict()

            def create_node(
                parent=dict(
                    depth=-1,
                    width=2.*size,
                    center=means),
                octant=np.array([.5]*3)):
                pcenter = np.array(parent['center'])
                node = dict(
                    depth=parent['depth']+1,
                    width=.5*parent['width'],
                    center=tuple(pcenter+(2*octant-1)*.25*parent['width']),
                    is_lead=False)
                if node['depth'] >= depth:
                    node['normal'] = np.zeros(3)
                    node['is_leaf'] = True
                nodes.append(node)
                return node

            octree = create_node()

            def min_max_normalise(point):
                return np.array((point - lows) / voxel_size)

            def get_closest_vertex(relative_coord):
                return np.round(relative_coord).astype(int)

            def init_node(node):
                nodes.remove(node)
                for octant_bit in octant_bits:
                    node[tuple(octant_bit)] = create_node(node, octant_bit)

            def insert_leaf(index):
                octants = list()
                tmpid = index
                for i in range(depth):
                    octants.append(tmpid % 2)
                    tmpid = tmpid // 2
                current_node = octree
                for octant in reversed(octants):
                    octant = tuple(octant)
                    if octant not in current_node:
                        init_node(current_node)
                    current_node = current_node[octant]
                current_node['index'] = index
                index_to_leaf[tuple(index)] = current_node

            def get_surroundings(vertex):
                indices = np.expand_dims(vertex, axis=0)+octant_bits-1
                surroundings = list()
                for index in indices:
                    if tuple(index) not in index_to_leaf:
                        insert_leaf(index)
                    surroundings.append(index_to_leaf[tuple(index)])
                return surroundings

            def get_trilinears(relative_coord, closest_vertex):
                delta = np.expand_dims(relative_coord - closest_vertex, axis=0)
                trilinears = np.prod(
                        octant_bits * (.5-delta) + (1-octant_bits) * (.5+delta),
                        axis=-1)
                return trilinears

            def get_surroundings_and_trilinears(location):
                relative_coord = min_max_normalise(point)
                closest_vertex = get_closest_vertex(relative_coord)
                surroundings = get_surroundings(closest_vertex)
                trilinears = get_trilinears(relative_coord, closest_vertex)
                return surroundings, trilinears

            def add_point(point, normal):
                for node, weight in zip(*get_surroundings_and_trilinears(point)):
                    node['normal'] += weight * normal

            for point, normal in zip(points, normals):
                add_point(point, normal)

            true_radius = radius / num_voxels * size
            def aux_range_search(node, location):
                result = list()
                dist = np.amax(np.abs(location - node['center']))
                if node['depth']==depth:
                    if dist < true_radius:
                        result.append(node)
                elif dist < true_radius + node['width']:
                    for octant_bit in octant_bits:
                        octant_bit = tuple(octant_bit)
                        if octant_bit in node:
                            result.extend(aux_range_search(node[octant_bit], location))
                return result
            def range_search(location):
                location = np.array(location)
                return aux_range_search(octree, location)

            fields = ['center', 'gradient', 'divergence', 'v', 'chi']
            fields_transform = {
                    'center':['x', 'y', 'z'],
                    'gradient':['nx', 'ny', 'nz']
                }
            def octree_fields():
                attribs = list(set(nodes[0].keys()).intersection(fields))
                values = dict(zip(attribs, [[] for a in attribs]))
                for node in nodes:
                    for attrib in attribs:
                        values[attrib].append(node[attrib])
                for attrib in attribs:
                    values[attrib] = np.array(values[attrib])
                for k in fields_transform:
                    for i, v in enumerate(fields_transform[k]):
                        values[v] = values[k][:,i]
                    del values[k]
                return values

            def real_to_voxel_scaling(x):
                return x/voxel_size

            return octree, nodes, real_to_voxel_scaling, range_search, get_surroundings_and_trilinears, octree_fields

        octree, nodes, real_to_voxel_scaling, range_search, sur_and_tril, octree_fields = build_octree()
        filter, dd_filter = filter(real_to_voxel_scaling)

        def conv_like_op():
            neighbours = dict()
            print('nodes len',len(nodes))

            print('neighbour searching...')
            for i, node in tqdm(enumerate(nodes)):
                neighbours[i] = range_search(node['center'])

            def iterate_over_nodes(func):
                print('{} is being executed...'.format(func.name))
                for i, node in tqdm(enumerate(nodes)):
                    func(node, neighbours[i])
            return iterate_over_nodes
        op_executor = conv_like_op()

        def compute_gradient():
            def f(node, neighbours):
                node['gradient'] = np.zeros(3)
                for neighbour in neighbours:
                    node['gradient'] += filter(np.array(neighbour['center'])-np.array(node['center'])) * neighbour['normal']
            f.name = 'gradient computation'
            return f
        op_executor(compute_gradient())

        def compute_divergence():
            def f(node, neighbours):
                node['divergence'] = 0.
                for neighbour in neighbours:
                    delta = np.array(neighbour['center'])-np.array(node['center'])
                    if (delta < 1e-10).all():
                        continue
                    dgrad = np.array(neighbour['gradient'])-np.array(node['gradient'])
                    n_delta = np.sum(delta**2.)**.5
                    node['divergence'] += filter(n_delta) * np.sum(delta * dgrad / n_delta)
            f.name = 'divergence computation'
            return f
        op_executor(compute_divergence())

        def project_on_kernel_space():
            def f(node, neighbours):
                node['v'] = 0.
                for neighbour in neighbours:
                    node['v'] += filter(np.array(neighbour['center'])-np.array(node['center'])) * neighbour['divergence']
            f.name = 'projection on finite basis'
            return f
        op_executor(project_on_kernel_space())
        v = np.array([ node['v'] for node in nodes ])

        def dxxf(x, ax):
            """
                x is of dimension 1
            """
            nonax = list(range(3))
            nonax.remove(ax)
            return filter(x[nonax]) * dd_filter(x[ax])
        def ddf_f_product(node1, node2):
            """
                approximate dot product of functions
            """
            c1 = np.array(node1['center'])
            c2 = np.array(node2['center'])
            return -6. * filter(c2-c1) + np.sum([dxxf(c2-c1, i) for i in range(3)])
        from scipy import sparse as sp
        import scipy.sparse.linalg as splna
        for id, node in enumerate(nodes):
            node['id'] = id
        def get_laplacian_matrix():
            # construct a sparse matrix
            data = list()
            rows = list()
            cols = list()

            def add_to_list(n1, n2):
                data.append(ddf_f_product(n1, n2))
                rows.append(n1['id'])
                cols.append(n2['id'])

            def compute_dot_product():
                def f(node, neighbours):
                    add_to_list(node, node)
                    for neighbour in neighbours:
                        add_to_list(node, neighbour)
                f.name = 'laplacian computation'
                return f
            op_executor(compute_dot_product())
            return sp.coo_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes))).tocsr()

        L = get_laplacian_matrix()

        print('solving Poisson equation...')
        chis = splna.spsolve(L, v)
        for chi, node in zip(chis, nodes):
            node['chi'] = chi

        print('computing the mean of chi...')
        total_chi = 0.
        for point in tqdm(points):
            for sur, tril in zip(*sur_and_tril(point)):
                total_chi += tril * sur['chi']
        mean_chi = total_chi / points.shape[0]

        print('mean_chi', mean_chi)
        volumes = np.zeros(shape=(2**depth,)*3)

        return octree_fields()

    return reconstructor

if __name__ == '__main__':
    test()
