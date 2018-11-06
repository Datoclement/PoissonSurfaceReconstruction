import numpy as np

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

octant_bits = np.array(
        [[(i&(1<<j))>0 for j in range(3)] for i in range(8)],
        dtype=int)

def test():

    # base function test
    import matplotlib.pyplot as plt
    l, h = intv = (-4, +4)
    prec = 1000
    x = np.linspace(l,h,prec*(h-l)+1)
    plt.plot(x,f3(x),'r',label='f3')
    plt.plot(x,f2(x),'b',label='f2')
    plt.plot(x,f1(x),'k',label='f1')
    y = f1(x)
    n = 6
    half = prec//2
    for i in range(n):
        plt.plot(x,y,label='true_f'+str(i+1))
        cs = np.cumsum(y)
        y = 1/prec*(np.concatenate([cs[half:],np.zeros(half)+cs[-1]])\
            - np.concatenate([np.zeros(half),cs[:-half]]))
    plt.legend()
    plt.show()

def poisson_surface_reconstruction(depth):

    num_voxels = 2**depth

    def reconstructor(points, normals):

        def init_smoothing_filter():

            def smoothing_filter(x):
                return np.prod(f3(x))

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

            def add_point(point, normal):
                relative_coord = min_max_normalise(point)
                closest_vertex = get_closest_vertex(relative_coord)
                surroundings = get_surroundings(closest_vertex)
                trilinears = get_trilinears(relative_coord, closest_vertex)
                for node, weight in zip(surroundings, trilinears):
                    node['normal'] += weight * normal

            for point, normal in zip(points, normals):
                add_point(point, normal)

            true_radius = radius / num_voxels * size
            def range_search(location):
                location = np.array(location)
                def aux_range_search(node):
                    result = list()
                    dist = np.amax(np.abs(location - node['center']))
                    if node['depth']==depth:
                        if dist < true_radius:
                            result.append(node)
                    elif dist < true_radius + node['width']:
                        for octant_bit in octant_bits:
                            octant_bit = tuple(octant_bit)
                            if octant_bit in node:
                                result.extend(aux_range_search(node[octant_bit]))
                    return result
                return aux_range_search(octree)

            return octree, nodes, range_search

        octree, nodes, range_search = build_octree()

        def compute_gradient():
            print('nodes len',len(nodes))
            for i,node in enumerate(nodes):
                if (i+1) % 1000 == 0:
                    print('finished',i)
                node['gradient'] = np.zeros(3)
                relevants = range_search(node['center'])
                for relevant in relevants:
                    # print("relevant",relevant)
                    node['gradient'] += filter(np.array(relevant['center'])-np.array(node['center'])) * relevant['normal']
        compute_gradient()

        node_locs = list()
        node_grads = list()
        for node in nodes:
            node_locs.append(node['center'])
            node_grads.append(node['gradient'])

        node_locs = np.array(node_locs)
        node_grads = np.array(node_grads)
        print('np.amin(node_locs,axis=0)',np.amin(node_locs,axis=0))
        print('np.amax(node_locs,axis=0)',np.amax(node_locs,axis=0))
        # print('node_locs[:10]',node_locs[:10])
        print('node_grads[:10]',node_grads[:10])
        return node_locs, node_grads

        # compute_divergence()
        # project_on_kernel_space()
        # laplacian = get_laplacian_matrix()
        # scalars = use_poisson_solver()
    return reconstructor

if __name__ == '__main__':
    test()
