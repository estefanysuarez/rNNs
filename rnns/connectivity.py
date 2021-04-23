
import random
import numpy as np

def upscale_rnn(original_network, fraction_intrinsic, neuron_density=None,\
              return_mask=False, intrinsic_sparsity=0.9, extrinsic_sparsity=0.2,\
              allow_self_conns=True):
    """
    This method was adapted from https://github.com/AlGoulas/bio2art.

    Generate scaled_network from a biological neural network (connectome).
    This operation allows the contruction of artiificial neural networks
    with recurrent matrices that obey the topology and weight strength
    contraints of a biological neural network.

    Input
    -----
    original_network: (N,N) ndarray
        If the network is directed, rows should represent source regions, and
        columns target regions.

    neuron_density: (N,) ndarray, default None
        N corresponds the number of brain regions in original_network.
        Each entry of  neuron_density[i] denotes the number of neurons in region
        i. NOTE: if None (default) then one neuron will be assigned to each
        region will be assigned one.

    fraction_intrinsic: float (0 1]
        Percentage of the strength of the outgoing connections of a region to
        be used as the total strength of the intrinsic weights. If None, and
        self connections exist in original_network (i.e., the diagonal is
        different from zero), then self-connections are used as the total
        strength of the intrinsic weights.

    extrinsic_sparsity: float (0 1], default 0.2
        The percentage of all possible target neurons for each source neuron
        to form connections with.
        Note that at least 1 neuron will function as target in case that the
        resulting percentage is <1.
        This parameter can be used to make the sparisty of scaled_network vary
        around the density dictated by the actual biological connectomes.
        Note that this parameter is meaningful only if at least one region
        has more than 1 neuron, that is, for some i, neuron_density[i]>1.

    intrinsic_sparsity: float (0 1], default 1.
        Same as extrinsic_sparsity, but for the within-region/intrinsic
        connections.

    allow_self_conn: bool, default True
        Specify if the diagonal entries (denoting self-to-self neuron
        connections) should be kept of or not.

    Output
    ------
    original_network: ndrarray of shape (N,N)
        The actual biological neural network that was used, with no
        modificiations/scaling (see data_name for N)

    scaled_network: ndarray of shape (M,M)
        The rescaled neural network.
        (M is bound to the parameter neuron_density)

    region_neuron_ids: list of lists of int
        List of lists for tracking the neurons of the scaled_network network.
        region_neuron_ids[1] contains a list with integers that denote the
        neurons of region 1 in scaled_network as
        scaled_network[region_neuron_ids[1], region_neuron_ids[1]]

    """
    # if neuron_density is not specified then populate each region with 1
    # neuron
    if neuron_density is None:
        neuron_density = np.ones((original_network.shape[0]), dtype=int)

    n_neurons = np.sum(neuron_density).astype(int) # total number of neurons

    if(neuron_density.shape[0] != original_network.shape[0]):
        print(f'Size of neuron_density must be equal to the number of brain \
                regions in connectome:{original_network.shape[0]}')
        return

    # list of neuron ids per brain region
    sections = [np.sum(neuron_density[:i]).astype(int) for i in range(1, len(neuron_density))]
    neuron_ids_per_roi = np.split(np.arange(n_neurons), sections)

    # sum of outgoing weights for each region - used for calculation of
    # intrinsic and extrinsic weights
    out_strength = np.sum(original_network, axis=1)

    # initialize the neuron to neuron connectivity matrix
    scaled_network = np.zeros((n_neurons, n_neurons))

    if return_mask: mask = np.zeros((n_neurons, n_neurons)).astype(int)

    # start populating the neuron-toneuron connectivity matrix
    # by region
    for source_roi in range(original_network.shape[0]):

        # intrinsic connectivity
        source_neurons = neuron_ids_per_roi[source_roi]

        connected = False
        while not connected:
            intrinsic_conn = np.vstack([np.random.binomial(1, intrinsic_sparsity, len(source_neurons)) for _ in source_neurons]).astype(int)
            if np.sum(intrinsic_conn) > 0:
                connected = True

        if fraction_intrinsic is None and original_network[source_roi,source_roi] > 0:
            intrinsic_wei = original_network[source_roi,source_roi]/np.sum(intrinsic_conn)
        else:
            intrinsic_wei = (fraction_intrinsic*out_strength[source_roi])/np.sum(intrinsic_conn)
        scaled_network[np.ix_(source_neurons, source_neurons)] = intrinsic_conn * intrinsic_wei

        if return_mask: mask[np.ix_(source_neurons, source_neurons)] = 1

        # extrinsic connectivity
        target_rois = np.nonzero(original_network[source_roi,:] > 0)[0]
        for target_roi in target_rois:

            if target_roi != source_roi:
                target_neurons = neuron_ids_per_roi[target_roi]

                connected = False
                while not connected:
                    extrinsic_conn = np.vstack([np.random.binomial(1, extrinsic_sparsity, len(target_neurons)) for _ in source_neurons])
                    if np.sum(extrinsic_conn) > 0:
                        connected = True

                extrinsic_wei = out_strength[source_roi]/np.sum(extrinsic_conn)
                scaled_network[np.ix_(source_neurons, target_neurons)] = extrinsic_conn * extrinsic_wei

                if return_mask: mask[np.ix_(source_neurons, target_neurons)] = 1

    # delete self-connections
    if not allow_self_conns: np.fill_diagonal(scaled_network, 0)

    if return_mask:
        return neuron_ids_per_roi, scaled_network, mask
    else:
        return neuron_ids_per_roi, scaled_network
