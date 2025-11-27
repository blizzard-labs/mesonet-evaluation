import numpy as np
import time
import os.path
import scipy 
import pickle as pickle
import brian2 as b
from struct import unpack
from brian2 import *
from brian2tools import *
import os


MNIST_data_path = 'mnist/'     #specify where your data is

#------------------------------------------------------------------------------ 
# functions
#------------------------------------------------------------------------------  
def get_labeled_data(picklename, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename, 'rb'))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]
    
        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in range(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]
            
        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data

def normalize_weights(connection,norm):
    n_input = connection.source.N
    n_e = connection.target.N
    temp_conn = np.copy(connection.w)
    temp_conn = temp_conn.reshape((n_input,n_e))
    colSums = np.sum(temp_conn, axis = 0)
    colFactors = norm/colSums
    for j in range(n_e):
        temp_conn[:,j] *= colFactors[j]
    connection.w = temp_conn.reshape((n_input*n_e))
    return connection

def get_new_assignments(result_monitor, input_numbers):
    #print result_monitor.shape
    n_e = result_monitor.shape[1]
    assignments = np.ones(n_e) * -1 # initialize them as not assigned
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e    
    rate = [0] * n_e    
    for j in range(10):
        num_inputs = len(np.where(input_nums == j)[0])
        if num_inputs > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_inputs
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j 
    return assignments

def get_new_assignments_for_10(result_monitor, input_numbers, power):
    #print result_monitor.shape
    #print input_numbers.shape
    n_e = result_monitor.shape[1]
    assignments = np.zeros((n_e,10)) # initialize them as not assigned
    rate = np.zeros((10,n_e))
    count = np.zeros((10))
    for n in range(input_numbers.shape[0]):
        rate[input_numbers[n],:] += result_monitor[n,:]
        count[input_numbers[n]] += 1
    for n in range(10):
        rate[n,:] = rate[n,:] / count[n]   
    for n in range(n_e):
        rate_power = np.power(rate[:,n], power)
        if np.sum(rate_power) > 0:
            assignments[n,:] = [rate_power[i]/np.sum(rate_power) for i in range(10)]
    return assignments

def get_recognized_number_proportion(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    summed_proportion = summed_rates/ np.sum(summed_rates)
    return summed_proportion

def get_recognized_number_proportion_for_10(assignments_for_10, spike_rates):
    summed_rates = [0] * 10
    for i in range(10):
        summed_rates[i] = np.sum(spike_rates * assignments_for_10[:,i]) / len(spike_rates)
    summed_proportion = summed_rates/ np.sum(summed_rates)
    return summed_proportion

def attribute_timing_to_upstream_synapse(spike_monitor_post,
                                        spike_times_pre,
                                        connections_xa1,
                                        connections_upstream,
                                        timing_threshold,
                                        epsilon_timing,
                                        A_plus=0.01,
                                        A_minus=-0.012,
                                        tau_plus=20.0,
                                        tau_minus=20.0,
                                        use_fast_mode=False,
                                        max_active_synapses=None,
                                        time_window_ms=None,
                                        mismatch_bin_width_ms=5.0,
                                        subsample_rate=1.0):
    """
    Find pairs of X->A synapses whose pre->post timing mismatches are similar,
    then locate an upstream neuron that projects to both presynaptic neurons and
    update the most-plastic upstream outgoing synapse using an STDP kernel computed
    from the average mismatch of the pair.

    Fast mode optimizations:
    - `use_fast_mode` : bool, if True use heuristics (binning, filtering) for speed.
    - `max_active_synapses` : int or None, filter to top K synapses by spike count (None=all).
    - `time_window_ms` : float or None, only consider spikes in last N ms (None=all).
    - `mismatch_bin_width_ms` : float, bin mismatch values into groups (e.g., ±2ms) for faster matching.
    - `subsample_rate` : float in (0,1], randomly sample fraction of synapses (0.1=10%, 1.0=all).

    Notes / assumptions:
    - `connections_xa1` are the synapses from layer X (presyn neurons) to A1 (post neurons);
      each synapse has .i (pre index) and .j (post index) and optionally attributes like
      `eta_p` representing plasticity and `w` for weight.
    - `connections_upstream` are synapses from an upstream layer U to the X layer (their
      .i are upstream neuron indices, .j are indices of X neurons). We search for an
      upstream neuron u that has outgoing synapses to both presyn neurons of the pair.
    - `spike_monitor_post` is a SpikeMonitor for A1 (provides .i and .t) or a dict-like
      mapping post neuron -> spike times in ms.
    - `spike_times_pre` is a dict-like mapping pre neuron index -> list/array of pre spike times in ms.

    Returns:
    --------
    applied_updates : list of tuples
        Each tuple is (upstream_syn_index, delta_w) applied to `connections_upstream.w`.
    """
    # helper: compute STDP delta from delta_t in ms
    def stdp_delta(delta_t_ms):
        if np.isnan(delta_t_ms):
            return 0.0
        if delta_t_ms > 0:
            return A_plus * np.exp(-delta_t_ms / tau_plus)
        else:
            return A_minus * np.exp(delta_t_ms / tau_minus)

    # Build post spike times per neuron (ms)
    post_spikes = {}
    # Accept either a SpikeMonitor-like object or a precomputed dict/array
    if hasattr(spike_monitor_post, 'i') and hasattr(spike_monitor_post, 't'):
        # collect post spikes from monitor
        for idx, t in zip(spike_monitor_post.i, spike_monitor_post.t):
            post_spikes.setdefault(int(idx), []).append(float(t / b.ms))
    elif isinstance(spike_monitor_post, dict):
        # assume already in ms
        for k, v in spike_monitor_post.items():
            post_spikes[int(k)] = [float(x) for x in v]
    else:
        # fallback: empty
        post_spikes = {}

    # Ensure spike_times_pre is dict-like: pre neuron -> list of spike times (ms)
    pre_spikes = {}
    if isinstance(spike_times_pre, dict):
        for k, v in spike_times_pre.items():
            pre_spikes[int(k)] = [float(x) for x in v]
    else:
        # if monitor-like object was passed instead (has .i and .t)
        if hasattr(spike_times_pre, 'i') and hasattr(spike_times_pre, 't'):
            for idx, t in zip(spike_times_pre.i, spike_times_pre.t):
                pre_spikes.setdefault(int(idx), []).append(float(t / b.ms))

    # Heuristic 1: Filter spikes by time window (keep only recent spikes)
    if time_window_ms is not None:
        # Find max spike time across all neurons
        all_times = []
        for times in list(post_spikes.values()) + list(pre_spikes.values()):
            all_times.extend(times)
        if len(all_times) > 0:
            max_time = np.max(all_times)
            cutoff_time = max_time - time_window_ms
            # filter post spikes
            for k in post_spikes:
                post_spikes[k] = [t for t in post_spikes[k] if t >= cutoff_time]
            # filter pre spikes
            for k in pre_spikes:
                pre_spikes[k] = [t for t in pre_spikes[k] if t >= cutoff_time]
    
    # Compute per-synapse mismatch: mean(post - nearest_pre) over available spike pairs
    n_syn = len(connections_xa1.i)
    syn_mismatch = np.full(n_syn, np.nan, dtype=float)
    syn_spike_count = np.zeros(n_syn, dtype=int)  # track activity for filtering
    
    for s_idx in range(n_syn):
        pre_idx = int(connections_xa1.i[s_idx])
        post_idx = int(connections_xa1.j[s_idx])
        pre_times = np.asarray(pre_spikes.get(pre_idx, []), dtype=float)
        post_times = np.asarray(post_spikes.get(post_idx, []), dtype=float)
        
        syn_spike_count[s_idx] = len(pre_times) + len(post_times)
        
        if pre_times.size == 0 or post_times.size == 0:
            continue
        # for each post spike, find nearest pre spike and record delta = post - pre
        deltas = []
        for t_post in post_times:
            idx_min = np.argmin(np.abs(pre_times - t_post))
            delta = t_post - pre_times[idx_min]
            deltas.append(delta)
        if len(deltas) > 0:
            syn_mismatch[s_idx] = np.mean(deltas)

    # Heuristic 2: Filter to top K most active synapses
    active_syn_indices = np.where(~np.isnan(syn_mismatch))[0]
    if max_active_synapses is not None and len(active_syn_indices) > max_active_synapses:
        # rank by spike count + mismatch confidence
        scores = syn_spike_count[active_syn_indices]
        top_k_idx = np.argsort(scores)[-max_active_synapses:]
        active_syn_indices = active_syn_indices[top_k_idx]
    
    print('Active synapses considered for pairing:', active_syn_indices.shape[0])
    
    # Heuristic 3: Subsample synapses if requested
    if subsample_rate < 1.0:
        n_keep = max(1, int(len(active_syn_indices) * subsample_rate))
        active_syn_indices = np.random.choice(active_syn_indices, size=n_keep, replace=False)

    # Heuristic 4: Find candidate pairs using mismatch sliding window
    if use_fast_mode and mismatch_bin_width_ms > 0:
        threshold = mismatch_bin_width_ms

        # Extract valid synapses with their mismatch values
        valid = []
        for syn_i in active_syn_indices:
            m = syn_mismatch[syn_i]
            if not np.isnan(m):
                valid.append((syn_i, m))

        # Sort by mismatch value
        valid.sort(key=lambda x: x[1])
        candidate_pairs = []
        n = len(valid)

        # Sliding window
        j = 0
        for i in range(n):
            # Expand j until mismatch difference exceeds threshold
            while j < n and (valid[j][1] - valid[i][1]) <= threshold:
                if j > i:
                    candidate_pairs.append((valid[i][0], valid[j][0]))
                j += 1

            # Important: reset j = i+1 only if needed
            # (This prevents skipping valid windows)
            if j < i + 1:
                j = i + 1

    else:
        # Full pairwise (fallback)
        candidate_pairs = []
        for i, s1 in enumerate(active_syn_indices):
            for s2 in active_syn_indices[i+1:]:
                candidate_pairs.append((s1, s2))
    
    print('Candidate synapse pairs to evaluate:', len(candidate_pairs))

    # Find pairs of synapses with similar mismatches
    applied_updates = []
    
    # Build upstream map
    upstream_i = np.array(connections_upstream.i, dtype=int)
    upstream_j = np.array(connections_upstream.j, dtype=int)
    upstream_map = {}
    for u_idx, target in zip(upstream_i, upstream_j):
        upstream_map.setdefault(int(u_idx), set()).add(int(target))

    # Process candidate pairs
    for s1, s2 in candidate_pairs:
        m1 = syn_mismatch[s1]
        m2 = syn_mismatch[s2]
        if np.isnan(m1) or np.isnan(m2):
            continue
        if abs(m1 - m2) > timing_threshold:
            continue
        
        # candidate pair
        pre1 = int(connections_xa1.i[s1])
        pre2 = int(connections_xa1.i[s2])
        
        # find upstream neurons u that project to BOTH pre1 and pre2
        candidates = [u for u, tgtset in upstream_map.items() if (pre1 in tgtset and pre2 in tgtset)]
        if len(candidates) == 0:
            continue
        
        # For each candidate upstream neuron, find the most plastic synapse
        best_syn_idx = None
        best_eta = -np.inf
        
        for u in candidates:
            syn_idx_u_pre1 = np.where((upstream_i == u) & (upstream_j == pre1))[0]
            syn_idx_u_pre2 = np.where((upstream_i == u) & (upstream_j == pre2))[0]
            syn_indices_u = np.concatenate((syn_idx_u_pre1, syn_idx_u_pre2)) if (syn_idx_u_pre1.size + syn_idx_u_pre2.size) > 0 else np.array([], dtype=int)
            
            if syn_indices_u.size == 0:
                continue
            
            if hasattr(connections_upstream, 'eta_p'):
                eta_vals = np.array(connections_upstream.eta_p[syn_indices_u], dtype=float)
            elif hasattr(connections_upstream, 'eta'):
                eta_vals = np.array(connections_upstream.eta[syn_indices_u], dtype=float)
            else:
                eta_vals = np.zeros(len(syn_indices_u))
            
            local_max_idx = np.argmax(eta_vals)
            local_eta = float(eta_vals[local_max_idx])
            chosen_syn = int(syn_indices_u[local_max_idx])
            
            if local_eta > best_eta:
                best_eta = local_eta
                best_syn_idx = chosen_syn

        
        if best_syn_idx is None:
            continue
        
        # Compute update
        avg_mismatch = 0.5 * (m1 + m2)
        stdp_dw = stdp_delta(avg_mismatch)
        delta_w = float(epsilon_timing) * float(stdp_dw)
        
        # Apply and record
        try:
            w_arr = np.array(connections_upstream.w, dtype=float)
            w_arr[best_syn_idx] += delta_w
            connections_upstream.w = w_arr
        except Exception:
            pass
        
        applied_updates.append((int(best_syn_idx), float(delta_w)))

    return applied_updates

    

    
