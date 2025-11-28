import numpy as np
import os.path
import scipy 
import brian2 as b
from brian2 import *
import time
from math import *
import argparse
import csv

from Functions import *

#set parameters
np.random.seed(0)
prefs.codegen.target = 'numpy'
prefs.codegen.cpp.extra_compile_args_gcc = ['-march=native']

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=0.00075)
parser.add_argument('--timing_threshold', type=float, default=5.0)   # ms
parser.add_argument('--epsilon_timing', type=float, default=0.07)
parser.add_argument('--run_name', type=str, default='run1')
parser.add_argument('--fast_timing_updates', action='store_true', help='Enable fast mode for timing updates')
parser.add_argument('--max_active_synapses', type=int, default=None, help='Limit to top K active synapses')
parser.add_argument('--time_window_ms', type=float, default=None, help='Only use spikes from last N ms')
parser.add_argument('--subsample_rate', type=float, default=1.0, help='Fraction of synapses to use (0.0-1.0)')
args = parser.parse_args()

#---------------------------------------------------Build network----------------------------------------------------------
Learning = True

print ('The settings of network are as follow:')
print ('---------------------------------------------')

#the amount of input
n_input = 784
n_layer_2 = 400

# Multi-scale layer A3 configuration (Inception-style)
a3_kernel_configs = [
    {'size': 28, 'stride': 1, 'num_kernels': 4, 'channels_per_kernel': 25},  # fine details
    {'size': 24, 'stride': 4, 'num_kernels': 2, 'channels_per_kernel': 25},  # medium details  
    {'size': 16, 'stride': 6, 'num_kernels': 1, 'channels_per_kernel': 25},  # coarse details
]

# Calculate A3 neurons
n_layer_3 = 0
a3_neuron_groups = []
for config in a3_kernel_configs:
    feature_map_size = int(((sqrt(n_input) - config['size']) / config['stride'] + 1)**2)
    neurons_per_kernel = config['channels_per_kernel'] * feature_map_size
    n_layer_3 += config['num_kernels'] * neurons_per_kernel
    a3_neuron_groups.append({
        'kernel_size': config['size'],
        'stride': config['stride'],
        'num_kernels': config['num_kernels'],
        'channels': config['channels_per_kernel'],
        'feature_map_size': feature_map_size,
        'neurons_per_kernel': neurons_per_kernel
    })

print('A3 layer (multi-scale convolutional):', n_layer_3, 'neurons')
print('A3 configuration:', a3_neuron_groups)
print('A2 layer (feature integration):', n_layer_2, 'neurons')

#connection parameters
kernel_type_num = 3
kernel_size_each_type = [28, 24, 16]
stride_each_type = [1, 4, 6]
feature_map_size_each_type = [int(((sqrt(n_input) - kernel_size_each_type[i]) / stride_each_type[i] + 1)**2) for i in range(kernel_type_num)]
print ('Num of kernel type:', kernel_type_num)
print ('Kernel size and Stride of each kernel type:', kernel_size_each_type, stride_each_type)
print ('Feature map size of each kernel type:', feature_map_size_each_type)

feature_map_num = 448
kernel_num_each_type = [4, 2, 1]
kernel_num = np.sum(kernel_num_each_type)
print ('Feature map num:', feature_map_num)
print ('Kernel num of each kernel type:', kernel_num_each_type)
print ('Num of kernel:', kernel_num)

#the amount of neurons
neuron_num_each_kernel = []
feature_map_size_each_kernel = []
kernel_size_each_kernel = []
stride_each_kernel = []
for kernel_type in range(kernel_type_num):
    for kernel in range(kernel_num_each_type[kernel_type]):
        neuron_num_each_kernel.append(feature_map_num * feature_map_size_each_type[kernel_type])
        feature_map_size_each_kernel.append(feature_map_size_each_type[kernel_type])
        kernel_size_each_kernel.append(kernel_size_each_type[kernel_type])
        stride_each_kernel.append(stride_each_type[kernel_type])

neuron_num = np.sum(neuron_num_each_kernel)
print ('Neurons num of each kernel:', neuron_num_each_kernel)
print ('Num of Neurons:', neuron_num)
print ('---------------------------------------------')
    
#neuron parameters
v_rest_e = -65. * b.mV
v_reset_e = -65. * b.mV
v_thresh_e = -52. * b.mV
refrac_e = 5. * b.ms

#synapses parameters
Delay = 10*b.ms
tc_pre_ee = 20*b.ms
tc_post_1_ee = 20*b.ms
tc_post_2_ee = 40*b.ms
nu_ee_pre =  0.0001
nu_ee_post = 0.01
wmax_ee = 1.0
wmin_ee = 1e-7
ihn = 24
norm = 78.4

#plasticity decay parameters
tc_eta_decay = 100 * b.second  # time constant for eta decay
eta_decay_factor = 0.95  # multiplicative factor per STDP event (0 < factor < 1)
nu_ee_pre_init = nu_ee_pre  # store initial learning rates
nu_ee_post_init = nu_ee_post
alpha = 0.00075 #! TWEAK THIS VALUE TO FIND BEST PERFORMANCE, changes influence of delta w on eta
eta_max_factor= 1.5

# Relevant timing attribution parameters
timing_threshold = 5.0 * b.ms  # threshold for similar timing differences
epsilon_timing = 0.001  # learning rate for timing-based updates
    
alpha = args.alpha
timing_threshold = args.timing_threshold * b.ms
epsilon_timing = args.epsilon_timing
run_name = args.run_name
fast_timing_updates = args.fast_timing_updates
max_active_synapses = args.max_active_synapses
time_window_ms = args.time_window_ms
subsample_rate = args.subsample_rate

# create directory for logs
log_dir = os.path.join('./logs', run_name)
os.makedirs(log_dir, exist_ok=True)
plasticity_csv = os.path.join(log_dir, 'plasticity_log.csv')
# write header
with open(plasticity_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['iter','time',
                'mean_eta_p_x1a3','std_eta_p_x1a3','p50_eta_p_x1a3','p90_eta_p_x1a3',
                'mean_eta_q_x1a3','std_eta_q_x1a3','p50_eta_q_x1a3','p90_eta_q_x1a3',
                'mean_eta_p_a3a2','std_eta_p_a3a2','p50_eta_p_a3a2','p90_eta_p_a3a2',
                'mean_eta_q_a3a2','std_eta_q_a3a2','p50_eta_q_a3a2','p90_eta_q_a3a2',
                'mean_eta_p_a2a1','std_eta_p_a2a1','p50_eta_p_a2a1','p90_eta_p_a2a1',
                'mean_eta_q_a2a1','std_eta_q_a2a1','p50_eta_q_a2a1','p90_eta_q_a2a1',
                'mean_theta_mv','std_theta_mv',
                'weight_change_norm_x1a3', 'weight_change_avg_x1a3', 'weight_change_max_x1a3',
                'weight_change_norm_a3a2', 'weight_change_avg_a3a2', 'weight_change_max_a3a2',
                'weight_change_norm_a2a1', 'weight_change_avg_a2a1', 'weight_change_max_a2a1',
                'accuracy'])
timing_updates_csv = os.path.join(log_dir, 'timing_updates_log.csv')
# write header for timing updates
with open(timing_updates_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['iter', 'time', 'connection', 'n_updates', 'sum_abs_dw', 'max_abs_dw'])

if Learning == False:
    scr_e = 'v = v_reset_e'
else:
    tc_theta = 1e5 * b.ms
    theta_plus_e = 0.05 * b.mV
    scr_e = 'v = v_reset_e; theta += theta_plus_e'
offset = 20.0*b.mV
thresh_e = 'v>(theta - offset + ' + str(v_thresh_e/b.mV) + '*mV' + ')'

#equation of excitatory neurons
neuron_eqs_e = '''
            dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt
            I_synE = ge * nS * (        -v)                             : amp
            I_synI = gi * nS * (-100.*mV-v)                             : amp
            dge/dt = -ge/(1.0*ms)                                       : 1
            dgi/dt = -gi/(2.0*ms)                                       : 1
            '''
if Learning == False:
    neuron_eqs_e += '\n  theta      :volt'
else:
    neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'

#equation of STDP
eqs_stdp_ee = '''
                    w                                      : 1
                    delta_w                                : 1
                    post2before                            : 1
                    
                    dpre/dt    = -pre/(tc_pre_ee)          : 1 (event-driven)
                    dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                    dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
                    
                    deta_p/dt = -(eta_p - eta_decay_factor*nu_ee_pre_init) / tc_eta_decay : 1 (event-driven)
                    deta_q/dt = -(eta_q - eta_decay_factor*nu_ee_post_init) / tc_eta_decay : 1 (event-driven)
                '''
eqs_stdp_pre_ee = '''
                    ge+=w;
                    pre = 1.;
                    delta_w = eta_p * post1;
                    w = int(w>0)*clip(w - delta_w , wmin_ee, wmax_ee);
                    eta_p = eta_p + 1.0 / clip((eta_max_factor * nu_ee_pre_init - alpha * delta_w), 1e-12, 1e9);
                    eta_p = clip(eta_p, eta_decay_factor*nu_ee_pre_init, eta_max_factor * nu_ee_pre_init)
'''

eqs_stdp_post_ee = '''
                    post2before = post2;
                    delta_w = eta_q * pre * post2before
                    w = int(w>0)*clip(w + delta_w, wmin_ee, wmax_ee);
                    post1 = 1.;
                    post2 = 1.;
                    eta_q = eta_q + 1.0 / clip((eta_max_factor * nu_ee_post_init - alpha * delta_w), 1e-12, 1e9);
                    eta_q = clip(eta_q, eta_decay_factor*nu_ee_post_init, eta_max_factor * nu_ee_post_init)            
'''
                    
#create empty dict
neuron_groups = {}
connections = {}
spike_counters = {}
net = {}
    
#create neuron group
neuron_groups['X1'] = b.PoissonGroup(n_input, 0*b.hertz)

# Multi-scale convolutional layer A3
neuron_groups['A3'] = b.NeuronGroup(n_layer_3, neuron_eqs_e, method='euler', threshold=thresh_e, refractory=refrac_e, reset= scr_e)
neuron_groups['A3'].v = v_rest_e - 40. * b.mV
neuron_groups['A3'].theta = np.ones((n_layer_3)) * 20.0*b.mV

# Feature integration layer A2
neuron_groups['A2'] = b.NeuronGroup(n_layer_2, neuron_eqs_e, method='euler', threshold=thresh_e, refractory=refrac_e, reset= scr_e)
neuron_groups['A2'].v = v_rest_e - 40. * b.mV
neuron_groups['A2'].theta = np.ones((n_layer_2)) * 20.0*b.mV

# Output layer A1
neuron_groups['A1'] = b.NeuronGroup(neuron_num, neuron_eqs_e, method='euler', threshold=thresh_e, refractory=refrac_e, reset= scr_e)
neuron_groups['A1'].v = v_rest_e - 40. * b.mV
neuron_groups['A1'].theta = np.ones((neuron_num)) * 20.0*b.mV
    
#create connections AA
start = time.time()
weightMatrix = np.zeros((neuron_num, neuron_num))
mark = 0
for kernel in range(kernel_num):
    feature_map_size = feature_map_size_each_kernel[kernel]
    for src in range(mark, mark+neuron_num_each_kernel[kernel]):
        S = src - mark
        src_z = int(S/feature_map_size)
        src_y = int((S - src_z*feature_map_size) / sqrt(feature_map_size))
        src_x = int(S - src_z*feature_map_size - src_y*sqrt(feature_map_size))
        for tar in range(mark, mark+neuron_num_each_kernel[kernel]):
            T = tar - mark
            tar_z = int(T / feature_map_size)
            tar_y = int((T - tar_z*feature_map_size) / sqrt(feature_map_size))
            tar_x = int(T - tar_z*feature_map_size - tar_y*sqrt(feature_map_size))
            if src_x == tar_x and src_y == tar_y and src_z != tar_z:
                weightMatrix[src,tar] = ihn
    mark += neuron_num_each_kernel[kernel]
weightMatrix = weightMatrix.reshape((neuron_num*neuron_num))
connections['A1A1'] = b.Synapses(neuron_groups['A1'], neuron_groups['A1'], 'w:1',on_pre='gi+=w')
connections['A1A1'].connect()
connections['A1A1'].w = weightMatrix
end = time.time()
print ('time needed to create connection A1A1:', end - start)

#create connections X1A3 (multi-scale, locally connected)
start = time.time()
weightMatrix_X1A3 = np.zeros((n_input, n_layer_3))

neuron_offset = 0
for group_idx, group_config in enumerate(a3_neuron_groups):
    kernel_size = group_config['kernel_size']
    stride = group_config['stride']
    feature_map_size = group_config['feature_map_size']
    channels = group_config['channels']
    num_kernels = group_config['num_kernels']
    
    input_width = int(sqrt(n_input))
    output_width = int(sqrt(feature_map_size))
    
    for kernel_idx in range(num_kernels):
        for channel in range(channels):
            for out_y in range(output_width):
                for out_x in range(output_width):
                    # Calculate output neuron index
                    tar = neuron_offset + kernel_idx * (channels * feature_map_size) + \
                          channel * feature_map_size + out_y * output_width + out_x
                    
                    # Connect to receptive field
                    for ky in range(kernel_size):
                        for kx in range(kernel_size):
                            src_y = out_y * stride + ky
                            src_x = out_x * stride + kx
                            if src_y < input_width and src_x < input_width:
                                src = src_y * input_width + src_x
                                weightMatrix_X1A3[src, tar] = 0.3 * np.random.rand() + wmin_ee
    
    neuron_offset += num_kernels * channels * feature_map_size

weightMatrix_X1A3 = weightMatrix_X1A3.reshape((n_input * n_layer_3))

if Learning:
    connections['X1A3'] = b.Synapses(neuron_groups['X1'], neuron_groups['A3'], 
                                      eqs_stdp_ee, on_pre=eqs_stdp_pre_ee, on_post=eqs_stdp_post_ee)
else:
    connections['X1A3'] = b.Synapses(neuron_groups['X1'], neuron_groups['A3'], 'w : 1', on_pre='ge+=w')

connections['X1A3'].connect()
connections['X1A3'].w = weightMatrix_X1A3
connections['X1A3'].delay = 'rand()*'+ str(Delay/b.ms) +'*ms'

if Learning:
    connections['X1A3'].eta_p = nu_ee_pre_init
    connections['X1A3'].eta_q = nu_ee_post_init

print('time needed to create connection X1A3:', time.time() - start)

#create connections A3A3 (lateral inhibition within kernel groups)
start = time.time()
weightMatrix_A3A3 = np.zeros((n_layer_3, n_layer_3))

neuron_offset = 0
for group_config in a3_neuron_groups:
    feature_map_size = group_config['feature_map_size']
    channels = group_config['channels']
    num_kernels = group_config['num_kernels']
    
    for kernel_idx in range(num_kernels):
        kernel_start = neuron_offset + kernel_idx * (channels * feature_map_size)
        
        # Lateral inhibition at same spatial location across channels
        for spatial_pos in range(feature_map_size):
            for src_channel in range(channels):
                src = kernel_start + src_channel * feature_map_size + spatial_pos
                for tar_channel in range(channels):
                    if src_channel != tar_channel:
                        tar = kernel_start + tar_channel * feature_map_size + spatial_pos
                        weightMatrix_A3A3[src, tar] = ihn
    
    neuron_offset += num_kernels * channels * feature_map_size

weightMatrix_A3A3 = weightMatrix_A3A3.reshape((n_layer_3 * n_layer_3))
connections['A3A3'] = b.Synapses(neuron_groups['A3'], neuron_groups['A3'], 'w:1', on_pre='gi+=w')
connections['A3A3'].connect()
connections['A3A3'].w = weightMatrix_A3A3
print('time needed to create connection A3A3:', time.time() - start)

#create connections A3A2 (fully connected feature integration)
start = time.time()

if Learning:
    connections['A3A2'] = b.Synapses(neuron_groups['A3'], neuron_groups['A2'], 
                                      eqs_stdp_ee, on_pre=eqs_stdp_pre_ee, on_post=eqs_stdp_post_ee)
else:
    connections['A3A2'] = b.Synapses(neuron_groups['A3'], neuron_groups['A2'], 'w : 1', on_pre='ge+=w')

connections['A3A2'].connect()
connections['A3A2'].w = '0.3*rand()+wmin_ee'
connections['A3A2'].delay = 'rand()*'+ str(Delay/b.ms) +'*ms'

if Learning:
    connections['A3A2'].eta_p = nu_ee_pre_init
    connections['A3A2'].eta_q = nu_ee_post_init

print('time needed to create connection A3A2:', time.time() - start)

#create connections X1A2 (kept for legacy compatibility, but unused in forward pass)
start = time.time()
if Learning:
    connections['X1A2'] = b.Synapses(neuron_groups['X1'], neuron_groups['A2'], eqs_stdp_ee, on_pre=eqs_stdp_pre_ee, on_post=eqs_stdp_post_ee)
else:
    connections['X1A2'] = b.Synapses(neuron_groups['X1'], neuron_groups['A2'], 'w : 1', on_pre='ge+=w')
# Note: We do NOT connect X1A2 - it's disabled in the new architecture
# connections['X1A2'].connect()
# connections['X1A2'].w = '0.3*rand()+wmin_ee'
# connections['X1A2'].delay = 'rand()*'+ str(Delay/b.ms) +'*ms'
# if Learning:
#     connections['X1A2'].eta_p = nu_ee_pre_init
#     connections['X1A2'].eta_q = nu_ee_post_init
print ('time needed to skip connection X1A2 (disabled in 3-layer architecture):', time.time() - start)

#create connections A2A1
start = time.time()
if Learning:
    connections['A2A1'] = b.Synapses(neuron_groups['A2'], neuron_groups['A1'], eqs_stdp_ee, on_pre=eqs_stdp_pre_ee, on_post=eqs_stdp_post_ee)
else:
    connections['A2A1'] = b.Synapses(neuron_groups['A2'], neuron_groups['A1'], 'w : 1', on_pre='ge+=w')
connections['A2A1'].connect()
connections['A2A1'].w = '0.3*rand()+wmin_ee'
connections['A2A1'].delay = 'rand()*'+ str(Delay/b.ms) +'*ms'
if Learning:
    connections['A2A1'].eta_p = nu_ee_pre_init
    connections['A2A1'].eta_q = nu_ee_post_init
print ('time needed to create connection A2A1:', time.time() - start)

# --- monitoring helpers ---------------------------------------------------
# prev_weights for convergence metric
prev_weights_x1a3 = np.array(connections['X1A3'].w).astype(float)
prev_weights_a3a2 = np.array(connections['A3A2'].w).astype(float)
prev_weights_a2a1 = np.array(connections['A2A1'].w).astype(float)

def log_plasticity(iteration, accuracy_val=None):
    def stats(a):
        return np.mean(a), np.std(a), np.percentile(a,50), np.percentile(a,90)

    # X1A3 stats
    eta_p_x1a3 = np.array(connections['X1A3'].eta_p, dtype=float)
    eta_q_x1a3 = np.array(connections['X1A3'].eta_q, dtype=float)
    mean_p_x1a3, std_p_x1a3, p50_p_x1a3, p90_p_x1a3 = stats(eta_p_x1a3)
    mean_q_x1a3, std_q_x1a3, p50_q_x1a3, p90_q_x1a3 = stats(eta_q_x1a3)

    # A3A2 stats
    eta_p_a3a2 = np.array(connections['A3A2'].eta_p, dtype=float)
    eta_q_a3a2 = np.array(connections['A3A2'].eta_q, dtype=float)
    mean_p_a3a2, std_p_a3a2, p50_p_a3a2, p90_p_a3a2 = stats(eta_p_a3a2)
    mean_q_a3a2, std_q_a3a2, p50_q_a3a2, p90_q_a3a2 = stats(eta_q_a3a2)

    # A2A1 stats
    eta_p_a2a1 = np.array(connections['A2A1'].eta_p, dtype=float)
    eta_q_a2a1 = np.array(connections['A2A1'].eta_q, dtype=float)
    mean_p_a2a1, std_p_a2a1, p50_p_a2a1, p90_p_a2a1 = stats(eta_p_a2a1)
    mean_q_a2a1, std_q_a2a1, p50_q_a2a1, p90_q_a2a1 = stats(eta_q_a2a1)

    theta_vals = np.array(neuron_groups['A1'].theta / b.mV, dtype=float)
    mean_theta, std_theta = np.mean(theta_vals), np.std(theta_vals)

    # Weight changes for all layers
    cur_w_x1a3 = np.array(connections['X1A3'].w, dtype=float)
    w_change_x1a3 = np.linalg.norm(cur_w_x1a3 - prev_weights_x1a3)
    w_change_avg_x1a3 = np.mean(np.abs(cur_w_x1a3 - prev_weights_x1a3))
    w_change_max_x1a3 = np.max(np.abs(cur_w_x1a3 - prev_weights_x1a3))
    
    cur_w_a3a2 = np.array(connections['A3A2'].w, dtype=float)
    w_change_a3a2 = np.linalg.norm(cur_w_a3a2 - prev_weights_a3a2)
    w_change_avg_a3a2 = np.mean(np.abs(cur_w_a3a2 - prev_weights_a3a2))
    w_change_max_a3a2 = np.max(np.abs(cur_w_a3a2 - prev_weights_a3a2))
    
    cur_w_a2a1 = np.array(connections['A2A1'].w, dtype=float)
    w_change_a2a1 = np.linalg.norm(cur_w_a2a1 - prev_weights_a2a1)
    w_change_avg_a2a1 = np.mean(np.abs(cur_w_a2a1 - prev_weights_a2a1))
    w_change_max_a2a1 = np.max(np.abs(cur_w_a2a1 - prev_weights_a2a1))

    with open(plasticity_csv, 'a', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow([iteration, time.time(),
                       mean_p_x1a3, std_p_x1a3, p50_p_x1a3, p90_p_x1a3,
                       mean_q_x1a3, std_q_x1a3, p50_q_x1a3, p90_q_x1a3,
                       mean_p_a3a2, std_p_a3a2, p50_p_a3a2, p90_p_a3a2,
                       mean_q_a3a2, std_q_a3a2, p50_q_a3a2, p90_q_a3a2,
                       mean_p_a2a1, std_p_a2a1, p50_p_a2a1, p90_p_a2a1,
                       mean_q_a2a1, std_q_a2a1, p50_q_a2a1, p90_q_a2a1,
                       mean_theta, std_theta,
                       w_change_x1a3, w_change_avg_x1a3, w_change_max_x1a3,
                       w_change_a3a2, w_change_avg_a3a2, w_change_max_a3a2,
                       w_change_a2a1, w_change_avg_a2a1, w_change_max_a2a1,
                       '' if accuracy_val is None else accuracy_val])
    
    nonlocal_prev_weights_assign(cur_w_x1a3, cur_w_a3a2, cur_w_a2a1)

def nonlocal_prev_weights_assign(arr_x1a3, arr_a3a2, arr_a2a1):
    global prev_weights_x1a3, prev_weights_a3a2, prev_weights_a2a1
    prev_weights_x1a3 = arr_x1a3.copy()
    prev_weights_a3a2 = arr_a3a2.copy()
    prev_weights_a2a1 = arr_a2a1.copy()

#create monitors
spike_counters['A1'] = b.SpikeMonitor(neuron_groups['A1'], record=True)
spike_counters['A2'] = b.SpikeMonitor(neuron_groups['A2'], record=True)
spike_counters['A3'] = b.SpikeMonitor(neuron_groups['A3'], record=True)

#create networks
net['M1'] = Network(
    neuron_groups['A1'], 
    neuron_groups['A2'], 
    neuron_groups['A3'],
    neuron_groups['X1'], 
    connections['X1A3'],
    connections['A3A3'],
    connections['A3A2'], 
    connections['A2A1'], 
    connections['A1A1'], 
    spike_counters['A1'], 
    spike_counters['A2'],
    spike_counters['A3']
)
#-----------------------------------------------------------------------------------------------------------------------

# load MNIST
start = time.time()
training = get_labeled_data('./training')
end = time.time()
print ('time needed to load training set:', end - start)

#specify the location
save_path = './weights/'
load_path = './weights/'

#the time-window of simulation
single_example_time =   0.7 * b.second
resting_time = 0.3 * b.second

#the the interval of process data and show information
progress_interval = 10
validate_interval = 5000   #no less than 2000
save_interval = 30

#number of samples for training
n_train = 1000
train_begin = 0    #specify which iteration you want the training to begin from 

#load trained weight to continue
if train_begin:
    connections['X1A3'].w = np.load(load_path + 'X1A3' + '_' + str(train_begin) + '.npy')
    connections['A3A2'].w = np.load(load_path + 'A3A2' + '_' + str(train_begin) + '.npy')
    connections['A2A1'].w = np.load(load_path + 'A2A1' + '_' + str(train_begin) + '.npy')
    neuron_groups['A1'].theta = np.load(load_path + 'theta_A1' + '_' + str(train_begin) + '.npy') *b.volt
    neuron_groups['A2'].theta = np.load(load_path + 'theta_A2' + '_' + str(train_begin) + '.npy') *b.volt
    neuron_groups['A3'].theta = np.load(load_path + 'theta_A3' + '_' + str(train_begin) + '.npy') *b.volt

#the intensity of rate coding
intensity_step = 0.25
start_intensity = 0.5

#the threshold of retrain
retrain_gate = np.sum([3*feature_map_size_each_kernel[kernel] for kernel in range(kernel_num)])

# run the simulation and set inputs
previous_spike_count = {}
current_spike_count = {}
assignments = {}
result_monitor = {}
results_proportion = {}
accuracy = {}

previous_spike_count['A1'] = np.zeros(neuron_num)
current_spike_count['A1'] = np.zeros(neuron_num)
assignments['A1'] = np.zeros(neuron_num)
result_monitor['A1'] = np.zeros((validate_interval,neuron_num))
results_proportion['A1'] = np.zeros((10, validate_interval))
accuracy['A1'] = []
input_numbers = np.zeros(validate_interval)

neuron_groups['X1'].rates = 0*b.hertz
net['M1'].run(0*b.second)

start = time.time()

j = train_begin
max_retries = 30
last_printed_j = None

input_intensity = start_intensity
while j < n_train:   
    
    print('Training iteration:', j+1, '/', n_train)
    
    if last_printed_j != j:
        last_printed_j = j
        retry_count = 0
     
    Rates = training['x'][j%60000,:,:].reshape((n_input)) * input_intensity

    neuron_groups['X1'].rates = Rates*b.hertz
    connections['X1A3'] = normalize_weights(connections['X1A3'],norm)
    connections['A3A2'] = normalize_weights(connections['A3A2'],norm)
    connections['A2A1'] = normalize_weights(connections['A2A1'],norm)

    net['M1'].run(single_example_time)
    
    current_spike_count['A1'] = np.asarray(spike_counters['A1'].count[:])- previous_spike_count['A1']
    previous_spike_count['A1'] = np.copy(spike_counters['A1'].count[:])
    
    spike_num = np.sum(current_spike_count['A1'])

    if spike_num < retrain_gate:
        retry_count += 1
        
        
        if retry_count >= max_retries:  # << added: check limit
            print(f"  Skipping sample {j} after {retry_count} retries (spike_num={int(spike_num)} < {int(retrain_gate)})")
            # reset for next sample
            input_intensity = start_intensity
            neuron_groups['X1'].rates = 0*b.hertz
            neuron_groups['A3'].v = v_rest_e - 40. * b.mV
            neuron_groups['A2'].v = v_rest_e - 40. * b.mV
            neuron_groups['A1'].v = v_rest_e - 40. * b.mV
            net['M1'].run(resting_time)
            j += 1  # << skip to next sample
            continue  # << go to top of loop
        
        
        input_intensity += intensity_step
        neuron_groups['X1'].rates = 0*b.hertz
        net['M1'].run(resting_time)
    else:
        print('Training iteration:', j+1, '/', n_train, '- spikes:', int(spike_num))
        result_monitor['A1'][j%validate_interval,:] = current_spike_count['A1']
        input_numbers[j%validate_interval] = training['y'][j%60000][0]

        if j%5 == 0:
        
            # Apply timing-based attribution: A2→A1 propagated to A3→A2
            print('Applying timing-based updates A2→A1 to A3→A2... ==========')
            applied_updates_a3a2 = attribute_timing_to_upstream_synapse(
                spike_counters['A1'],
                spike_counters['A2'],
                connections['A2A1'],
                connections['A3A2'],
                float(timing_threshold / b.ms),
                epsilon_timing,
                use_fast_mode=fast_timing_updates,
                max_active_synapses=max_active_synapses,
                time_window_ms=time_window_ms,
                subsample_rate=subsample_rate
            )
            
            # Apply timing-based attribution: A3→A2 propagated to X1→A3
            print('Applying timing-based updates A3→A2 to X1→A3... ==========')
            applied_updates_x1a3 = attribute_timing_to_upstream_synapse(
                spike_counters['A2'],
                spike_counters['A3'],
                connections['A3A2'],
                connections['X1A3'],
                float(timing_threshold / b.ms),
                epsilon_timing,
                use_fast_mode=fast_timing_updates,
                max_active_synapses=max_active_synapses,
                time_window_ms=time_window_ms,
                subsample_rate=subsample_rate
            )

            # Apply updates to A3A2
            try:
                w_arr_a3a2 = np.array(connections['A3A2'].w, dtype=float)
                n_updates_a3a2 = 0
                sum_abs_dw_a3a2 = 0.0
                max_abs_dw_a3a2 = 0.0
                if isinstance(applied_updates_a3a2, list):
                    for syn_idx, dw in applied_updates_a3a2:
                        try:
                            syn_i = int(syn_idx)
                            dwf = float(dw)
                            w_arr_a3a2[syn_i] = np.clip(w_arr_a3a2[syn_i] + dwf, wmin_ee, wmax_ee)
                            n_updates_a3a2 += 1
                            sum_abs_dw_a3a2 += abs(dwf)
                            if abs(dwf) > max_abs_dw_a3a2:
                                max_abs_dw_a3a2 = abs(dwf)
                        except Exception:
                            continue
                connections['A3A2'].w = w_arr_a3a2
            except Exception:
                n_updates_a3a2 = 0
                sum_abs_dw_a3a2 = 0.0
                max_abs_dw_a3a2 = 0.0
            
            # Apply updates to X1A3
            try:
                w_arr_x1a3 = np.array(connections['X1A3'].w, dtype=float)
                n_updates_x1a3 = 0
                sum_abs_dw_x1a3 = 0.0
                max_abs_dw_x1a3 = 0.0
                if isinstance(applied_updates_x1a3, list):
                    for syn_idx, dw in applied_updates_x1a3:
                        try:
                            syn_i = int(syn_idx)
                            dwf = float(dw)
                            w_arr_x1a3[syn_i] = np.clip(w_arr_x1a3[syn_i] + dwf, wmin_ee, wmax_ee)
                            n_updates_x1a3 += 1
                            sum_abs_dw_x1a3 += abs(dwf)
                            if abs(dwf) > max_abs_dw_x1a3:
                                max_abs_dw_x1a3 = abs(dwf)
                        except Exception:
                            continue
                connections['X1A3'].w = w_arr_x1a3
            except Exception:
                n_updates_x1a3 = 0
                sum_abs_dw_x1a3 = 0.0
                max_abs_dw_x1a3 = 0.0
            
            # Log both updates
            with open(timing_updates_csv, 'a', newline='') as f:
                wcsv = csv.writer(f)
                wcsv.writerow([j, time.time(), 'A3A2', n_updates_a3a2, sum_abs_dw_a3a2, max_abs_dw_a3a2])
                wcsv.writerow([j, time.time(), 'X1A3', n_updates_x1a3, sum_abs_dw_x1a3, max_abs_dw_x1a3])
        
        neuron_groups['X1'].rates = 0*b.hertz
        net['M1'].run(resting_time)
        input_intensity = start_intensity

        j += 1
        
        if j%20 == 0:
            print("=== Diagnostics at every 20 iterations===")
            print(f"Mean theta A1: {np.mean(neuron_groups['A1'].theta/b.mV):.3f} mV")
            print(f"Max theta A1: {np.max(neuron_groups['A1'].theta/b.mV):.3f} mV")
            print(f"Mean weight X1A3: {np.mean(connections['X1A3'].w):.6f}")
            print(f"Mean weight A3A2: {np.mean(connections['A3A2'].w):.6f}")
            print(f"Mean weight A2A1: {np.mean(connections['A2A1'].w):.6f}")
            print(f"Mean eta_p X1A3: {np.mean(connections['X1A3'].eta_p):.6f}")
        
        if j % progress_interval == 0:
            print ('Progress: ', j, '/', n_train, '(', time.time() - start, 'seconds)')
            start = time.time()
            
        if j % validate_interval == 0:
            assignments['A1'] = get_new_assignments(result_monitor['A1'][:], input_numbers[:])
            test_results = np.zeros((10, validate_interval))
            for k in range(validate_interval):
                results_proportion['A1'][:,k] = get_recognized_number_proportion(assignments['A1'], result_monitor['A1'][k,:])
                test_results[:,k] = np.argsort(results_proportion['A1'][:,k])[::-1]
            difference = test_results[0,:] - input_numbers[:]
            correct = len(np.where(difference == 0)[0])
            acc = correct/float(validate_interval) * 100
            accuracy['A1'].append(acc)
            print ('Validate accuracy: ', acc, '(last)', np.max(accuracy['A1']), '(best)')

            log_plasticity(j, accuracy_val=acc)
            
        if j % save_interval == 0:
            np.save(save_path + 'X1A3' + '_' + str(j), connections['X1A3'].w)
            np.save(save_path + 'A3A2' + '_' + str(j), connections['A3A2'].w)
            np.save(save_path + 'A2A1' + '_' + str(j), connections['A2A1'].w)
            np.save(save_path + 'theta_A1' + '_' + str(j), neuron_groups['A1'].theta)
            np.save(save_path + 'theta_A2' + '_' + str(j), neuron_groups['A2'].theta)
            np.save(save_path + 'theta_A3' + '_' + str(j), neuron_groups['A3'].theta)
            log_plasticity(j, accuracy_val=None)