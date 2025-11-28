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

print ('='*60)
print ('2-Layer Inception-style SNN with Adaptive STDP + Credit Assignment')
print ('='*60)
print ('Architecture: X1 (input) → A1 (hidden, multi-scale conv) → A2 (output)')
print ('='*60)

#the amount of input
n_input = 784

#connection parameters - Inception-style multi-scale for hidden layer A1
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

# Hidden layer A1: Multi-scale Inception-style (locally connected)
neuron_num_each_kernel_A1 = []
feature_map_size_each_kernel_A1 = []
kernel_size_each_kernel_A1 = []
stride_each_kernel_A1 = []
for kernel_type in range(kernel_type_num):
    for kernel in range(kernel_num_each_type[kernel_type]):
        neuron_num_each_kernel_A1.append(feature_map_num * feature_map_size_each_type[kernel_type])
        feature_map_size_each_kernel_A1.append(feature_map_size_each_type[kernel_type])
        kernel_size_each_kernel_A1.append(kernel_size_each_type[kernel_type])
        stride_each_kernel_A1.append(stride_each_type[kernel_type])

n_layer_1 = int(np.sum(neuron_num_each_kernel_A1))  # Hidden layer (A1)
print ('Hidden layer A1 neurons per kernel:', neuron_num_each_kernel_A1)
print ('Total Hidden layer A1 neurons:', n_layer_1)

# Output layer A2: Classification layer
# Option 1: Smaller output layer for efficiency
# Option 2: Same multi-scale structure as A1
output_layer_mode = 'compact'  # 'compact' or 'full_inception'

if output_layer_mode == 'compact':
    # Compact output: one neuron per class × some redundancy
    n_layer_2 = 400  # Similar to original mesonet's integration layer
    neuron_num_each_kernel_A2 = [n_layer_2]  # Single group
    print ('Output layer A2 (compact):', n_layer_2, 'neurons')
else:
    # Full Inception-style output
    neuron_num_each_kernel_A2 = neuron_num_each_kernel_A1.copy()
    n_layer_2 = n_layer_1
    print ('Output layer A2 (full Inception):', n_layer_2, 'neurons')

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

#plasticity decay parameters (adaptive STDP)
tc_eta_decay = 100 * b.second
eta_decay_factor = 0.95
nu_ee_pre_init = nu_ee_pre
nu_ee_post_init = nu_ee_post
alpha = 0.00075
eta_max_factor = 1.5

# Timing attribution parameters
timing_threshold = 5.0 * b.ms
epsilon_timing = 0.001
    
# Override with command line args
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
with open(plasticity_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['iter','time',
                'mean_eta_p_x1a1','std_eta_p_x1a1','p50_eta_p_x1a1','p90_eta_p_x1a1',
                'mean_eta_q_x1a1','std_eta_q_x1a1','p50_eta_q_x1a1','p90_eta_q_x1a1',
                'mean_eta_p_a1a2','std_eta_p_a1a2','p50_eta_p_a1a2','p90_eta_p_a1a2',
                'mean_eta_q_a1a2','std_eta_q_a1a2','p50_eta_q_a1a2','p90_eta_q_a1a2',
                'mean_theta_a1_mv','std_theta_a1_mv',
                'mean_theta_a2_mv','std_theta_a2_mv',
                'weight_change_norm_x1a1', 'weight_change_avg_x1a1', 'weight_change_max_x1a1',
                'weight_change_norm_a1a2', 'weight_change_avg_a1a2', 'weight_change_max_a1a2',
                'accuracy'])

timing_updates_csv = os.path.join(log_dir, 'timing_updates_log.csv')
with open(timing_updates_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['iter', 'time', 'connection', 'n_updates', 'sum_abs_dw', 'max_abs_dw'])

if Learning == False:
    scr_e = 'v = v_reset_e'
else:
    tc_theta = 1e7 * b.ms
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

#equation of STDP with adaptive learning rates
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
    
#create neuron groups
# Input layer (Poisson encoded)
neuron_groups['X1'] = b.PoissonGroup(n_input, 0*b.hertz)

# Hidden layer A1 (multi-scale Inception-style)
neuron_groups['A1'] = b.NeuronGroup(n_layer_1, neuron_eqs_e, method='euler', threshold=thresh_e, refractory=refrac_e, reset=scr_e)
neuron_groups['A1'].v = v_rest_e - 40. * b.mV
neuron_groups['A1'].theta = np.ones((n_layer_1)) * 20.0*b.mV

# Output layer A2
neuron_groups['A2'] = b.NeuronGroup(n_layer_2, neuron_eqs_e, method='euler', threshold=thresh_e, refractory=refrac_e, reset=scr_e)
neuron_groups['A2'].v = v_rest_e - 40. * b.mV
neuron_groups['A2'].theta = np.ones((n_layer_2)) * 20.0*b.mV

#create connections A1A1 (lateral inhibition in hidden layer - Inception style)
start = time.time()
weightMatrix_A1A1 = np.zeros((n_layer_1, n_layer_1))
mark = 0
for kernel in range(kernel_num):
    feature_map_size = feature_map_size_each_kernel_A1[kernel]
    for src in range(mark, mark+neuron_num_each_kernel_A1[kernel]):
        S = src - mark
        src_z = int(S/feature_map_size)
        src_y = int((S - src_z*feature_map_size) / sqrt(feature_map_size))
        src_x = int(S - src_z*feature_map_size - src_y*sqrt(feature_map_size))
        for tar in range(mark, mark+neuron_num_each_kernel_A1[kernel]):
            T = tar - mark
            tar_z = int(T / feature_map_size)
            tar_y = int((T - tar_z*feature_map_size) / sqrt(feature_map_size))
            tar_x = int(T - tar_z*feature_map_size - tar_y*sqrt(feature_map_size))
            if src_x == tar_x and src_y == tar_y and src_z != tar_z:
                weightMatrix_A1A1[src,tar] = ihn
    mark += neuron_num_each_kernel_A1[kernel]
weightMatrix_A1A1 = weightMatrix_A1A1.reshape((n_layer_1*n_layer_1))
connections['A1A1'] = b.Synapses(neuron_groups['A1'], neuron_groups['A1'], 'w:1', on_pre='gi+=w')
connections['A1A1'].connect()
connections['A1A1'].w = weightMatrix_A1A1
end = time.time()
print ('time needed to create connection A1A1 (lateral inhibition):', end - start)

#create connections X1A1 (multi-scale locally connected - Inception style with adaptive STDP)
start = time.time()
weightMatrix_X1A1 = np.zeros((n_input, n_layer_1))

if Learning:
    mark = 0
    for kernel in range(kernel_num):
        feature_map_size = feature_map_size_each_kernel_A1[kernel]
        kernel_size = kernel_size_each_kernel_A1[kernel]
        stride = stride_each_kernel_A1[kernel]
        output_size = int(sqrt(feature_map_size))
        input_size = int(sqrt(n_input))
        
        for tar in range(mark, mark + neuron_num_each_kernel_A1[kernel]):
            T = tar - mark
            tar_z = int(T / feature_map_size)  # feature map index
            tar_spatial = T % feature_map_size
            tar_y = int(tar_spatial / output_size)
            tar_x = int(tar_spatial % output_size)
            
            # Calculate receptive field in input
            for ky in range(kernel_size):
                for kx in range(kernel_size):
                    src_y = tar_y * stride + ky
                    src_x = tar_x * stride + kx
                    if src_y < input_size and src_x < input_size:
                        src = src_y * input_size + src_x
                        weightMatrix_X1A1[src, tar] = 0.3 * np.random.rand() + wmin_ee
        
        mark += neuron_num_each_kernel_A1[kernel]

weightMatrix_X1A1 = weightMatrix_X1A1.reshape((n_input * n_layer_1))

if Learning:
    connections['X1A1'] = b.Synapses(neuron_groups['X1'], neuron_groups['A1'], 
                                      eqs_stdp_ee, on_pre=eqs_stdp_pre_ee, on_post=eqs_stdp_post_ee)
else:
    connections['X1A1'] = b.Synapses(neuron_groups['X1'], neuron_groups['A1'], 'w : 1', on_pre='ge+=w')

connections['X1A1'].connect()
connections['X1A1'].w = weightMatrix_X1A1
connections['X1A1'].delay = 'rand()*'+ str(Delay/b.ms) +'*ms'

if Learning:
    connections['X1A1'].eta_p = nu_ee_pre_init
    connections['X1A1'].eta_q = nu_ee_post_init

end = time.time()
print ('time needed to create connection X1A1:', end - start)
print ('X1A1 total synapses:', len(connections['X1A1'].w))
print ('X1A1 non-zero synapses:', np.sum(np.array(connections['X1A1'].w) > 0))

#create connections A1A2 (hidden to output - fully connected with adaptive STDP)
start = time.time()

if Learning:
    connections['A1A2'] = b.Synapses(neuron_groups['A1'], neuron_groups['A2'], 
                                      eqs_stdp_ee, on_pre=eqs_stdp_pre_ee, on_post=eqs_stdp_post_ee)
else:
    connections['A1A2'] = b.Synapses(neuron_groups['A1'], neuron_groups['A2'], 'w : 1', on_pre='ge+=w')

connections['A1A2'].connect()
connections['A1A2'].w = '0.3*rand()+wmin_ee'
connections['A1A2'].delay = 'rand()*'+ str(Delay/b.ms) +'*ms'

if Learning:
    connections['A1A2'].eta_p = nu_ee_pre_init
    connections['A1A2'].eta_q = nu_ee_post_init

end = time.time()
print ('time needed to create connection A1A2:', end - start)
print ('A1A2 synapses:', n_layer_1 * n_layer_2)

# --- monitoring helpers ---------------------------------------------------
prev_weights_x1a1 = np.array(connections['X1A1'].w).astype(float)
prev_weights_a1a2 = np.array(connections['A1A2'].w).astype(float)

def log_plasticity(iteration, accuracy_val=None):
    def stats(a):
        return np.mean(a), np.std(a), np.percentile(a,50), np.percentile(a,90)

    # X1A1 stats
    eta_p_x1a1 = np.array(connections['X1A1'].eta_p, dtype=float)
    eta_q_x1a1 = np.array(connections['X1A1'].eta_q, dtype=float)
    mean_p_x1a1, std_p_x1a1, p50_p_x1a1, p90_p_x1a1 = stats(eta_p_x1a1)
    mean_q_x1a1, std_q_x1a1, p50_q_x1a1, p90_q_x1a1 = stats(eta_q_x1a1)

    # A1A2 stats
    eta_p_a1a2 = np.array(connections['A1A2'].eta_p, dtype=float)
    eta_q_a1a2 = np.array(connections['A1A2'].eta_q, dtype=float)
    mean_p_a1a2, std_p_a1a2, p50_p_a1a2, p90_p_a1a2 = stats(eta_p_a1a2)
    mean_q_a1a2, std_q_a1a2, p50_q_a1a2, p90_q_a1a2 = stats(eta_q_a1a2)

    # Theta stats for both layers
    theta_a1 = np.array(neuron_groups['A1'].theta / b.mV, dtype=float)
    mean_theta_a1, std_theta_a1 = np.mean(theta_a1), np.std(theta_a1)
    
    theta_a2 = np.array(neuron_groups['A2'].theta / b.mV, dtype=float)
    mean_theta_a2, std_theta_a2 = np.mean(theta_a2), np.std(theta_a2)

    # Weight changes
    global prev_weights_x1a1, prev_weights_a1a2
    
    cur_w_x1a1 = np.array(connections['X1A1'].w, dtype=float)
    w_change_x1a1 = np.linalg.norm(cur_w_x1a1 - prev_weights_x1a1)
    w_change_avg_x1a1 = np.mean(np.abs(cur_w_x1a1 - prev_weights_x1a1))
    w_change_max_x1a1 = np.max(np.abs(cur_w_x1a1 - prev_weights_x1a1))

    cur_w_a1a2 = np.array(connections['A1A2'].w, dtype=float)
    w_change_a1a2 = np.linalg.norm(cur_w_a1a2 - prev_weights_a1a2)
    w_change_avg_a1a2 = np.mean(np.abs(cur_w_a1a2 - prev_weights_a1a2))
    w_change_max_a1a2 = np.max(np.abs(cur_w_a1a2 - prev_weights_a1a2))

    with open(plasticity_csv, 'a', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow([iteration, time.time(),
                       mean_p_x1a1, std_p_x1a1, p50_p_x1a1, p90_p_x1a1,
                       mean_q_x1a1, std_q_x1a1, p50_q_x1a1, p90_q_x1a1,
                       mean_p_a1a2, std_p_a1a2, p50_p_a1a2, p90_p_a1a2,
                       mean_q_a1a2, std_q_a1a2, p50_q_a1a2, p90_q_a1a2,
                       mean_theta_a1, std_theta_a1,
                       mean_theta_a2, std_theta_a2,
                       w_change_x1a1, w_change_avg_x1a1, w_change_max_x1a1,
                       w_change_a1a2, w_change_avg_a1a2, w_change_max_a1a2,
                       '' if accuracy_val is None else accuracy_val])
    
    prev_weights_x1a1 = cur_w_x1a1.copy()
    prev_weights_a1a2 = cur_w_a1a2.copy()

#create monitors
spike_counters['A1'] = b.SpikeMonitor(neuron_groups['A1'], record=True)
spike_counters['A2'] = b.SpikeMonitor(neuron_groups['A2'], record=True)

#create networks
net['M1'] = Network(
    neuron_groups['A1'], 
    neuron_groups['A2'],
    neuron_groups['X1'], 
    connections['X1A1'],
    connections['A1A1'],
    connections['A1A2'], 
    spike_counters['A1'],
    spike_counters['A2']
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
single_example_time = 0.35 * b.second
resting_time = 0.15 * b.second

#the interval of process data and show information
progress_interval = 10
validate_interval = 5000
save_interval = 500

#number of samples for training
n_train = 1000
train_begin = 0

#load trained weight to continue
if train_begin:
    connections['X1A1'].w = np.load(load_path + 'X1A1' + '_' + str(train_begin) + '.npy')
    connections['A1A2'].w = np.load(load_path + 'A1A2' + '_' + str(train_begin) + '.npy')
    neuron_groups['A1'].theta = np.load(load_path + 'theta_A1' + '_' + str(train_begin) + '.npy') * b.volt
    neuron_groups['A2'].theta = np.load(load_path + 'theta_A2' + '_' + str(train_begin) + '.npy') * b.volt

#the intensity of rate coding
intensity_step = 0.125
start_intensity = 0.25

#the threshold of retrain (based on output layer A2)
retrain_gate = 5 * n_layer_2  # Minimum spikes expected in output layer

# run the simulation and set inputs
previous_spike_count = {}
current_spike_count = {}
assignments = {}
result_monitor = {}
results_proportion = {}
accuracy = {}

# Track output layer A2 for classification
previous_spike_count['A2'] = np.zeros(n_layer_2)
current_spike_count['A2'] = np.zeros(n_layer_2)
assignments['A2'] = np.zeros(n_layer_2)
result_monitor['A2'] = np.zeros((validate_interval, n_layer_2))
results_proportion['A2'] = np.zeros((10, validate_interval))
accuracy['A2'] = []

# Also track hidden layer A1 for credit assignment
previous_spike_count['A1'] = np.zeros(n_layer_1)
current_spike_count['A1'] = np.zeros(n_layer_1)

input_numbers = np.zeros(validate_interval)

neuron_groups['X1'].rates = 0*b.hertz
net['M1'].run(0*b.second)

start = time.time()

j = train_begin
max_retries = 30
last_printed_j = None

input_intensity = start_intensity
while j < n_train:   
    
    if last_printed_j != j:
        last_printed_j = j
        retry_count = 0
     
    Rates = training['x'][j%60000,:,:].reshape((n_input)) * input_intensity

    neuron_groups['X1'].rates = Rates*b.hertz
    connections['X1A1'] = normalize_weights(connections['X1A1'], norm)
    connections['A1A2'] = normalize_weights(connections['A1A2'], norm)

    net['M1'].run(single_example_time)
    
    # Track spikes in output layer A2
    current_spike_count['A2'] = np.asarray(spike_counters['A2'].count[:]) - previous_spike_count['A2']
    previous_spike_count['A2'] = np.copy(spike_counters['A2'].count[:])
    
    # Track spikes in hidden layer A1 (for credit assignment)
    current_spike_count['A1'] = np.asarray(spike_counters['A1'].count[:]) - previous_spike_count['A1']
    previous_spike_count['A1'] = np.copy(spike_counters['A1'].count[:])
    
    spike_num = np.sum(current_spike_count['A2'])

    if spike_num < retrain_gate:
        retry_count += 1
        
        if retry_count >= max_retries:
            print(f"  Skipping sample {j} after {retry_count} retries (spike_num={int(spike_num)} < {int(retrain_gate)})")
            input_intensity = start_intensity
            neuron_groups['X1'].rates = 0*b.hertz
            neuron_groups['A1'].v = v_rest_e - 40. * b.mV
            neuron_groups['A2'].v = v_rest_e - 40. * b.mV
            net['M1'].run(resting_time)
            j += 1
            continue
        
        input_intensity += intensity_step
        neuron_groups['X1'].rates = 0*b.hertz
        net['M1'].run(resting_time)
    else:
        result_monitor['A2'][j%validate_interval,:] = current_spike_count['A2']
        input_numbers[j%validate_interval] = training['y'][j%60000][0]

        # ============================================================
        # TIMING-BASED CREDIT ASSIGNMENT: A1→A2 performance → X1→A1
        # ============================================================
        if j%5 == 0:
            print('Applying timing-based updates A1→A2 to X1→A1... ==========')
            applied_updates_x1a1 = attribute_timing_to_upstream_synapse(
                spike_counters['A2'],  # downstream (output) spikes
                spike_counters['A1'],  # upstream (hidden) spikes
                connections['A1A2'],   # downstream connection
                connections['X1A1'],   # upstream connection to update
                float(timing_threshold / b.ms),
                epsilon_timing,
                use_fast_mode=fast_timing_updates,
                max_active_synapses=max_active_synapses,
                time_window_ms=time_window_ms,
                subsample_rate=subsample_rate
            )

            # Apply updates to X1A1
            try:
                w_arr_x1a1 = np.array(connections['X1A1'].w, dtype=float)
                n_updates_x1a1 = 0
                sum_abs_dw_x1a1 = 0.0
                max_abs_dw_x1a1 = 0.0
                if isinstance(applied_updates_x1a1, list):
                    for syn_idx, dw in applied_updates_x1a1:
                        try:
                            syn_i = int(syn_idx)
                            dwf = float(dw)
                            w_arr_x1a1[syn_i] = np.clip(w_arr_x1a1[syn_i] + dwf, wmin_ee, wmax_ee)
                            n_updates_x1a1 += 1
                            sum_abs_dw_x1a1 += abs(dwf)
                            if abs(dwf) > max_abs_dw_x1a1:
                                max_abs_dw_x1a1 = abs(dwf)
                        except Exception:
                            continue
                connections['X1A1'].w = w_arr_x1a1
                print(f"  Applied {n_updates_x1a1} timing updates to X1A1, sum|dw|={sum_abs_dw_x1a1:.6f}")
            except Exception as e:
                print(f"  Error applying timing updates: {e}")
                n_updates_x1a1 = 0
                sum_abs_dw_x1a1 = 0.0
                max_abs_dw_x1a1 = 0.0
            
            # Log timing updates
            with open(timing_updates_csv, 'a', newline='') as f:
                wcsv = csv.writer(f)
                wcsv.writerow([j, time.time(), 'X1A1', n_updates_x1a1, sum_abs_dw_x1a1, max_abs_dw_x1a1])
        
        neuron_groups['X1'].rates = 0*b.hertz
        net['M1'].run(resting_time)
        input_intensity = start_intensity

        j += 1
        
        if j%20 == 0:
            print("=== Diagnostics at iteration", j, "===")
            print(f"Mean theta A1 (hidden): {np.mean(neuron_groups['A1'].theta/b.mV):.3f} mV")
            print(f"Mean theta A2 (output): {np.mean(neuron_groups['A2'].theta/b.mV):.3f} mV")
            print(f"Mean weight X1A1: {np.mean(connections['X1A1'].w):.6f}")
            print(f"Mean weight A1A2: {np.mean(connections['A1A2'].w):.6f}")
            print(f"Mean eta_p X1A1: {np.mean(connections['X1A1'].eta_p):.6f}")
            print(f"Mean eta_p A1A2: {np.mean(connections['A1A2'].eta_p):.6f}")
            print(f"A1 spikes this sample: {int(np.sum(current_spike_count['A1']))}")
            print(f"A2 spikes this sample: {int(np.sum(current_spike_count['A2']))}")
        
        if j % progress_interval == 0:
            print ('Progress: ', j, '/', n_train, '(', time.time() - start, 'seconds)')
            start = time.time()
            
        if j % validate_interval == 0:
            # Classification based on output layer A2
            assignments['A2'] = get_new_assignments(result_monitor['A2'][:], input_numbers[:])
            test_results = np.zeros((10, validate_interval))
            for k in range(validate_interval):
                results_proportion['A2'][:,k] = get_recognized_number_proportion(assignments['A2'], result_monitor['A2'][k,:])
                test_results[:,k] = np.argsort(results_proportion['A2'][:,k])[::-1]
            difference = test_results[0,:] - input_numbers[:]
            correct = len(np.where(difference == 0)[0])
            acc = correct/float(validate_interval) * 100
            accuracy['A2'].append(acc)
            print ('Validate accuracy: ', acc, '(last)', np.max(accuracy['A2']), '(best)')

            log_plasticity(j, accuracy_val=acc)
            
        if j % save_interval == 0:
            np.save(save_path + 'X1A1' + '_' + str(j), connections['X1A1'].w)
            np.save(save_path + 'A1A2' + '_' + str(j), connections['A1A2'].w)
            np.save(save_path + 'theta_A1' + '_' + str(j), neuron_groups['A1'].theta)
            np.save(save_path + 'theta_A2' + '_' + str(j), neuron_groups['A2'].theta)
            log_plasticity(j, accuracy_val=None)

print('='*60)
print('Training complete!')
print(f'Final accuracy: {accuracy["A2"][-1] if accuracy["A2"] else "N/A"}%')
print(f'Best accuracy: {np.max(accuracy["A2"]) if accuracy["A2"] else "N/A"}%')
print('='*60)