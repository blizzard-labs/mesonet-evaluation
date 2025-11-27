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
prefs.codegen.target = 'auto'
prefs.codegen.cpp.extra_compile_args_gcc = ['-march=native']

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=0.00075)
parser.add_argument('--timing_threshold', type=float, default=7.0)   # ms
parser.add_argument('--epsilon_timing', type=float, default=0.001)
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
                'mean_eta_p_x1a2','std_eta_p_x1a2','p50_eta_p_x1a2','p90_eta_p_x1a2',
                'mean_eta_q_x1a2','std_eta_q_x1a2','p50_eta_q_x1a2','p90_eta_q_x1a2',
                'mean_eta_p_a2a1','std_eta_p_a2a1','p50_eta_p_a2a1','p90_eta_p_a2a1',
                'mean_eta_q_a2a1','std_eta_q_a2a1','p50_eta_q_a2a1','p90_eta_q_a2a1',
                'mean_theta_mv','std_theta_mv',
                'weight_change_norm_x1a2', 'weight_change_avg_x1a2', 'weight_change_max_x1a2',
                'weight_change_norm_a2a1', 'weight_change_avg_a2a1', 'weight_change_max_a2a1',
                'accuracy'])
timing_updates_csv = os.path.join(log_dir, 'timing_updates_log.csv')
# write header for timing updates
with open(timing_updates_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['iter', 'time', 'n_updates', 'sum_abs_dw', 'max_abs_dw'])

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
neuron_groups['A2'] = b.NeuronGroup(n_layer_2, neuron_eqs_e, method='euler', threshold=thresh_e, refractory=refrac_e, reset= scr_e)
neuron_groups['A2'].v = v_rest_e - 40. * b.mV
neuron_groups['A2'].theta = np.ones((n_layer_2)) * 20.0*b.mV
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

#create connections X1A2
start = time.time()
if Learning:
    connections['X1A2'] = b.Synapses(neuron_groups['X1'], neuron_groups['A2'], eqs_stdp_ee, on_pre=eqs_stdp_pre_ee, on_post=eqs_stdp_post_ee)
else:
    connections['X1A2'] = b.Synapses(neuron_groups['X1'], neuron_groups['A2'], 'w : 1', on_pre='ge+=w')
connections['X1A2'].connect()
connections['X1A2'].w = '0.3*rand()+wmin_ee'
connections['X1A2'].delay = 'rand()*'+ str(Delay/b.ms) +'*ms'
if Learning:
    connections['X1A2'].eta_p = nu_ee_pre_init
    connections['X1A2'].eta_q = nu_ee_post_init
print ('time needed to create connection X1A2:', time.time() - start)

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
prev_weights_x1a2 = np.array(connections['X1A2'].w).astype(float)
prev_weights_a2a1 = np.array(connections['A2A1'].w).astype(float)

def log_plasticity(iteration, accuracy_val=None):
    def stats(a):
        return np.mean(a), np.std(a), np.percentile(a,50), np.percentile(a,90)

    eta_p_x1a2 = np.array(connections['X1A2'].eta_p, dtype=float)
    eta_q_x1a2 = np.array(connections['X1A2'].eta_q, dtype=float)
    mean_p_x1a2, std_p_x1a2, p50_p_x1a2, p90_p_x1a2 = stats(eta_p_x1a2)
    mean_q_x1a2, std_q_x1a2, p50_q_x1a2, p90_q_x1a2 = stats(eta_q_x1a2)

    eta_p_a2a1 = np.array(connections['A2A1'].eta_p, dtype=float)
    eta_q_a2a1 = np.array(connections['A2A1'].eta_q, dtype=float)
    mean_p_a2a1, std_p_a2a1, p50_p_a2a1, p90_p_a2a1 = stats(eta_p_a2a1)
    mean_q_a2a1, std_q_a2a1, p50_q_a2a1, p90_q_a2a1 = stats(eta_q_a2a1)

    theta_vals = np.array(neuron_groups['A1'].theta / b.mV, dtype=float)
    mean_theta, std_theta = np.mean(theta_vals), np.std(theta_vals)

    cur_w_x1a2 = np.array(connections['X1A2'].w, dtype=float)
    w_change_x1a2 = np.linalg.norm(cur_w_x1a2 - prev_weights_x1a2)
    w_change_avg_x1a2 = np.mean(np.abs(cur_w_x1a2 - prev_weights_x1a2))
    w_change_max_x1a2 = np.max(np.abs(cur_w_x1a2 - prev_weights_x1a2))

    cur_w_a2a1 = np.array(connections['A2A1'].w, dtype=float)
    w_change_a2a1 = np.linalg.norm(cur_w_a2a1 - prev_weights_a2a1)
    w_change_avg_a2a1 = np.mean(np.abs(cur_w_a2a1 - prev_weights_a2a1))
    w_change_max_a2a1 = np.max(np.abs(cur_w_a2a1 - prev_weights_a2a1))

    with open(plasticity_csv, 'a', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow([iteration, time.time(),
                       mean_p_x1a2, std_p_x1a2, p50_p_x1a2, p90_p_x1a2,
                       mean_q_x1a2, std_q_x1a2, p50_q_x1a2, p90_q_x1a2,
                       mean_p_a2a1, std_p_a2a1, p50_p_a2a1, p90_p_a2a1,
                       mean_q_a2a1, std_q_a2a1, p50_q_a2a1, p90_q_a2a1,
                       mean_theta, std_theta,
                       w_change_x1a2, w_change_avg_x1a2, w_change_max_x1a2,
                       w_change_a2a1, w_change_avg_a2a1, w_change_max_a2a1,
                       '' if accuracy_val is None else accuracy_val])
    
    nonlocal_prev_weights_assign(cur_w_x1a2, cur_w_a2a1)

def nonlocal_prev_weights_assign(arr_x1a2, arr_a2a1):
    global prev_weights_x1a2, prev_weights_a2a1
    prev_weights_x1a2 = arr_x1a2.copy()
    prev_weights_a2a1 = arr_a2a1.copy()

#create monitors
spike_counters['A1'] = b.SpikeMonitor(neuron_groups['A1'], record=True)
spike_counters['A2'] = b.SpikeMonitor(neuron_groups['A2'], record=True)

#create networks
net['M1'] = Network(neuron_groups['A1'], neuron_groups['A2'], neuron_groups['X1'], connections['X1A2'], connections['A2A1'], connections['A1A1'], spike_counters['A1'], spike_counters['A2'])
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
single_example_time =   0.35 * b.second
resting_time = 0.15 * b.second

#the the interval of process data and show information
progress_interval = 10
validate_interval = 5000   #no less than 2000
save_interval = 30

#number of samples for training
n_train = 1000
train_begin = 0    #specify which iteration you want the training to begin from 

#load trained weight to continue
if train_begin:
    connections['X1A2'].w = np.load(load_path + 'X1A2' + '_' + str(train_begin) + '.npy')
    connections['A2A1'].w = np.load(load_path + 'A2A1' + '_' + str(train_begin) + '.npy')
    neuron_groups['A1'].theta = np.load(load_path + 'theta_A1' + '_' + str(train_begin) + '.npy') *b.volt
    neuron_groups['A2'].theta = np.load(load_path + 'theta_A2' + '_' + str(train_begin) + '.npy') *b.volt

#the intensity of rate coding
intensity_step = 0.125
start_intensity = 0.25

#the threshold of retrain
retrain_gate = np.sum([5*feature_map_size_each_kernel[kernel] for kernel in range(kernel_num)])

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
input_intensity = start_intensity
while j < n_train:

    Rates = training['x'][j%60000,:,:].reshape((n_input)) * input_intensity

    neuron_groups['X1'].rates = Rates*b.hertz
    connections['X1A2'] = normalize_weights(connections['X1A2'],norm)
    connections['A2A1'] = normalize_weights(connections['A2A1'],norm)

    net['M1'].run(single_example_time)
    
    current_spike_count['A1'] = np.asarray(spike_counters['A1'].count[:])- previous_spike_count['A1']
    previous_spike_count['A1'] = np.copy(spike_counters['A1'].count[:])
    
    spike_num = np.sum(current_spike_count['A1'])

    if spike_num < retrain_gate:
        input_intensity += intensity_step
        neuron_groups['X1'].rates = 0*b.hertz
        net['M1'].run(resting_time)
    else:
        result_monitor['A1'][j%validate_interval,:] = current_spike_count['A1']
        input_numbers[j%validate_interval] = training['y'][j%60000][0]

        # Apply timing-based attribution to upstream synapses for A2A1
        applied_updates = attribute_timing_to_upstream_synapse(
            spike_counters['A1'],
            spike_counters['A2'],
            connections['A2A1'],
            connections['X1A2'],
            float(timing_threshold / b.ms),
            epsilon_timing,
            use_fast_mode=fast_timing_updates,
            max_active_synapses=max_active_synapses,
            time_window_ms=time_window_ms,
            subsample_rate=subsample_rate
        )
        
        try:
            w_arr = np.array(connections['X1A2'].w, dtype=float)
            n_updates = 0
            sum_abs_dw = 0.0
            max_abs_dw = 0.0
            if isinstance(applied_updates, list):
                for syn_idx, dw in applied_updates:
                    try:
                        syn_i = int(syn_idx)
                        dwf = float(dw)
                        w_arr[syn_i] = np.clip(w_arr[syn_i] + dwf, wmin_ee, wmax_ee)
                        n_updates += 1
                        sum_abs_dw += abs(dwf)
                        if abs(dwf) > max_abs_dw:
                            max_abs_dw = abs(dwf)
                    except Exception:
                        continue
            connections['X1A2'].w = w_arr
        except Exception:
            n_updates = 0
            sum_abs_dw = 0.0
            max_abs_dw = 0.0
        
        with open(timing_updates_csv, 'a', newline='') as f:
            wcsv = csv.writer(f)
            wcsv.writerow([j, time.time(), n_updates, sum_abs_dw, max_abs_dw])

        neuron_groups['X1'].rates = 0*b.hertz
        net['M1'].run(resting_time)
        input_intensity = start_intensity

        j += 1
        
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
            np.save(save_path + 'X1A2' + '_' + str(j), connections['X1A2'].w)
            np.save(save_path + 'A2A1' + '_' + str(j), connections['A2A1'].w)
            np.save(save_path + 'theta_A1' + '_' + str(j), neuron_groups['A1'].theta)
            np.save(save_path + 'theta_A2' + '_' + str(j), neuron_groups['A2'].theta)
            log_plasticity(j, accuracy_val=None)
