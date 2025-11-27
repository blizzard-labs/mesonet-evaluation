import numpy as np
import os.path
from Functions import get_new_assignments, get_recognized_number_proportion

# specify the location
load_path = './activity/'
load_ending = '990'       #specify which weight file you want to load from load_path (only ending number needed)
n_assignment = 100
n_test = 100

print('Loading data from:', load_path)
print('Using weights from run:', load_ending)
print('Number of assignment samples:', n_assignment)
print('Number of test samples:', n_test)

# Load assignment data
try:
    activity_assignment = np.load(load_path + 'Activity_assignment_A1' +'_'+ str(n_assignment) + load_ending + '.npy')
    labels_assignment = np.load(load_path + 'Labels_assignment' + str(n_assignment) + load_ending + '.npy')
    print('Assignment data loaded successfully.')
except IOError as e:
    print('Error loading assignment data:', e)
    print('Please run Test.py first to generate the activity files.')
    exit()

# Load testing data
try:
    activity_testing = np.load(load_path + 'Activity_testing_A1' +'_'+ str(n_test) + load_ending + '.npy')
    labels_testing = np.load(load_path + 'Labels_testing' + str(n_test) + load_ending + '.npy')
    print('Testing data loaded successfully.')
except IOError as e:
    print('Error loading testing data:', e)
    print('Please run Test.py first to generate the activity files.')
    exit()


# Get assignments for the output neurons
print('Assigning labels to output neurons...')
assignments = get_new_assignments(activity_assignment, labels_assignment)
print('Assignments complete.')

# Evaluate the performance on the test set
print('Evaluating performance on the test set...')
correct_predictions = 0
total_predictions = n_test

for i in range(total_predictions):
    # Get the spike rates for the current test image
    spike_rates = activity_testing[i,:]
    
    # Get the recognized number proportion
    proportions = get_recognized_number_proportion(assignments, spike_rates)
    
    # Predict the digit
    prediction = np.argmax(proportions)
    
    # Check if the prediction is correct
    if prediction == labels_testing[i]:
        correct_predictions += 1

# Calculate accuracy
accuracy = (correct_predictions / float(total_predictions)) * 100 if total_predictions > 0 else 0
print('---------------------------------------------')
print('Evaluation finished.')
print('Accuracy: %.2f%%' % accuracy)
print('---------------------------------------------')