from csv import reader
import numpy as np

def perceptron_algo(theta, x, label, offset, testing):
    h = label * (np.matmul(x,theta)+offset)
    theta_sym = theta[0][0]
    theta_int = theta[1][0]
    if np.sign([h]) < 0:
        if testing:
            return 0
        theta_sym = theta[0][0] + label * x[0]
        theta_int = theta[1][0] + label * x[1]
        offset = offset+label
    if testing:
        return 1
    return np.array([[theta_sym],[theta_int]]),offset

if __name__=='__main__':
    theta = np.array([[0],[0]]) # initial theta values (theta 0)
    EPOCHS = 1 # Change this value to fix number of epochs: 1 for qn 1a, 5 for qn 1b
    offset = 0 # Change this value to fix offset

    epoch_count = 0
    correct_values = 0
    values_count = 0

    #TRAINING
    with open('train_1_5.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        while epoch_count < EPOCHS:
            for values in csv_reader:
                symmetry = float(values[0])
                intensity = float(values[1])
                x = np.array([symmetry,intensity])
                label = float(values[2])
                theta, offset = perceptron_algo(theta, x, label, offset, testing = False)
            epoch_count += 1
        print("Training complete, Theta value is:", theta)

    #TESTING
    result = 0
    with open('test_1_5.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for values in csv_reader:
                values_count += 1
                symmetry = float(values[0])
                intensity = float(values[1])
                x = np.array([symmetry,intensity])
                label = float(values[2])
                result = perceptron_algo(theta, x, label, offset, testing = True)
                correct_values += result
        accuracy = correct_values/values_count * 100
        print("Theta", theta, "Final offset", offset, 'Accuracy:', accuracy)
