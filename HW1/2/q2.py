import numpy as np
import matplotlib.pyplot as plt
import math
import random
import copy

def closed_form(y,x):
    b=0
    A=0
    features = x.shape[1]
    for i in range(len(y)):
        b += np.dot(y[i], x[i].reshape(1,features))
        A += np.dot(x[i].reshape(1,features), x[i].reshape(1,features).T)
    b = 1/len(y) * b
    A = 1/len(y) * A
    return b,A

def closed_form_2(y,x):
    return np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))

def empiricial_risk(y,x,theta):
    loss = 0
    # print(x[1].shape)
    for i in range(len(y)):
        loss += math.pow(y[i]- np.dot(theta, x[i]),2)/2
    return 1/(2 * len(y))*loss

def batch_gradient_descent(theta,learning_rate, b, A):
    theta = theta - learning_rate * ( -b + np.dot(A, theta))
    return theta

def stochastic_gradient_descent(theta,learning_rate, y, x):
    point = random.randint(0, len(y)-1)
    x = np.array([x[point][0],x[point][1]]).reshape(2,1)
    y = np.array([y[point][0]]).reshape(1,1)
    theta = theta + learning_rate * np.dot((y - np.dot(theta, x)), x.reshape(1,2))
    return theta

def PolyRegress(x,y,d):
    inputs = copy.deepcopy(x)
    for i in range(2,d+1):
        new_features = 0
        f = lambda x: x ** i
        new_features = f(x)
        inputs = np.c_[inputs,new_features]
    theta = closed_form_2(y, inputs).reshape(1,d)
    training_error = empiricial_risk(y,inputs,theta)
    return inputs, theta, training_error


if __name__=="__main__":
    x_1 = np.genfromtxt('hw1x.dat', delimiter='\t', dtype = float).reshape(200,1)
    x_2 = np.array([[1 for i in range(len(x_1))]]).T
    y = np.genfromtxt('hw1y.dat', delimiter='\t', dtype = float).reshape(200,1)
    
    x = np.append(x_1,x_2,axis=1).reshape(200,2)
    b, A = closed_form(y,x)
    theta = np.dot(np.linalg.inv(A),b.reshape(1,2))
    print("Theta closed form method 1:", theta)

    theta_2 = closed_form_2(y,x).reshape(1,2)
    print("Theta closed form method 2:", theta_2)

    training_error = empiricial_risk(y,x,theta)
    print("Training error method 1:",training_error)
    training_error_2 = empiricial_risk(y,x,theta_2)
    print("Training error method 2:",training_error_2)
    plt.plot(x_1, y, 'ro')

    y_r = np.dot(theta,x.T).T
    y_r_2 = np.dot(theta_2,x.T).T
    x_r = np.arange(0,2,0.01)
    plt.plot(x_r,y_r, label = "Closed form 1")
    plt.plot(x_r,y_r_2, label = "Closed form 2")
    plt.xlim(0,2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    #batch gradient descent
    print("---------BATCH GRADIENT DESCENT---------")
    learning_rate = 0.01
    update = 5
    theta = np.array([0, 0]).reshape(1,2)

    for i in range(update):
        theta = batch_gradient_descent(theta, learning_rate, b, A)
    print("Theta:", theta)
    training_error = empiricial_risk(y,x,theta)
    print("Training error:",training_error)


    #stochastic gradient descent
    print("---------STOCHASTIC GRADIENT DESCENT---------")
    learning_rate = 0.01
    update = 5
    theta = np.array([0, 0]).reshape(1,2)

    for i in range(update):
        theta = stochastic_gradient_descent(theta, learning_rate, y, x)
    print("Theta:", theta)
    training_error = empiricial_risk(y,x,theta)
    print("Training error:",training_error)


    print("---------Polynomial Regression---------")
    error_list = []
    num_features = []
    for j in range(2,16):
        x, theta, training_error = PolyRegress(x_1,y,j)
        y_r = np.dot(theta,x.T).T
        x_r = np.arange(0,2,0.01)
        plt.plot(x_r,y_r)
        error_list.append(training_error)
        num_features.append(j)
    print("Training Errors:", error_list)
    plt.plot(x_1, y, 'ro')
    plt.xlim(0,2)
    plt.ylim(0,10)
    plt.show()
    plt.plot(num_features,error_list)
    plt.ylim(0,2)
    plt.show()
    
