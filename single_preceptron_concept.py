import numpy as np 
import argparse

## OR gate implementation
# X_train=np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]]) ## 1st column indicates bias term
X_train=np.array([[0,0],[0,1],[1,0],[1,1]]) ## includes just feature of the input only
Y_train = np.array([0,1,1,1])

def step_func(z):
    return 1.0 if (z > 0) else 0.0

def linear_func(test_input, weight):
    biased_input = np.insert(test_input,0,1).reshape(-1,1) #insert bias term in the first column; it is not necessary to reshape when training with single example
    linear_output = np.dot(weight, biased_input) 
    return (biased_input, linear_output)

## training without vectorization
def train_perceptron(X_train, Y_train, lr, epochs):
    m,n = X_train.shape ## m = no of training examples ; n = no of features in each training example
    # weight initialization denoted as theta
    # theta = np.zeros((1, n)) ## 1st column for bias and 2nd and 3rd column for input feature weight
    theta = np.zeros((1, n+1)) ## 1st column for bias and 2nd and 3rd column for input feature weight ## 1 row and n+1 no of column
    for a in range(epochs): ## no of epochs
        for i in range(m): ## iterationg for each training example
            # biased_input = np.insert(X_train[m],0,1) ## insert bias term as 1 in the 1st column of input feature and then transpose it
            # biased_input = np.insert(X_train[i],0,1).reshape(-1,1) #insert bias term in the first column; if not reshped then also okay
            # linear_output = np.dot(theta, biased_input) 
            biased_input, linear_output = linear_func(X_train[i], theta)
            y_hat = step_func(linear_output)
            
            if(linear_output - Y_train[i]) != 0:
                theta += lr*(Y_train[i]-y_hat)*biased_input.T
                # print("weights updated as",theta)
        print(f'{a+1}/{epochs} epoch completed')
    # print("final weights are",theta)
    return theta



###
def test_func(test_data, trained_weights):
    linear_output = linear_func(test_data, trained_weights)[1]
    prediction = step_func(linear_output) 
    print("predicted output is:")
    print(prediction)
    return prediction


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("test_data", help="insert test_data for or gate. eg [0,0] or [0,1] and so on.")
    args = parser.parse_args()
    thetas = train_perceptron(X_train, Y_train, 0.5, 3)
    test = eval(args.test_data)
    test_func(test, thetas)