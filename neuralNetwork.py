import activation
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sn


class NeuralNetwork():
    #two layer neural network

    def __init__(self, layer_size, learning_rate,iterations):
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.x = None
        self.y = None


    def initialize_params(self, inputs, layer_size, outputs):
        self.params['w1'] = np.random.randn(layer_size, inputs.shape[0]) * 0.01
        self.params['b1'] = np.zeros((layer_size, 1))
        self.params['w2'] = np.random.randn(outputs.shape[0], layer_size) * 0.01
        self.params['b2'] = np.zeros((outputs.shape[0], 1))


    def forward_propagation(self, x_train, parameters):
        # Found z1 multiplying inputs with the weights and collecting bias. z1 = (x1*w1) + b1
        z1 = self.np.dot(self.parameters['w1'], x_train) + self.parameters['b1']
        a1 = self.tanh(z1) #hidden layer
        z2 = self.np.dot(self.parameters['w2'],a1) + self.parameters['b2']
        a2 = self.sigmoid(z2) #output layer

        result = {  "z1":z1,
                    "a1":a1,
                    "z1":z1,
                    "z2":z2
        } # save calculated parameters 

        return a2,result


    def calculate_cost(self,a2,y):
        self.loss = self.np.multiply(np.log(a2),y)
        self.cost = -(self.np.sum(loss) / y.shape[1])
        return cost


    def back_propagation(self,parameters,result,x,y):
        # Computes the derivatives and update weights and bias according
        dl_z2 = result['a2'] - y
        dl_w2 = np.dot(dl_z2,result['a1'].T)/x.shape[1]
        dl_b2 = np.sum(dl_z2,axis=1,keepdims=True)/x.shape[1]
        dl_z1 = np.dot(parameters['w2'].T,dl_z2)*(1-np.power(result['a1'],2))
        dl_w1 = np.dot(dl_z1,x.T)/x.shape[1]
        dl_b1=np.sum(dl_z1,axis=1,keepdims=True)/x.shape[1]

        grads={
        "dweight1":dl_w1,
        "dbias1":dl_b1,
        "dweight2":dl_w2,
        "dbias2":dl_b2}
        return grads


    def update_parameters(self):
        return

    


    def two_layer_neural_network(self,x_train,y_train,x_test,y_test,iteration,hidden_layer_size):

        parameters = initialize_params(x_train,hidden_layer_size,y_train)

        for i in range(0,iteration):
            # forward propagation
            a2,result=forward_propagation(x_train,parameters)

            #cost
            cost = calculate_cost(a2,y)

            # back propagation
            grads=back_propagation(parameters,result,x_train,y_train) 

            #update parameters ????
