import numpy as np

np.set_printoptions(suppress=True)


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1.0 - x)
    return 1.0 / (1 + np.exp(-x))


def relu(x, derivative=False):
    if derivative:

        x = np.copy(x)

        x[x>0] = 1
        x[x<0] = 0
 
        return x
    return np.maximum(x*0, x)


import numpy as np

np.set_printoptions(suppress=True)


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1.0 - x)
    return 1.0 / (1 + np.exp(-x))

def relu(x, derivative=False):
    if derivative:

        x = np.copy(x)

        x[x>0] = 1
        x[x<0] = 0
 
        return x
    return np.maximum(x*0, x)

class Network:
    def __init__(self, layer_structures, activation=sigmoid, iterations=1000, learning_rate=None):

        
        self.iterations = iterations
        
        #self.activation_function = activation
        
        self.layer_structures = layer_structures
        inputCount = layer_structures[0]

        self.inputCount = inputCount
        layerCount = len(layer_structures)-1
        self.layers = [None] * layerCount

        lastOutput = inputCount
        for i in range(layerCount):
            outputs = layer_structures[i+1]

#             if i == 0:
#                 outputs = inputCount

            layer = np.random.random((lastOutput+1, outputs))
            self.layers[i] = layer
            lastOutput = outputs
           
        if callable(activation):
            self.activation_function = activation
            self.learning_rate = 1
        elif activation.lower() == 'relu':
            self.activation_function = relu
            self.learning_rate = .001
        elif activation.lower() == 'sigmoid':
            self.activation_function = sigmoid
            self.activation = 1
            
        if learning_rate != None:
            self.learning_rate = learning_rate
            
    def reset_layers(self):
        layer_structures = self.layer_structures
        inputCount = layer_structures[0]

        self.inputCount = inputCount
        layerCount = len(layer_structures)-1
        self.layers = [None] * layerCount

        lastOutput = inputCount
        for i in range(layerCount):
            outputs = layer_structures[i+1]

#             if i == 0:
#                 outputs = inputCount

            layer = np.random.random((lastOutput+1, outputs))
            self.layers[i] = layer
            lastOutput = outputs
        
        
    def predict(self, in_set, classification=True):

        if False:
            out = [np.array(in_set)]
            last_out = np.array(in_set)

            for i, layer in enumerate(self.layers):
                last_out = self.activation_function(np.dot(last_out, layer))
                out.append(last_out)

            return out
        else:
            last_out = self.add_bias(np.array(in_set))
            for i, layer in enumerate(self.layers):
                last_out = self.add_bias(self.activation_function(np.dot(last_out, layer)))

            if classification:
                np.argmax(np.delete(last_out, [len(last_out[0])-1], axis=1), axis=1)

            return np.delete(last_out, [len(last_out[0])-1], axis=1)
        
    def add_bias(self, arr):
        
        b = np.array([np.ones(len(arr))])
        return np.concatenate((arr, b.T), axis=1)

    def fit(self, in_sets, out_sets, reset_layers=True):
        if len(in_sets) != len(out_sets):
            exit('TRAINING SET COUNT MISMATCH')
            
        if reset_layers:
            self.reset_layers()

        for iter in range(self.iterations):

            out = [np.array(in_sets)]
            last_out = self.add_bias(np.array(in_sets))

            for i, layer in enumerate(self.layers):
                last_out = self.add_bias(self.activation_function(np.dot(last_out, layer)))
                out.append(last_out)

            out_layers = out
            
            out_layers[-1] = np.delete(out_layers[-1], [len(out_layers[-1][0])-1], axis=1)
            #print(out_layers[-1].shape)

            last_error = out_sets - out_layers[-1]

            last_delta = last_error * self.activation_function(out_layers[-1], True)

            deltas = [last_delta]
            
            flag = False

            for k in range(len(self.layers) - 1, 0, -1):
                #print(k, "ld", last_delta, '\nld shape', last_delta.shape, "layer shape", self.layers[k].shape)
                
                if flag:
                    last_delta = last_delta.T[:-1].T
                else:
                    flag = True
                layer_error = last_delta.dot(self.layers[k].T)

                layer_delta = layer_error * self.activation_function(out_layers[k], True)

                last_delta = layer_delta

                deltas.append(layer_delta)

                
            flag = False
            for k in range(len(self.layers) - 1, 0, -1):
                #print("d", deltas[-k-1].shape, "o", out_layers[k].shape, "w", self.layers[k].shape)
                
                change = out_layers[k].T.dot(deltas[-k - 1])
                
                if flag:
                    change = change.T[:-1].T
                else:
                    flag = True
                
                
                
                #print("change", change.shape)
                
                self.layers[k] += change * self.learning_rate

    def storage(self):
        out = []
        for layer in self.layers:
            out.append(layer.tolist())

        return out

    def load(self, layers):
        out = []
        for layer in layers:
            out.append(np.array(layer))
        self.layers = out

    def __repr__(self):
        return str(self.layers)

    def __str__(self):
        return self.__repr__()

if __name__ == '__main__':

    net = Network([2, 3, 1], iterations=10000)

    ins = [[1,1], [0,0], [1,0], [0,1]]
    outs = [[1], [0], [0], [0]]

    net.fit(ins, outs)

    print(net.predict(ins))
