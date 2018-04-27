import numpy as np

np.set_printoptions(suppress=True)


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1.0 - x)
    return 1.0 / (1 + np.exp(-x))


def relu(x, derivative=False):
    if derivative:
        return np.ones(x.shape) * np.maximum(x*0, x)
    return np.maximum(x*0, x)


class Network:

    def __init__(self, layer_structures, iterations=100, activation_function='sigmoid', learning_rate=.001):
        self.layer_structures = layer_structures
        self.learning_rate = learning_rate
        self.iterations = iterations

        if activation_function.lower() == 'relu':
            self.activation_function = relu
        else:
            self.activation_function = sigmoid


    def init_network_layers(self):
        inputCount = self.layer_structures[0]
        layerCount = len(self.layer_structures)
        self.layers = [None] * layerCount
        lastOutput = inputCount

        for i in range(layerCount):
            outputs = self.layer_structures[i]

            if i == 0:
                outputs = inputCount

            layer = 2 * np.random.random((lastOutput+1, outputs)) - 1
            self.layers[i] = layer
            lastOutput = outputs


    # TODO: classification flag isn't used??
    def predict(self, in_set, all_layers=False, classification=False):
        if all_layers:
            out = [np.array(in_set)]
            last_out = np.array(in_set)

            for i, layer in enumerate(self.layers):
                # TODO: replace sigmoid?
                last_out = self.activation_function(np.dot(last_out, layer))
                out.append(last_out)

            return out
        else:
            last_out = self.add_bias(np.array(in_set))
            for i, layer in enumerate(self.layers):
                # TODO: replace sigmoid?
                last_out = self.add_bias(self.activation_function(np.dot(last_out, layer)))

            if classification:
                return np.argmax(np.delete(last_out, [len(last_out[0])-1], axis=1), axis=1)
            else:
                return np.delete(last_out, [len(last_out[0])-1], axis=1)


    def add_bias(self, arr):
        b = np.array([np.ones(len(arr))])
        return np.concatenate((arr, b.T), axis=1)


    def fit(self, X, y, reset_layers=True):
        if len(X) != len(y):
            exit('TRAINING SET COUNT MISMATCH')

        # if reset_layers is set to False, reuse the
        # layers from the previous call to fit
        if reset_layers:
            self.init_network_layers()

        X_to_arr = np.array(X)

        for iter in range(self.iterations):
            out = [X_to_arr]
            last_out = self.add_bias(X_to_arr)

            for i, layer in enumerate(self.layers):
                # TODO: replace sigmoid?
                last_out = self.add_bias(self.activation_function(np.dot(last_out, layer)))
                out.append(last_out)

            out_layers = out
            out_layers[-1] = np.delete(out_layers[-1], [len(out_layers[-1][0])-1], axis=1)
            #print(out_layers[-1].shape)
            last_error = y - out_layers[-1]

            # TODO: replace sigmoid?
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

                # TODO: replace sigmoid?
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
                self.layers[k] += change*self.learning_rate



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

    net = Network([2, 3, 1], iterations=1000)

    ins = [[1,1], [0,0], [1,0], [0,1]]
    outs = [[1], [0], [0], [0]]

    net.fit(ins, outs)

    print(net.predict(ins))