import numpy as np
import GUI as GUI

class NN:
    def __init__(self, dimensionX, dimensiony, dimensionH):
        self.input = dimensionX
        self.y = dimensiony
        self.hidden1 = dimensionH
        self.weights1 = np.random.rand(self.hidden1, self.input)
        self.bias1 = np.random.rand(self.hidden1, 1)
        self.weights2 = np.random.rand(self.y, self.hidden1)
        self.bias2 = np.random.rand(self.y, 1)

    def forward(self, input):
        inp = np.expand_dims(np.array(input), axis=1)
        out = np.dot(self.weights1, inp) + self.bias1
        out = self.sigmoid(out)
        out = np.dot(self.weights2, out) + self.bias2
        out = self.sigmoid(out)
        return out

    def predict(self, input):
        outputs = []
        for x in input:
            prob = self.forward(x)
            prob.tolist()
            outputs.append(prob)
        return (outputs)

    def sigmoid(self, z):
        z = np.clip(z, -5, 5)  # Clip to prevent overflow --> problem not here
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, output):
        return output * (1.0 - output)

    def MSE(self, predicted, real):
        real = self.oneHotEncoding(real, self.y)
        mse = 0
        for value in range(len(predicted)):
            mse += np.mean((real[value] - predicted[value]) ** 2)
        return mse

    def oneHotEncoding(self, label, outputD):
        encode = []
        for z in range(0, self.y):
            if z == label:
                encode.append(1)
            else:
                encode.append(0)
        return encode

    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def train(self, X, y, epochN,lr):
        for iterations in range(epochN):
            print("Epoch: " + str(iterations + 1))
            loss = 0
            for data in range(len(X)):
                inp = np.expand_dims(np.array(X[data]), axis=1)
                Z1 = np.dot(self.weights1, inp) + self.bias1
                A1 = self.sigmoid(Z1)
                Z2 = np.dot(self.weights2, A1) + self.bias2
                A2 = self.sigmoid(Z2)
                yLabel = np.expand_dims(self.oneHotEncoding(y[data], 10), axis=1)

                dZ2 = -(yLabel - A2) * self.sigmoid_derivative(A2)
                dW2 = 1/10* np.dot(dZ2, A1.T)
                dB2 = 1/10*np.sum(dZ2, axis=1, keepdims=True)




                dZ1 = np.dot(self.weights2.T,dZ2)*self.sigmoid_derivative(A1)
                dW1 = 1/10*np.dot(dZ1,inp.T)
                dB1 = 1/10*(np.sum(dZ1, axis=1, keepdims=True))

                self.weights2 = self.weights2 - lr* dW2
                self.bias2 = self.bias2 - lr* dB2
                self.weights1 = self.weights1 - lr*dW1
                self.bias1 = self.bias1 - lr*dB1

                loss = loss + self.MSE(A2, y[data])
            print(loss)

    def save_weights(self):
        np.savetxt("weights1",self.weights1)
        np.savetxt("weights2", self.weights2)
        np.savetxt("bias1", self.bias1)
        np.savetxt("bias2", self.bias2)

    def load_weights(self,pre=True):
        if pre == False:
            self.weights1 = np.loadtxt("weights1")
            self.weights2 = np.loadtxt("weights2")
            self.bias1 = np.expand_dims(np.loadtxt("bias1"),axis=1)
            self.bias2 = np.expand_dims(np.loadtxt("bias2"), axis=1)
        else:
            self.weights1 = np.loadtxt("pretrained/weights1")
            self.weights2 = np.loadtxt("pretrained/weights2")
            self.bias1 = np.expand_dims(np.loadtxt("pretrained/bias1"), axis=1)
            self.bias2 = np.expand_dims(np.loadtxt("pretrained/bias2"), axis=1)

    def evaluate(self, X, y):
        out = self.predict(X)
        predictions = np.argmax(out, axis=1)
        score = 0
        for x in range(len(out)):
            if y[x] == predictions[x]:
                score += 1
        score = score / len(predictions) * 100
        print("The model predicted " + str(score) + "% of the pictures correct")

    def draw_number(self):
            root = GUI.drawer(self)
            root.mainloop()