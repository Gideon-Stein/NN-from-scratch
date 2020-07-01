from mlxtend.data import loadlocal_mnist
import Network as NN

class program:
        def __init__(self):
                self.On = True
                self.X, self.y = loadlocal_mnist(
                        images_path='./data/train-images.idx3-ubyte',
                        labels_path='./data/train-labels.idx1-ubyte')
                self.X_test, self.y_test = loadlocal_mnist(
                        images_path='./data/t10k-images.idx3-ubyte',
                        labels_path='./data/t10k-labels.idx1-ubyte')
                self.sizeX = self.X.shape[1]
                self.sizey = 10
                self.network = False

        def run(self):
                print("Hello.Please insert command.")
                while self.On:
                        inp = input()
                        if inp == "create":
                                print("specify hidden layer size:")
                                hN =int(input())
                                self.network = NN.NN(self.sizeX, self.sizey, hN)
                                print("model created.")
                        elif inp == "load pre" and self.network:
                                self.network.load_weights()
                                print("weights loaded")
                        elif inp == "train"and self.network:
                                print("how many epochs?")
                                ep = int(input())
                                print("specify learning rate:")
                                lr = float(input())
                                self.network.train(self.X, self.y, ep, lr)
                        elif inp == "evaluate" and self.network:
                                self.network.evaluate(self.X_test,self.y_test)
                        elif inp == "save" and self.network:
                                self.network.save_weights()
                                print("weights saved")
                        elif inp == "load own" and self.network:
                                self.network.load_weights(False)
                                print("weights loaded")
                        elif inp == "draw" and self.network:
                                self.network.draw_number()


                        elif inp == "quit" and self.network:
                                self.On = False
                                print("good bye")
                        else:
                                print("command unknown or model not created")



