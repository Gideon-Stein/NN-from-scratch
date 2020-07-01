import tkinter as tk
import numpy as np

class drawer(tk.Tk):
    def __init__(self, model):
        tk.Tk.__init__(self)
        self.model = model
        self.previous_x = self.previous_y = 0
        self.x = self.y = 0
        self.points_recorded = []
        self.canvas = tk.Canvas(self, width=280, height=280, bg = "black", cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.button_print = tk.Button(self, text = "Submit Number", command = self.submit_number)
        self.button_print.pack(side="top", fill="both", expand=True)
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        self.button_clear.pack(side="top", fill="both", expand=True)
        self.button_clear = tk.Button(self, text="redraw", command=self.draw_example)
        self.button_clear.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<Motion>", self.tell_me_where_you_are)
        self.canvas.bind("<B1-Motion>", self.draw_from_where_you_are)

    def clear_all(self):
        self.canvas.delete("all")
        self.points_recorded = []

    def submit_number(self):     #####Make this better.
        out = []
        for x in self.points_recorded:
            out.append(x//10)
        formated = [0]*784
        for y in range (0,len(out),2):
            formated[(out[y+1]-1)*28+out[y]] = 255
            formated[(out[y + 1] - 2) * 28 + out[y]] = 255
            formated[(out[y + 1] - 0) * 28 + out[y]] = 255
            formated[(out[y  +1] - 1) * 28 + out[y]+1] = 255
            formated[(out[y  +1] - 1) * 28 + out[y] -1] = 255


        self.draw_predict(formated)

    def draw_example(self):
        out = []
        for x in self.points_recorded:
            out.append(x//10)
        for x in range (0,len(out)-3,2):
            self.canvas.create_line(out[x], out[x+1], out[x+2], out[x+3], fill="yellow")

    def tell_me_where_you_are(self, event):
        self.previous_x = event.x
        self.previous_y = event.y


    def draw_from_where_you_are(self, event):
        self.x = event.x
        self.y = event.y
        self.canvas.create_line(self.previous_x, self.previous_y,self.x, self.y,fill="yellow",width=3)
        self.points_recorded.append(self.x)
        self.points_recorded.append(self.y)
        self.previous_x = self.x
        self.previous_y = self.y

    def draw_predict(self, directory):
        out1 = self.model.forward(directory)
        out15 = np.squeeze(np.array(out1),axis=1)
        out2 = self.model.softmax(out15)
        print(out2)

        print("The model assigns the following probabilities to the known digits:")
        for guess in range(len(out2)):
            print("For the digit " + str(guess) + " : " + str(out2[guess] * 100) + " %")


