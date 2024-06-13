from treeModels.base_tree import *
from treeModels import *
import pandas as pd
from sklearn.model_selection import train_test_split

import tkinter as tk
import random

class PlotTree:
    def __init__(self, tree: BaseTree):
        self.tree = tree

    def draw_node(self, canvas, x, y, text):
        colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
        node_width = 200
        node_height = 70
        fill_color = random.choice(colors)  # Choose a random color
        canvas.create_rectangle(x - node_width/2, y - node_height/2, x + node_width/2, y + node_height/2, fill=fill_color, outline='black')
        canvas.create_text(x, y, text=text, fill='black', justify='center')

    def draw_arrow(self, canvas, x1, y1, x2, y2, text):
        canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, fill='black')
        canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2, text=text, fill='black')

    def plot(self, canvas, x, y, dx):
        text = f'decision: {self.tree.get_label() if not self.tree.is_leaf() else "Leaf"}\n'
        text += f'samples: {self.tree.n_samples()}\n'
        text += f'class proportion: {self.tree.get_class_proportion()}\n'
        text += f'class: {self.tree.get_class()}'

        if not self.tree.is_leaf():
            for i, key in enumerate(self.tree.forest.keys()):
                new_x = x + (i - len(self.tree.forest) / 2) * (dx + 50)  # Increase distance between nodes
                new_y = y + 100
                self.draw_arrow(canvas, x, y, new_x, new_y, f'value: {key}')
                PlotTree(self.tree.forest[key]).plot(canvas, new_x, new_y, dx / 2)

        self.draw_node(canvas, x, y, text)


    def show(self):
        root = tk.Tk()
        root.geometry("1200x700")

        canvas = tk.Canvas(root, bg='white')  # Set background color to white
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar_x = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)
        scrollbar_x.pack(side="bottom", fill="x")

        scrollbar_y = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
        scrollbar_y.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        self.plot(canvas, 400, 50, 300)

        root.mainloop()


def main():

    df = pd.read_csv('/Users/santiagodarnes/Documents/UNSAM/Algoritmos2/Sin título/TPFinalAlgo2/play_tennis.csv')
    df[df['outlook'] == 'Overcast']

    X = df.drop("play", axis=1)
    Y = df['play']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)

    PlotTree(model.tree).show()


if __name__ == '__main__':
    main()