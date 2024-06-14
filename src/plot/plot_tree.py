from treeModels.base_tree import *
from treeModels import *
from treeModels.decision_algorithms import *
import pandas as pd
from sklearn.model_selection import train_test_split

import tkinter as tk
import random

class PlotTree:
    def __init__(self, tree):
        self.tree = tree

    def draw_node(self, canvas, x, y, text):
        colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
        node_width = 200
        node_height = 70
        fill_color = random.choice(colors)
        canvas.create_rectangle(x - node_width / 2, y - node_height / 2, x + node_width / 2, y + node_height / 2, fill=fill_color, outline='black')
        canvas.create_text(x, y, text=text, fill='black', justify='center')

    def draw_arrow(self, canvas, x1, y1, x2, y2, text):
        arrow_y_offset = 35 
        canvas.create_line(x1, y1 + arrow_y_offset, x2, y2 - arrow_y_offset, arrow=tk.LAST, fill='black')
        canvas.create_text((x1 + x2) / 2, ((y1 + arrow_y_offset-20) + (y2 - arrow_y_offset)) / 2, text=text, fill='black', font=('Arial', 10, 'bold'))

    def plot(self, canvas, x, y, dx):
        text = f'' if self.tree.is_leaf() else f'{self.tree.get_label()}\n'
        text += f'samples: {self.tree.n_samples()}\n'
        text += f'impurity: {abs(round(self.tree.get_impurity(), 3))}\n'
        text += f'class: {self.tree.get_class()}'

        self.draw_node(canvas, x, y, text)

        if not self.tree.is_leaf():
            num_children = len(self.tree.forest)
            child_dx = dx / max(num_children, 1)

            for i, key in enumerate(self.tree.forest.keys()):
                new_x = x + (i - (num_children - 1) / 2) * (child_dx + 250)
                new_y = y + 150
                self.draw_arrow(canvas, x, y, new_x, new_y, f'{key}')
                PlotTree(self.tree.forest[key]).plot(canvas, new_x, new_y, child_dx / 2)

    def show(self):
        root = tk.Tk()
        root.geometry("1600x1200")

        canvas = tk.Canvas(root, bg='white') 
        canvas.pack(side="top", fill="both", expand=True)

        scrollbar_x = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)
        scrollbar_x.pack(side="bottom", fill="x")

        scrollbar_y = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
        scrollbar_y.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        self.plot(canvas, 800, 50, 1200) 

        root.mainloop()


def main():

    df = pd.read_csv('/Users/santiagodarnes/Documents/UNSAM/Algoritmos2/Sin título/TPFinalAlgo2/CarEval.csv')

    X = df.drop("class values", axis=1)
    Y = df['class values']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    model = DecisionTreeClassifier(max_depth=3, algorithm=DecisionAlgorithm.C45)
    model.fit(X_train, Y_train)

    PlotTree(model.tree).show()


if __name__ == '__main__':
    main()