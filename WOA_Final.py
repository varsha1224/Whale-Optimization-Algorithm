# WHALE OPTIMIZATION ALGORITHM

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import random
import math
import sys
import copy 
import matplotlib.pyplot as plt
import numpy as np

# FITNESS FUNCTIONS

# rastrigin function
def fitness_rastrigin(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitness_value


# sphere function
def fitness_sphere(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += (xi * xi);
    return fitness_value;


# whale class
class whale:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)
        self.position = [0.0 for i in range(dim)]

        for i in range(dim):
            self.position[i] = ((maxx - minx) * self.rnd.random() + minx)

        self.fitness = fitness(self.position) # curr fitness


# whale optimization algorithm(WOA)
def woa(fitness, max_iter, n, dim, minx, maxx):
    rnd = random.Random(0)

    # creating n random whales
    whalePopulation = [whale(fitness, dim, minx, maxx, i) for i in range(n)]

    # computing the value of best_position and best_fitness in the whale Population
    Xbest = [0.0 for i in range(dim)]
    Fbest = sys.float_info.max

    for i in range(n): # checking each whale
        if whalePopulation[i].fitness < Fbest:
            Fbest = whalePopulation[i].fitness
            Xbest = copy.copy(whalePopulation[i].position)

    # main loop of woa
    Iter = 0
    while Iter < max_iter:

        # after every 10 iterations 
        # printing iteration number and best fitness value so far
        if Iter % 10 == 0 and Iter > 1:
            print("Iter = " + str(Iter) + " best fitness = %.3f" % Fbest)

        # linearly decreased from 2 to 0
        a = 2 * (1 - Iter / max_iter)
        a2=-1+Iter*((-1)/max_iter)

        for i in range(n):
            A = 2 * a * rnd.random() - a
            C = 2 * rnd.random()
            b = 1
            l = (a2-1)*rnd.random()+1;
            p = rnd.random()

            D = [0.0 for i in range(dim)]
            D1 = [0.0 for i in range(dim)]
            Xnew = [0.0 for i in range(dim)]
            Xrand = [0.0 for i in range(dim)]
            if p < 0.5:
                if abs(A) > 1:
                    for j in range(dim):
                        D[j] = abs(C * Xbest[j] - whalePopulation[i].position[j])
                        Xnew[j] = Xbest[j] - A * D[j]
                else:
                    p = random.randint(0, n - 1)
                    while (p == i):
                        p = random.randint(0, n - 1)

                    Xrand = whalePopulation[p].position

                    for j in range(dim):
                        D[j] = abs(C * Xrand[j] - whalePopulation[i].position[j])
                        Xnew[j] = Xrand[j] - A * D[j]
            else:
                for j in range(dim):
                    D1[j] = abs(Xbest[j] - whalePopulation[i].position[j])
                    Xnew[j] = D1[j] * math.exp(b * l) * math.cos(2 * math.pi * l) + Xbest[j]

            for j in range(dim):
                whalePopulation[i].position[j] = Xnew[j]

        for i in range(n):
            # if Xnew < minx OR Xnew > maxx then clip it
            for j in range(dim):
                whalePopulation[i].position[j] = max(whalePopulation[i].position[j], minx)
                whalePopulation[i].position[j] = min(whalePopulation[i].position[j], maxx)

            whalePopulation[i].fitness = fitness(whalePopulation[i].position)

            if (whalePopulation[i].fitness < Fbest):
                Xbest = copy.copy(whalePopulation[i].position)
                Fbest = whalePopulation[i].fitness


        Iter += 1
    # end-while

    # returning the best solution
    return Xbest

# Creating a 3D plot for the Rastrigin function
def plot_rastrigin(dim):
    # Creating a meshgrid of x and y values
    x = np.linspace(-5.12, 5.12, 400)
    y = np.linspace(-5.12, 5.12, 400)
    X, Y = np.meshgrid(x, y)

    # Computing the Rastrigin values for each point in the grid
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            position = [X[i, j], Y[i, j]]
            Z[i, j] = fitness_rastrigin(position)

    # Creating a 3D surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Rastrigin Value')
    ax.set_title('Rastrigin Function')
    plt.show()

# Creating a 3D plot for the Sphere function
def plot_sphere(dim):
    # Creating a meshgrid of x and y values
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)

    # Computing the Sphere values for each point in the grid
    Z = X**2 + Y**2

    # Creating a 3D surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Sphere Value')
    ax.set_title('Sphere Function')
    plt.show()

def run_rastrigin_optimization():
    dim_value = int(dimension_var.get())
    fitness = fitness_rastrigin

    result_text = f"Goal is to minimize Rastrigin's function in {dim_value} variables\n"
    result_text += f"Function has known min = 0.0 at ({', '.join(['0'] * dim_value)})\n"

    num_whales = 50
    max_iter = 100
    result_text += f"Setting num_whales = {num_whales}\n"
    result_text += f"Setting max_iter = {max_iter}\n"
    result_text += "\nStarting WOA algorithm\n"

    best_position = woa(fitness, max_iter, num_whales, dim_value, -10.0, 10.0)
    result_text += "\nWOA completed\n"
    result_text += "\nBest solution found:\n"
    result_text += " ".join([f"%.6f" % best_position[k] for k in range(dim_value)]) + "\n"
    err = fitness(best_position)
    result_text += f"Fitness of best solution = {err:.6f}\n"
    result_text += "\nEnd WOA for Rastrigin\n"

    result_display.config(state=tk.NORMAL)
    result_display.delete("1.0", tk.END)
    result_display.insert(tk.END, result_text)
    result_display.config(state=tk.DISABLED)

    # Plot Rastrigin function
    plot_rastrigin(dim_value)

def run_sphere_optimization():
    dim_value = int(dimension_var.get())
    fitness = fitness_sphere

    result_text = f"Goal is to minimize Sphere function in {dim_value} variables\n"
    result_text += f"Function has known min = 0.0 at ({', '.join(['0'] * dim_value)})\n"

    num_whales = 50
    max_iter = 100
    result_text += f"Setting num_whales = {num_whales}\n"
    result_text += f"Setting max_iter = {max_iter}\n"
    result_text += "\nStarting WOA algorithm\n"

    best_position = woa(fitness, max_iter, num_whales, dim_value, -10.0, 10.0)
    result_text += "\nWOA completed\n"
    result_text += "\nBest solution found:\n"
    result_text += " ".join([f"%.6f" % best_position[k] for k in range(dim_value)]) + "\n"
    err = fitness(best_position)
    result_text += f"Fitness of best solution = {err:.6f}\n"
    result_text += "\nEnd WOA for Sphere\n"

    result_display.config(state=tk.NORMAL)
    result_display.delete("1.0", tk.END)
    result_display.insert(tk.END, result_text)
    result_display.config(state=tk.DISABLED)

    # Plot Sphere function
    plot_sphere(dim_value)

# Creating the main window
root = tk.Tk()
root.title("Whale Optimization Algorithm")

# Loading background image
background_image = Image.open("whale_bg.jpg")
background_photo = ImageTk.PhotoImage(background_image)

# Creating a label for the background image
background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Creating  a frame for the user interface
frame = ttk.LabelFrame(root, text="Select Optimization Function", padding=10)
frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

# Label and Entry for dimension input
dimension_label = ttk.Label(frame, text="Enter Dimension:")
dimension_label.grid(row=0, column=0, padx=5, pady=5)

dimension_var = tk.StringVar()
dimension_entry = ttk.Entry(frame, textvariable=dimension_var)
dimension_entry.grid(row=0, column=1, padx=5, pady=5)

# Button to run Rastrigin optimization
rastrigin_button = ttk.Button(frame, text="Optimize Rastrigin", command=run_rastrigin_optimization)
rastrigin_button.grid(row=1, column=0, columnspan=2, padx=5, pady=10, sticky="nsew")

# Button to run Sphere optimization
sphere_button = ttk.Button(frame, text="Optimize Sphere", command=run_sphere_optimization)
sphere_button.grid(row=2, column=0, columnspan=2, padx=5, pady=10, sticky="nsew")

# Text widget to display results
result_display = tk.Text(root, wrap=tk.WORD, state=tk.DISABLED, width=60, height=20)
result_display.grid(row=1, column=0, padx=20, pady=10)

# Close button
close_button = ttk.Button(root, text="Close", command=root.destroy)
close_button.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")

# Set column and row weights for resizing
root.columnconfigure(0, weight=1)
root.rowconfigure(1, weight=1)

# Start the Tkinter main loop
root.mainloop()

