import tkinter as tk
import numpy as np
import pickle
from tkinter import messagebox
import os


class PerceptronApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Персептрон: Распознавание 5 и сердечка")

        self.canvas_size = 200  # Размер холста
        self.grid_size = 20  # Размер сетки
        self.cell_size = self.canvas_size // self.grid_size

        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=2)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)

        self.clear_button = tk.Button(root, text="Очистить", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=0)

        self.done_button = tk.Button(root, text="Готово", command=self.process_image)
        self.done_button.grid(row=1, column=1)

        self.result_label = tk.Label(root, text="", font=("Helvetica", 16))
        self.result_label.grid(row=2, column=0, columnspan=2, pady=10)

        self.grid_data = np.zeros((self.grid_size, self.grid_size))
        self.learning_rate = 0.7
        self.weights_file = "perceptron_weights.pkl"


        self.weights = self.load_weights()
        # print(self.weights)

    def draw(self, event):
        x, y = event.x // self.cell_size, event.y // self.cell_size
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.grid_data[y, x] = 1
            self.canvas.create_rectangle(x * self.cell_size, y * self.cell_size,
                                         (x + 1) * self.cell_size, (y + 1) * self.cell_size, fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.grid_data = np.zeros((self.grid_size, self.grid_size))
        self.result_label.config(text="")

    def process_image(self):
        sum_output = np.sum(self.grid_data * self.weights)   # Взвешенная сумма
        result = 1 if sum_output > 0 else 0
        self.show_result(result)
        self.root.after(500, self.check_correctness, result)

    def show_result(self, result):
        if result == 1:
            message = "Это цифра 5!"
        else:
            message = "Это сердечко!"
        self.result_label.config(text=message)

    def check_correctness(self, result):
        answer = messagebox.askyesno("Проверка", "Результат правильный?")
        if not answer:
            self.train_perceptron(result)

    def train_perceptron(self, result):
        correction = 1 if result == 0 else -1
        self.weights += self.learning_rate * correction * self.grid_data
        self.save_weights()

    def save_weights(self):
        with open(self.weights_file, "wb") as f:
            pickle.dump(self.weights, f)
        print("Веса сохранены.")

    def load_weights(self):
        if os.path.exists(self.weights_file):
            with open(self.weights_file, "rb") as f:
                weights = pickle.load(f)
            print("Веса загружены из файла.")
        else:
            weights = np.random.uniform(-0.3, 0.3, (self.grid_size, self.grid_size))
            print("Файл весов не найден. Инициализация случайными значениями.")
        return weights


if __name__ == "__main__":
    root = tk.Tk()
    app = PerceptronApp(root)
    root.mainloop()
