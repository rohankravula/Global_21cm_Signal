import math
import random
import os
import datetime

output_file = "random_sine_dataset.txt"

def generate_unique_file():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"output_{timestamp}.txt"

    return os.path.join(".", filename)

def generate_random_sine_data(num_points=200):
    output_file = generate_unique_file()
    with open(output_file, "w") as file:
        file.write("A,b,x,Y\n")  # CSV Header
        for _ in range(num_points):
            A = random.uniform(1, 100)
            b = random.uniform(1, 100)
            x = random.uniform(1, 100)
            Y = A * (math.sin(x) + b)
            file.write(f"{A:.5f},{b:.5f},{x:.5f},{Y:.5f}\n")

generate_random_sine_data()
