import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
import random
from tqdm import tqdm
from PIL import Image

# Imposta la cartella di lavoro

BASE_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(BASE_FOLDER, "data")


# Importa la classe Map
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
from models import Map
sys.path.remove(parent_dir)

# Funzione per caricare la mappa
def load_map(map_id):
    with open(os.path.join(DATA_FOLDER, "maps.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
        maps = {m["id"]: Map.from_dict(m) for m in data}

    map_obj = maps[map_id]

    with open(os.path.join(DATA_FOLDER, f"walkable-{map_id}.bin"), "rb") as bin_file:
        walkable_image_bytes = bin_file.read()
        map_obj.walkable_image_bytes = walkable_image_bytes

    return map_obj

# Funzione per creare la matrice calpestabile
def create_walkable_matrix(map_obj):
    walkable_matrix = []
    for y in range(map_obj.height):
        row = []
        for x in range(map_obj.width):
            if map_obj.pixel_is_walkable(x, y):
                row.append(1)
            else:
                row.append(0)
        walkable_matrix.append(row)
    return walkable_matrix

# Funzione per trovare start e goal camminabili
def find_start_goal(walkable_matrix):
    height = len(walkable_matrix)
    width = len(walkable_matrix[0])

    walkable_points = [(i, j) for i in range(height) for j in range(width) if walkable_matrix[i][j] == 1]
    start = random.choice(walkable_points)
    goal = random.choice(walkable_points)
    while start == goal:
        goal = random.choice(walkable_points)
    return start, goal

# Funzione Q-learning
def q_learning(walkable_matrix, start, goal, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    height, width = len(walkable_matrix), len(walkable_matrix[0])
    Q = {}

    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    def is_valid(state):
        x, y = state
        return 0 <= x < height and 0 <= y < width and walkable_matrix[x][y] == 1

    for episode in tqdm(range(episodes), desc="Training Q-Learning"):
        state = start

        while state != goal:
            if random.uniform(0, 1) < epsilon or state not in Q:
                action = random.choice(actions)
            else:
                action = max(Q[state], key=Q[state].get)

            next_state = (state[0] + action[0], state[1] + action[1])

            if not is_valid(next_state):
                next_state = state

            reward = -1
            if next_state == goal:
                reward = 100

            if state not in Q:
                Q[state] = {}
            if action not in Q[state]:
                Q[state][action] = 0

            max_future_q = max(Q.get(next_state, {}).values(), default=0)
            Q[state][action] += alpha * (reward + gamma * max_future_q - Q[state][action])

            state = next_state

    # Ricostruzione percorso
    path = [start]
    state = start
    while state != goal:
        if state not in Q:
            break
        action = max(Q[state], key=Q[state].get)
        next_state = (state[0] + action[0], state[1] + action[1])
        if not is_valid(next_state) or next_state == state:
            break
        path.append(next_state)
        state = next_state

    return path

# Funzione per visualizzare il percorso
def plot_path(map_id, path):
    img_path = os.path.join(DATA_FOLDER, f"{map_id}.png")
    img = np.array(Image.open(img_path))

    plt.imshow(img)
    y_coords, x_coords = zip(*path)
    plt.plot(x_coords, y_coords, color="red", linewidth=2)
    plt.scatter(x_coords[0], y_coords[0], color="green", label="Start")
    plt.scatter(x_coords[-1], y_coords[-1], color="blue", label="Goal")
    plt.legend()
    plt.axis('off')
    plt.show()

# --- MAIN PROGRAM ---

if __name__ == "__main__":
    # Chiedi quale mappa usare
    available_maps = [703368, 703326, 703323]
    print(f"Mappe disponibili: {available_maps}")
    MAP_ID = int(input("Inserisci MAP_ID da usare: "))

    if MAP_ID not in available_maps:
        print("Mappa non valida.")
        exit()

    # Carica mappa
    map_obj = load_map(MAP_ID)
    walkable_matrix = create_walkable_matrix(map_obj)

    # Trova start e goal
    start, goal = find_start_goal(walkable_matrix)
    print(f"Start: {start}, Goal: {goal}")

    # Q-Learning
    path = q_learning(walkable_matrix, start, goal)

    # Mostra il percorso
    plot_path(MAP_ID, path)
