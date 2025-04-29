import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
import heapq
from PIL import Image
from scipy.ndimage import distance_transform_edt

BASE_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(BASE_FOLDER, "data")

sys.path.append(BASE_FOLDER)
from models import Map


def load_map(map_id):
    with open(os.path.join(DATA_FOLDER, "maps.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
        maps = {m["id"]: Map.from_dict(m) for m in data}
    map_obj = maps[map_id]
    with open(os.path.join(DATA_FOLDER, f"walkable-{map_id}.bin"), "rb") as bin_file:
        map_obj.walkable_image_bytes = bin_file.read()
    return map_obj


def create_walkable_matrix(map_obj, block_size=6):
    walkable_matrix = []
    for y in range(0, map_obj.height, block_size):
        row = []
        for x in range(0, map_obj.width, block_size):
            walkable = True
            for dy in range(block_size):
                for dx in range(block_size):
                    if not map_obj.pixel_is_walkable(x + dx, y + dy):
                        walkable = False
                        break
                if not walkable:
                    break
            row.append(1 if walkable else 0)
        walkable_matrix.append(row)
    return walkable_matrix


def compute_penalty_matrix(walkable_matrix, penalty_radius=2):
    binary = np.array(walkable_matrix)
    inverted = 1 - binary  # Ostacoli = 1, camminabili = 0
    dist_from_walls = distance_transform_edt(binary)
    penalty_matrix = np.where(dist_from_walls < penalty_radius, penalty_radius - dist_from_walls, 0)
    return penalty_matrix


def get_points_from_click(img_path):
    img = np.array(Image.open(img_path))
    fig, ax = plt.subplots()
    ax.imshow(img)
    coords = []

    def onclick(event):
        if event.xdata and event.ydata:
            coords.append((int(event.xdata), int(event.ydata)))
            ax.plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()
            if len(coords) == 2:
                plt.close()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    print("Clicca per selezionare START e GOAL (2 click).")
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    return coords if len(coords) == 2 else None


def pixel_to_block(x, y, block_size):
    return y // block_size, x // block_size


def block_to_pixel(row, col, block_size):
    return col * block_size + block_size // 2, row * block_size + block_size // 2


def a_star(walkable_matrix, start, goal, penalty_matrix=None):
    height, width = len(walkable_matrix), len(walkable_matrix[0])
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    def neighbors(pos):
        r, c = pos
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and walkable_matrix[nr][nc] == 1:
                yield (nr, nc)

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in neighbors(current):
            penalty = penalty_matrix[neighbor[0]][neighbor[1]] if penalty_matrix is not None else 0
            step_cost = np.hypot(neighbor[0] - current[0], neighbor[1] - current[1])
            tentative_g = g_score[current] + step_cost + penalty

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # Nessun percorso trovato


def heuristic(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def plot_path(map_id, path, walkable_matrix, block_size=6):
    img_path = os.path.join(DATA_FOLDER, f"{map_id}.png")
    img = np.array(Image.open(img_path))
    plt.figure(figsize=(10, 10))
    plt.imshow(img, alpha=0.6)

    for r in range(len(walkable_matrix)):
        for c in range(len(walkable_matrix[0])):
            if walkable_matrix[r][c] == 1:
                x, y = block_to_pixel(r, c, block_size)
                plt.plot(x, y, '.', color='lime', alpha=0.2)

    x_coords = [(col * block_size) + block_size // 2 for row, col in path]
    y_coords = [(row * block_size) + block_size // 2 for row, col in path]
    plt.plot(x_coords, y_coords, color="red", linewidth=2)
    plt.scatter(x_coords[0], y_coords[0], color="green", label="Start")
    plt.scatter(x_coords[-1], y_coords[-1], color="blue", label="Goal")
    plt.legend()
    plt.axis('off')
    plt.title("Percorso A*")
    plt.show()


if __name__ == "__main__":
    available_maps = [703368, 703326, 703323]
    print("Mappe disponibili:", available_maps)
    MAP_ID = int(input("Inserisci MAP_ID: "))

    if MAP_ID not in available_maps:
        print("MAP_ID non valido.")
        exit()

    map_obj = load_map(MAP_ID)
    block_size = 6
    walkable_matrix = create_walkable_matrix(map_obj, block_size)

    penalty_matrix = compute_penalty_matrix(walkable_matrix, penalty_radius=2)

    img_path = os.path.join(DATA_FOLDER, f"{MAP_ID}.png")
    clicked = get_points_from_click(img_path)
    if not clicked:
        print("Selezione punti fallita.")
        exit()

    start_pixel, goal_pixel = clicked
    start = pixel_to_block(*start_pixel, block_size)
    goal = pixel_to_block(*goal_pixel, block_size)

    if walkable_matrix[start[0]][start[1]] == 0 or walkable_matrix[goal[0]][goal[1]] == 0:
        print("Start o goal non calpestabili.")
        exit()

    path = a_star(walkable_matrix, start, goal, penalty_matrix)
    if not path:
        print("Nessun percorso trovato.")
    else:
        print(f"Trovato percorso di {len(path)} passi.")
        plot_path(MAP_ID, path, walkable_matrix, block_size)
