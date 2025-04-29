import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
import heapq
import random
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


def select_fixed_goal(img_path, walkable_matrix, block_size):
    img = np.array(Image.open(img_path))
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    for r in range(len(walkable_matrix)):
        for c in range(len(walkable_matrix[0])):
            if walkable_matrix[r][c] == 1:
                x, y = block_to_pixel(r, c, block_size)
                ax.plot(x, y, '.', color='lime', alpha=0.2)
    
    coords = []

    def onclick(event):
        if event.xdata and event.ydata:
            x, y = int(event.xdata), int(event.ydata)
            block_r, block_c = pixel_to_block(x, y, block_size)
            
            # Verifica che il punto selezionato sia calpestabile
            if 0 <= block_r < len(walkable_matrix) and 0 <= block_c < len(walkable_matrix[0]):
                if walkable_matrix[block_r][block_c] == 1:
                    coords.append((block_r, block_c))
                    ax.plot(event.xdata, event.ydata, 'bo', markersize=10)
                    fig.canvas.draw()
                else:
                    print("Il punto selezionato non è calpestabile. Riprova.")
            else:
                print("Punto fuori dai limiti della mappa. Riprova.")

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    print("Clicca per selezionare i punti di GOAL (premi invio quando hai finito).")
    plt.show(block=False)
    
    input("Premi invio quando hai finito di selezionare i punti goal...")
    plt.close(fig)
    fig.canvas.mpl_disconnect(cid)

    return coords


def save_goals(map_id, goals):
    # Salva i punti goal in un file JSON
    goals_file = os.path.join(DATA_FOLDER, f"goals-{map_id}.json")
    with open(goals_file, "w", encoding="utf-8") as f:
        json.dump(goals, f)
    print(f"Goals salvati in {goals_file}")


def load_goals(map_id):
    # Carica i punti goal da un file JSON
    goals_file = os.path.join(DATA_FOLDER, f"goals-{map_id}.json")
    if os.path.exists(goals_file):
        with open(goals_file, "r", encoding="utf-8") as f:
            goals_data = json.load(f)
            # Converti le liste in tuple per garantire compatibilità
            goals = [tuple(goal) for goal in goals_data]
            return goals
    return None


def generate_random_start(walkable_matrix):
    # Trova tutti i punti calpestabili
    walkable_points = []
    for r in range(len(walkable_matrix)):
        for c in range(len(walkable_matrix[0])):
            if walkable_matrix[r][c] == 1:
                walkable_points.append((r, c))
    
    # Seleziona un punto casuale
    if walkable_points:
        return random.choice(walkable_points)
    return None


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
    
    # Converti start e goal in tuple per garantire hashability
    start = tuple(start)
    goal = tuple(goal)

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


def find_shortest_path_to_any_goal(walkable_matrix, start, goals, penalty_matrix=None):
    best_path = None
    best_path_length = float('inf')
    best_goal = None
    
    # Assicurati che start sia una tupla
    start = tuple(start)
    
    # Debug: stampa informazioni sui goal
    print(f"Cercando percorso da {start} verso {len(goals)} goals")
    for i, goal in enumerate(goals):
        print(f"Goal {i+1}: {goal}, tipo: {type(goal)}")
    
    for goal in goals:
        # Assicurati che il goal sia una tupla
        goal_tuple = tuple(goal)
        path = a_star(walkable_matrix, start, goal_tuple, penalty_matrix)
        if path:
            print(f"Trovato percorso di {len(path)} passi verso {goal_tuple}")
            if len(path) < best_path_length:
                best_path = path
                best_path_length = len(path)
                best_goal = goal_tuple
        else:
            print(f"Nessun percorso trovato verso {goal_tuple}")
    
    return best_path, best_goal


def heuristic(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def plot_path(map_id, path, walkable_matrix, start, goal, block_size=6):
    img_path = os.path.join(DATA_FOLDER, f"{map_id}.png")
    img = np.array(Image.open(img_path))
    plt.figure(figsize=(10, 10))
    plt.imshow(img, alpha=0.6)

    for r in range(len(walkable_matrix)):
        for c in range(len(walkable_matrix[0])):
            if walkable_matrix[r][c] == 1:
                x, y = block_to_pixel(r, c, block_size)
                plt.plot(x, y, '.', color='lime', alpha=0.2)

    # Disegna il percorso
    if path:
        x_coords = [block_to_pixel(row, col, block_size)[0] for row, col in path]
        y_coords = [block_to_pixel(row, col, block_size)[1] for row, col in path]
        plt.plot(x_coords, y_coords, color="red", linewidth=2)
    
    # Disegna punto di partenza
    start_x, start_y = block_to_pixel(start[0], start[1], block_size)
    plt.scatter(start_x, start_y, color="green", s=100, zorder=10, label="Start (Random)")
    
    # Disegna punto di arrivo
    goal_x, goal_y = block_to_pixel(goal[0], goal[1], block_size)
    plt.scatter(goal_x, goal_y, color="blue", s=100, zorder=10, label="Goal")
    
    plt.legend()
    plt.axis('off')
    plt.title("Percorso A* da Start Random a Goal Fisso")
    plt.show()


def plot_all_goals(map_id, walkable_matrix, goals, block_size=6):
    img_path = os.path.join(DATA_FOLDER, f"{map_id}.png")
    img = np.array(Image.open(img_path))
    plt.figure(figsize=(10, 10))
    plt.imshow(img, alpha=0.6)

    # Disegna tutti i punti goal
    for i, goal in enumerate(goals):
        goal_x, goal_y = block_to_pixel(goal[0], goal[1], block_size)
        plt.scatter(goal_x, goal_y, color="blue", s=100, zorder=10)
        plt.text(goal_x + 10, goal_y + 10, f"Goal {i+1}", fontsize=12, color="white", 
                 bbox=dict(facecolor='black', alpha=0.7))
    
    plt.axis('off')
    plt.title(f"Punti Goal per la Mappa {map_id}")
    plt.show()


def verify_walkable_points(walkable_matrix, points):
    """Verifica che i punti siano validi nella matrice calpestabile"""
    for r, c in points:
        if not (0 <= r < len(walkable_matrix) and 0 <= c < len(walkable_matrix[0])):
            print(f"Punto fuori dai limiti: {(r, c)}")
            return False
        if walkable_matrix[r][c] != 1:
            print(f"Punto non calpestabile: {(r, c)}")
            return False
    return True


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
    
    # Carica i goal esistenti o crea nuovi goal
    goals = load_goals(MAP_ID)
    if not goals:
        print("Nessun goal trovato per questa mappa. Seleziona i punti goal.")
        img_path = os.path.join(DATA_FOLDER, f"{MAP_ID}.png")
        goals = select_fixed_goal(img_path, walkable_matrix, block_size)
        if goals:
            save_goals(MAP_ID, goals)
            plot_all_goals(MAP_ID, walkable_matrix, goals, block_size)
        else:
            print("Nessun goal selezionato.")
            exit()
    else:
        print(f"Caricati {len(goals)} punti goal.")
        
        # Verifica che i goal caricati siano validi
        if verify_walkable_points(walkable_matrix, goals):
            print("Tutti i goal sono validi.")
            plot_all_goals(MAP_ID, walkable_matrix, goals, block_size)
        else:
            print("ATTENZIONE: Alcuni goal non sono validi!")
            print("Vuoi selezionare nuovi goal? (s/n)")
            if input().lower() == 's':
                img_path = os.path.join(DATA_FOLDER, f"{MAP_ID}.png")
                goals = select_fixed_goal(img_path, walkable_matrix, block_size)
                if goals:
                    save_goals(MAP_ID, goals)
                    plot_all_goals(MAP_ID, walkable_matrix, goals, block_size)
                else:
                    print("Nessun goal selezionato.")
                    exit()
    
    # Genera un punto di partenza casuale
    start = generate_random_start(walkable_matrix)
    if not start:
        print("Impossibile generare un punto di partenza valido.")
        exit()
    
    print(f"Punto di partenza generato: {start}")
    
    # Trova il percorso più breve verso uno dei goal
    path, chosen_goal = find_shortest_path_to_any_goal(walkable_matrix, start, goals, penalty_matrix)
    
    if not path:
        print("Nessun percorso trovato verso alcun goal.")
        print("Dettagli di debug:")
        print(f"Start: {start}, Tipo: {type(start)}")
        print(f"Goals: {goals}")
        # Verifica che i punti siano calpestabili
        if walkable_matrix[start[0]][start[1]] != 1:
            print("ERRORE: Il punto di partenza non è calpestabile!")
    else:
        print(f"Selezionato percorso più breve di {len(path)} passi verso il goal {chosen_goal}.")
        plot_path(MAP_ID, path, walkable_matrix, start, chosen_goal, block_size)
