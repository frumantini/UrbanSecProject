import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
import heapq
import random
from PIL import Image
from scipy.ndimage import distance_transform_edt
import pickle
import time
import math

BASE_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(BASE_FOLDER, "data")

sys.path.append(BASE_FOLDER)
from models import Map

# Parametri ottimizzati
LEARNING_RATE = 0.2
EXPLORATION_RATE = 0.95
MIN_EXPLORATION_RATE = 0.01
DISCOUNT_FACTOR = 0.9
EPISODES = 10000
MAX_STEPS = 1000
CYCLE_PENALTY = -10
MAX_STATE_VISITS = 3
TEMPERATURE = 1.5


def load_map(map_id):
    """Carica la mappa dal file JSON"""
    with open(os.path.join(DATA_FOLDER, "maps.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
        maps = {m["id"]: Map.from_dict(m) for m in data}
    map_obj = maps[map_id]
    with open(os.path.join(DATA_FOLDER, f"walkable-{map_id}.bin"), "rb") as bin_file:
        map_obj.walkable_image_bytes = bin_file.read()
    return map_obj


def create_walkable_matrix(map_obj, block_size=5):
    """Crea una matrice di blocchi calpestabili"""
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


def compute_penalty_matrix(walkable_matrix, block_size=5, penalty_radius=None):
    """Calcola una matrice di penalità che PENALIZZA fortemente le aree vicino ai muri"""
    binary = np.array(walkable_matrix)
    
    if penalty_radius is None:
        penalty_radius = block_size * 0.8  # Aumentato il raggio di penalità
    
    dist_from_walls = distance_transform_edt(binary)
    dist_from_walls = np.maximum(dist_from_walls, 0)
    
    # PENALITÀ FORTE vicino ai muri (distanza piccola = penalità alta)
    penalty_matrix = np.where(
        (dist_from_walls < penalty_radius) & (dist_from_walls >= 0), 
        np.power(np.maximum(penalty_radius - dist_from_walls, 0), 1.5) * 3,  # Penalità molto forte
        0
    )
    
    # Identifica corridoi (BONUS negativo) e stanze grandi (penalità positiva)
    corridor_bonus = np.zeros_like(binary, dtype=float)
    room_penalty = np.zeros_like(binary, dtype=float)
    
    for r in range(1, binary.shape[0]-1):
        for c in range(1, binary.shape[1]-1):
            if binary[r, c] == 1:  # Punto calpestabile
                # Analizza l'area locale 5x5
                local_area = binary[max(0, r-2):min(binary.shape[0], r+3), 
                                   max(0, c-2):min(binary.shape[1], c+3)]
                walkable_in_area = np.sum(local_area)
                total_area = local_area.size
                
                # Conta i vicini calpestabili in 8 direzioni
                neighbors = 0
                wall_neighbors = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < binary.shape[0] and 0 <= nc < binary.shape[1]:
                            if binary[nr, nc] == 1:
                                neighbors += 1
                            else:
                                wall_neighbors += 1
                
                # Classifica il tipo di spazio
                if neighbors <= 3 and wall_neighbors >= 4:
                    # Corridoio stretto - BONUS FORTE (negativo)
                    corridor_bonus[r, c] = -12
                elif neighbors <= 5 and wall_neighbors >= 2:
                    # Semi-corridoio - bonus moderato (negativo)
                    corridor_bonus[r, c] = -6
                elif walkable_in_area >= 15 and walkable_in_area / total_area > 0.7:
                    # Stanza grande - penalità moderata (positivo)
                    room_penalty[r, c] = 2
    
    # Combina le penalità: vicino ai muri = ALTO (rosso), corridoi = BASSO (verde)
    final_penalty = penalty_matrix + room_penalty - corridor_bonus
    
    # Assicurati che i valori siano positivi per la visualizzazione
    # I corridoi avranno valori bassi (verde), i muri alti (rosso)
    final_penalty = np.maximum(final_penalty, 0)
    
    # Normalizzazione per la visualizzazione
    if final_penalty.max() > 0:
        final_penalty = 10 * final_penalty / final_penalty.max()
    
    return final_penalty

def select_point_on_map(img_path, walkable_matrix, block_size, title="Seleziona punto"):
    """Permette di selezionare un punto sulla mappa cliccando"""
    img = np.array(Image.open(img_path))
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img)
    
    # Mostra i punti calpestabili
    for r in range(len(walkable_matrix)):
        for c in range(len(walkable_matrix[0])):
            if walkable_matrix[r][c] == 1:
                x, y = block_to_pixel(r, c, block_size)
                ax.plot(x, y, '.', color='lime', alpha=0.2, markersize=1)
    
    selected_point = []

    def onclick(event):
        if event.xdata and event.ydata:
            x, y = int(event.xdata), int(event.ydata)
            block_r, block_c = pixel_to_block(x, y, block_size)
            
            if (0 <= block_r < len(walkable_matrix) and 
                0 <= block_c < len(walkable_matrix[0]) and
                walkable_matrix[block_r][block_c] == 1):
                
                selected_point.clear()
                selected_point.append((block_r, block_c))
                ax.clear()
                ax.imshow(img)
                
                # Remostra i punti calpestabili
                for r in range(len(walkable_matrix)):
                    for c in range(len(walkable_matrix[0])):
                        if walkable_matrix[r][c] == 1:
                            px, py = block_to_pixel(r, c, block_size)
                            ax.plot(px, py, '.', color='lime', alpha=0.2, markersize=1)
                
                # Mostra il punto selezionato
                ax.plot(event.xdata, event.ydata, 'ro', markersize=15)
                ax.set_title(f"{title} - Selezionato: {(block_r, block_c)}")
                fig.canvas.draw()
                print(f"Punto selezionato: {(block_r, block_c)}")
            else:
                print("Punto non valido. Seleziona un'area calpestabile (verde).")

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    ax.set_title(f"{title} - Clicca su un punto calpestabile (verde)")
    plt.show(block=False)
    
    input("Premi invio quando hai selezionato il punto...")
    plt.close(fig)
    fig.canvas.mpl_disconnect(cid)

    return selected_point[0] if selected_point else None


def select_goals(img_path, walkable_matrix, block_size):
    """Permette di selezionare multiple goals sulla mappa"""
    img = np.array(Image.open(img_path))
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img)
    
    # Mostra i punti calpestabili
    for r in range(len(walkable_matrix)):
        for c in range(len(walkable_matrix[0])):
            if walkable_matrix[r][c] == 1:
                x, y = block_to_pixel(r, c, block_size)
                ax.plot(x, y, '.', color='lime', alpha=0.2, markersize=1)
    
    goals = []

    def onclick(event):
        if event.xdata and event.ydata:
            x, y = int(event.xdata), int(event.ydata)
            block_r, block_c = pixel_to_block(x, y, block_size)
            
            if (0 <= block_r < len(walkable_matrix) and 
                0 <= block_c < len(walkable_matrix[0]) and
                walkable_matrix[block_r][block_c] == 1):
                
                goals.append((block_r, block_c))
                ax.plot(event.xdata, event.ydata, 'bo', markersize=10)
                ax.text(event.xdata + 10, event.ydata + 10, 
                       f"Goal {len(goals)}", fontsize=10, color='white',
                       bbox=dict(facecolor='blue', alpha=0.7))
                fig.canvas.draw()
                print(f"Goal {len(goals)} aggiunto: {(block_r, block_c)}")
            else:
                print("Punto non valido. Seleziona un'area calpestabile (verde).")

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    ax.set_title("Seleziona i punti GOAL - Clicca sui punti desiderati")
    plt.show(block=False)
    
    input("Premi invio quando hai finito di selezionare i goals...")
    plt.close(fig)
    fig.canvas.mpl_disconnect(cid)

    return goals


def save_goals(map_id, goals):
    """Salva i goals su file"""
    goals_file = os.path.join(DATA_FOLDER, f"goals-{map_id}.json")
    with open(goals_file, "w", encoding="utf-8") as f:
        json.dump(goals, f)
    print(f"Goals salvati in {goals_file}")


def load_goals(map_id):
    """Carica i goals dal file"""
    goals_file = os.path.join(DATA_FOLDER, f"goals-{map_id}.json")
    if os.path.exists(goals_file):
        with open(goals_file, "r", encoding="utf-8") as f:
            goals_data = json.load(f)
            return [tuple(goal) for goal in goals_data]
    return None


def pixel_to_block(x, y, block_size):
    """Converte coordinate pixel in coordinate blocco"""
    return y // block_size, x // block_size


def block_to_pixel(row, col, block_size):
    """Converte coordinate blocco in coordinate pixel"""
    return col * block_size + block_size // 2, row * block_size + block_size // 2


def get_valid_actions(state, walkable_matrix):
    """Restituisce tutte le azioni valide in 8 direzioni"""
    r, c = state
    height, width = len(walkable_matrix), len(walkable_matrix[0])
    actions = []
    
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width and walkable_matrix[nr][nc] == 1:
            actions.append((dr, dc))
    
    return actions


def get_next_state(state, action):
    """Calcola il prossimo stato"""
    r, c = state
    dr, dc = action
    return (r + dr, c + dc)


def calculate_reward(state, next_state, goal, penalty_matrix=None, visit_count=None, all_goals=None):
    """Calcola la ricompensa bilanciando distanza dal goal e Q-learning"""
    if next_state == goal:
        return 1000  # Ricompensa alta per raggiungere il goal
    
    # Penalità per avvicinarsi ad altri goal
    if all_goals:
        for other_goal in all_goals:
            if other_goal != goal:
                dist_to_other = np.hypot(next_state[0] - other_goal[0], next_state[1] - other_goal[1])
                if dist_to_other < 2.0:
                    return -15
    
    base_reward = -1.0
    
    # Reward per avvicinarsi al goal - MOLTO RIDOTTO
    current_dist = np.hypot(state[0] - goal[0], state[1] - goal[1])
    next_dist = np.hypot(next_state[0] - goal[0], next_state[1] - goal[1])
    dist_improvement = current_dist - next_dist
    
    # Reward/penalità basata sul miglioramento della distanza MOLTO RIDOTTA
    if dist_improvement > 0:
        improvement_reward = min(dist_improvement * 15, 10.0)  # Ulteriormente ridotto
        base_reward += improvement_reward
    else:
        base_reward += max(dist_improvement * 8, -8.0)  # Penalità ridotta
    
    # Penalità per cicli AUMENTATA
    if visit_count and next_state in visit_count:
        visits = visit_count[next_state]
        if visits >= MAX_STATE_VISITS:
            cycle_penalty = CYCLE_PENALTY * (visits / MAX_STATE_VISITS) * 4  # Aumentato
        else:
            cycle_penalty = CYCLE_PENALTY * (visits / MAX_STATE_VISITS) * 2  # Aumentato
        base_reward += cycle_penalty
    
    # Penalità ambientali mantenute
    if penalty_matrix is not None:
        penalty_value = penalty_matrix[next_state[0]][next_state[1]]
        
        if penalty_value > 5:  # Vicino ai muri
            base_reward -= penalty_value * 3.0
        elif penalty_value < 2:  # Corridoi o zone buone
            base_reward += (2 - penalty_value) * 2.0
        else:  # Zone intermedie
            base_reward -= penalty_value * 1.0
    
    return base_reward

def initialize_q_table(walkable_matrix):
    """Inizializza la tabella Q"""
    height, width = len(walkable_matrix), len(walkable_matrix[0])
    q_table = {}
    
    for r in range(height):
        for c in range(width):
            if walkable_matrix[r][c] == 1:
                state = (r, c)
                q_table[state] = {}
                
                for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                    if (0 <= r+dr < height and 0 <= c+dc < width and 
                        walkable_matrix[r+dr][c+dc] == 1):
                        q_table[state][(dr, dc)] = 0.0
    return q_table


def save_q_table(q_table, map_id):
    """Salva la tabella Q"""
    q_table_file = os.path.join(DATA_FOLDER, f"q_table-{map_id}.pkl")
    with open(q_table_file, "wb") as f:
        pickle.dump(q_table, f)
    print(f"Q-table salvata in {q_table_file}")


def load_q_table(map_id, walkable_matrix):
    """Carica la tabella Q"""
    q_table_file = os.path.join(DATA_FOLDER, f"q_table-{map_id}.pkl")
    if os.path.exists(q_table_file):
        with open(q_table_file, "rb") as f:
            q_table = pickle.load(f)
        print(f"Q-table caricata da {q_table_file}")
        return q_table
    else:
        print("Creazione di una nuova Q-table...")
        return initialize_q_table(walkable_matrix)


def select_action(state, q_table, valid_actions, exploration_rate, temperature=1.0, goal=None):
    """Selezione dell'azione che prioritizza i Q-values appresi"""
    if random.uniform(0, 1) < exploration_rate:
        # Esplorazione: usa principalmente Q-values con un piccolo bias verso il goal
        if state in q_table and q_table[state]:
            q_values = [q_table[state].get(action, 0) for action in valid_actions]
            q_values = np.array(q_values)
            
            # Bias verso il goal molto ridotto durante l'esplorazione
            if goal and random.uniform(0, 1) < 0.2:  # Solo 20% delle volte
                goal_bias = []
                for action in valid_actions:
                    next_s = (state[0] + action[0], state[1] + action[1])
                    current_dist = np.hypot(state[0] - goal[0], state[1] - goal[1])
                    next_dist = np.hypot(next_s[0] - goal[0], next_s[1] - goal[1])
                    improvement = current_dist - next_dist
                    goal_bias.append(improvement * 1)  # Bias molto ridotto
                
                q_values += np.array(goal_bias)
            
            # Softmax selection
            q_values = q_values - np.max(q_values)
            exp_values = np.exp(q_values / max(temperature, 0.1))
            probabilities = exp_values / np.sum(exp_values)
            
            action_idx = np.random.choice(len(valid_actions), p=probabilities)
            return valid_actions[action_idx]
        else:
            return random.choice(valid_actions)
    else:
        # Sfruttamento: usa PRINCIPALMENTE i Q-values
        if state in q_table and q_table[state]:
            # Trova l'azione con Q-value più alto
            best_q_value = float('-inf')
            best_actions = []
            
            for action in valid_actions:
                q_value = q_table[state].get(action, 0)
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_actions = [action]
                elif abs(q_value - best_q_value) < 0.01:  # Considera valori molto simili
                    best_actions.append(action)
            
            # Se ci sono più azioni con Q-value simile, usa il goal come tiebreaker
            if len(best_actions) > 1 and goal:
                return min(best_actions, 
                          key=lambda a: np.hypot(state[0]+a[0]-goal[0], state[1]+a[1]-goal[1]))
            else:
                return best_actions[0] if best_actions else valid_actions[0]
        else:
            # Nessuna conoscenza: vai verso il goal
            if goal:
                return min(valid_actions, 
                          key=lambda a: np.hypot(state[0]+a[0]-goal[0], state[1]+a[1]-goal[1]))
            return random.choice(valid_actions)        

def train_q_learning(walkable_matrix, goals, penalty_matrix=None):
    """Addestramento Q-learning migliorato"""
    q_table = initialize_q_table(walkable_matrix)
    
    all_steps = []
    all_rewards = []
    goal_reached_counts = {tuple(goal): 0 for goal in goals}
    success_episodes = 0
    
    print(f"Inizio training Q-learning per {EPISODES} episodi...")
    
    for episode in range(EPISODES):
        # Punto di partenza casuale
        walkable_points = [(r, c) for r in range(len(walkable_matrix)) 
                          for c in range(len(walkable_matrix[0])) 
                          if walkable_matrix[r][c] == 1]
        state = random.choice(walkable_points)
        
        # Selezione del goal - curriculum learning migliorato
        if episode < EPISODES * 0.2:
            # Prime fasi: goal più vicini
            goal = min(goals, key=lambda g: np.hypot(state[0] - g[0], state[1] - g[1]))
        elif episode < EPISODES * 0.6:
            # Fase intermedia: mix di goal vicini e casuali
            if random.random() < 0.7:
                goal = min(goals, key=lambda g: np.hypot(state[0] - g[0], state[1] - g[1]))
            else:
                goal = random.choice(goals)
        else:
            # Fase finale: goal completamente casuali
            goal = random.choice(goals)
        
        total_reward = 0
        steps = 0
        visit_count = {}
        path_states = []
        
        # Decadimento exploration rate più graduale
        current_exploration_rate = max(MIN_EXPLORATION_RATE, 
                                     EXPLORATION_RATE * (1 - 0.8 * episode / EPISODES))
        current_temperature = max(0.1, TEMPERATURE * (1 - 0.9 * episode / EPISODES))
        
        for step in range(MAX_STEPS):
            path_states.append(state)
            visit_count[state] = visit_count.get(state, 0) + 1
            valid_actions = get_valid_actions(state, walkable_matrix)
            
            if not valid_actions:
                break
            
            action = select_action(state, q_table, valid_actions, 
                                 current_exploration_rate, current_temperature, goal)
            next_state = get_next_state(state, action)
            
            reward = calculate_reward(state, next_state, goal, penalty_matrix, visit_count, goals)
            total_reward += reward
            
            # Aggiorna Q-value con learning rate adattivo
            next_valid_actions = get_valid_actions(next_state, walkable_matrix)
            next_max_q = 0
            if next_valid_actions and next_state in q_table:
                next_q_values = [q_table[next_state].get(next_action, 0) 
                                for next_action in next_valid_actions]
                if next_q_values:
                    next_max_q = max(next_q_values)
            
            if state not in q_table:
                q_table[state] = {}
            
            # Learning rate adattivo basato sul numero di visite
            state_visits = sum(1 for s in path_states if s == state)
            adaptive_lr = LEARNING_RATE / (1 + 0.1 * state_visits)
            
            old_q = q_table[state].get(action, 0)
            new_q = old_q + adaptive_lr * (reward + DISCOUNT_FACTOR * next_max_q - old_q)
            q_table[state][action] = new_q
            
            state = next_state
            steps += 1
            
            if state == goal:
                goal_reached_counts[goal] += 1
                success_episodes += 1
                # Bonus reward per tutti gli stati nel percorso di successo
                bonus = 10.0 / len(path_states)
                for i, path_state in enumerate(path_states):
                    if path_state in q_table and i < len(path_states) - 1:
                        next_path_state = path_states[i + 1]
                        action_taken = (next_path_state[0] - path_state[0], 
                                      next_path_state[1] - path_state[1])
                        if action_taken in q_table[path_state]:
                            q_table[path_state][action_taken] += bonus
                break
        
        all_steps.append(steps)
        all_rewards.append(total_reward)
        
        if episode % 1000 == 0:
            recent_steps = all_steps[-1000:] if len(all_steps) >= 1000 else all_steps
            avg_steps = sum(recent_steps) / len(recent_steps)
            success_rate = sum(1 for s in recent_steps if s < MAX_STEPS) / len(recent_steps)
            recent_success_rate = success_episodes / min(episode + 1, 1000) if episode < 1000 else success_episodes / 1000
            if episode >= 1000:
                success_episodes = sum(1 for s in all_steps[-1000:] if s < MAX_STEPS)
            
            print(f"Episodio {episode}: Passi medi={avg_steps:.1f}, Success rate={success_rate:.2f}, "
                  f"Exploration rate={current_exploration_rate:.3f}")
    
    print("\nTraining completato!")
    for goal, count in goal_reached_counts.items():
        print(f"Goal {goal}: raggiunto {count} volte ({100*count/EPISODES:.1f}%)")
    
    return q_table


def find_best_goal(start, goals, walkable_matrix, penalty_matrix):
    """Trova il goal migliore usando Dijkstra con penalità corrette"""
    base_cost = 1.0
    penalty_weight = 0.3  # Aumentato il peso delle penalità
    binary_walkable = np.array(walkable_matrix)
    
    # Ora penalty_matrix ha valori ALTI vicino ai muri, quindi li usiamo direttamente
    cost_matrix = np.where(
        binary_walkable == 1, 
        base_cost + penalty_matrix * penalty_weight,  # Più penalità = più costo
        float('inf')
    )
    
    height, width = cost_matrix.shape
    dist = np.full((height, width), float('inf'))
    dist[start[0], start[1]] = 0
    pq = [(0, start[0], start[1])]
    goals_set = set(goals)
    found_goals = {}
    
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    while pq and goals_set:
        d, r, c = heapq.heappop(pq)
        if (r, c) in goals_set:
            found_goals[(r, c)] = d
            goals_set.remove((r, c))
        
        for dr, dc in directions:
            nr, nc = r+dr, c+dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if np.isinf(cost_matrix[nr, nc]):
                continue
                
            move_cost = cost_matrix[nr, nc] * (1.0 if dr==0 or dc==0 else math.sqrt(2))
            new_dist = d + move_cost
            
            if new_dist < dist[nr, nc]:
                dist[nr, nc] = new_dist
                heapq.heappush(pq, (new_dist, nr, nc))
    
    if found_goals:
        return min(found_goals, key=found_goals.get)
    else:
        return min(goals, key=lambda g: np.hypot(start[0]-g[0], start[1]-g[1]))

def find_path_q_learning(walkable_matrix, start, goals, q_table, penalty_matrix=None, max_attempts=3):
    """Trova il percorso usando principalmente i Q-values appresi"""
    best_paths = []
    
    for attempt in range(max_attempts):
        if attempt == 0:
            best_goal = find_best_goal(start, goals, walkable_matrix, penalty_matrix)
        else:
            remaining_goals = [g for g in goals if g != best_goal]
            if remaining_goals:
                best_goal = random.choice(remaining_goals)
            else:
                best_goal = random.choice(goals)
        
        print(f"Tentativo {attempt + 1}: Goal selezionato: {best_goal}")
        
        path = [start]
        state = start
        visit_count = {}
        stuck_counter = 0  # Conta quante volte siamo rimasti nello stesso posto
        last_states = []   # Mantiene gli ultimi stati per rilevare loop
        
        for step in range(MAX_STEPS):
            if state == best_goal:
                print(f"Goal raggiunto in {step} passi!")
                return path, best_goal
                
            valid_actions = get_valid_actions(state, walkable_matrix)
            if not valid_actions:
                break
            
            visit_count[state] = visit_count.get(state, 0) + 1
            
            # Rileva se siamo bloccati in un loop
            last_states.append(state)
            if len(last_states) > 20:
                last_states.pop(0)
                
            # Conta quante volte gli ultimi 10 stati sono ripetuti
            if len(last_states) >= 10:
                unique_recent = len(set(last_states[-10:]))
                if unique_recent <= 3:  # Troppo pochi stati unici = loop
                    stuck_counter += 1
                else:
                    stuck_counter = max(0, stuck_counter - 1)
            
            # Logica anti-stallo: forza l'esplorazione solo se davvero bloccati
            force_exploration = (
                stuck_counter > 8 or                    # Loop evidente
                visit_count.get(state, 0) > 6 or       # Troppo tempo nello stesso stato
                (len(path) > 100 and len(set(path[-30:])) < 8)  # Pochi stati unici di recente
            )
            
            if force_exploration:
                print(f"Anti-stallo attivato al passo {step}, stuck_counter={stuck_counter}")
                # Scegli l'azione meno visitata che si avvicina al goal
                action_scores = []
                for action in valid_actions:
                    next_s = (state[0] + action[0], state[1] + action[1])
                    visits = visit_count.get(next_s, 0)
                    dist_to_goal = np.hypot(next_s[0] - best_goal[0], next_s[1] - best_goal[1])
                    
                    # Score: priorità agli stati meno visitati, poi distanza
                    score = -visits * 10 - dist_to_goal * 2
                    action_scores.append((action, score))
                
                action = max(action_scores, key=lambda x: x[1])[0]
                stuck_counter = max(0, stuck_counter - 2)  # Riduci il counter
                
            else:
                # Usa la policy normale con esplorazione molto bassa
                action = select_action(state, q_table, valid_actions, 
                                     0.02, 0.1, best_goal)  # Esplorazione minima
            
            next_state = get_next_state(state, action)
            path.append(next_state)
            state = next_state
            
            # Log periodico
            if step % 100 == 0 and step > 0:
                distance_to_goal = np.hypot(state[0] - best_goal[0], state[1] - best_goal[1])
                recent_unique = len(set(path[-20:]))
                print(f"Passo {step}: Distanza={distance_to_goal:.2f}, "
                      f"Stati unici recenti={recent_unique}, "
                      f"Stuck counter={stuck_counter}")
        
        # Salva questo tentativo
        final_distance = np.hypot(state[0] - best_goal[0], state[1] - best_goal[1])
        best_paths.append((path, best_goal, final_distance, len(path)))
        
        print(f"Tentativo {attempt + 1} completato. Distanza finale: {final_distance:.2f}, Passi: {len(path)}")
    
    # Scegli il migliore
    best_paths.sort(key=lambda x: (x[2], x[3]))
    best_path, best_goal, final_dist, path_length = best_paths[0]
    
    if final_dist < 3:
        print(f"Migliore risultato: Goal {best_goal}, distanza finale: {final_dist:.2f}")
    else:
        print(f"Nessun tentativo ha raggiunto il goal. Migliore: distanza {final_dist:.2f}")
    
    return best_path, best_goal


def plot_result(map_id, path, walkable_matrix, penalty_matrix, start, goal, block_size=5):
    """Visualizza il risultato con heatmap delle penalità"""
    img_path = os.path.join(DATA_FOLDER, f"{map_id}.png")
    img = np.array(Image.open(img_path))
    
    plt.figure(figsize=(14, 12))
    plt.imshow(img, alpha=0.6)
    
    # Heatmap delle penalità
    heatmap = np.kron(penalty_matrix, np.ones((block_size, block_size)))
    heatmap = heatmap[:img.shape[0], :img.shape[1]]
    plt.imshow(heatmap, cmap="RdYlGn_r", alpha=0.4, vmin=0, vmax=10)
    plt.colorbar(label='Livello di penalità')
    
    # Percorso
    if path:
        x_coords = [block_to_pixel(row, col, block_size)[0] for row, col in path]
        y_coords = [block_to_pixel(row, col, block_size)[1] for row, col in path]
        plt.plot(x_coords, y_coords, color="navy", linewidth=3, label="Percorso")
        
        # Punti di controllo
        step_markers = max(1, len(path) // 20)
        for i in range(0, len(path), step_markers):
            if i > 0:
                x, y = block_to_pixel(path[i][0], path[i][1], block_size)
                plt.scatter(x, y, color="gold", s=80, edgecolor='black', zorder=5)
    
    # Start e Goal
    start_x, start_y = block_to_pixel(start[0], start[1], block_size)
    plt.scatter(start_x, start_y, color="lime", s=300, edgecolor='black', 
               linewidth=2, zorder=10, label="Start")
    
    goal_x, goal_y = block_to_pixel(goal[0], goal[1], block_size)
    plt.scatter(goal_x, goal_y, color="red", s=300, marker='*', edgecolor='black',
               linewidth=2, zorder=10, label="Goal")
    
    plt.legend(loc='upper right', fontsize=12)
    plt.axis('off')
    
    # Statistiche
    distance = np.hypot(start[0] - goal[0], start[1] - goal[1])
    if path:
        efficiency = distance / len(path) if len(path) > 0 else 0
        title = f"Percorso: {len(path)} passi (Efficienza: {efficiency:.2f})"
    else:
        title = "Nessun percorso trovato!"
    
    plt.title(f"Q-Learning Pathfinding - Mappa {map_id}\n"
             f"Start: {start} → Goal: {goal} | {title}", fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.show()

def plot_all_goals(map_id, walkable_matrix, goals, block_size=5):
    img_path = os.path.join(DATA_FOLDER, f"{map_id}.png")
    img = np.array(Image.open(img_path))
    plt.figure(figsize=(10, 10))
    plt.imshow(img, alpha=0.6)

    for i, goal in enumerate(goals):
        goal_x, goal_y = block_to_pixel(goal[0], goal[1], block_size)
        plt.scatter(goal_x, goal_y, color="blue", s=100, zorder=10)
        plt.text(goal_x + 10, goal_y + 10, f"Goal {i+1}", fontsize=12, color="white", 
                 bbox=dict(facecolor='black', alpha=0.7))
    
    plt.axis('off')
    plt.title(f"Punti Goal per la Mappa {map_id}")
    plt.tight_layout()
    plt.show(block=True)

def main():
    available_maps = [703368, 703326, 703323, 703372, 703381, 703324]
    print("Mappe disponibili:", available_maps)
    MAP_ID = int(input("Inserisci MAP_ID: "))

    if MAP_ID not in available_maps:
        print("MAP_ID non valido.")
        return

    # Carica mappa
    map_obj = load_map(MAP_ID)
    block_size = 5
    walkable_matrix = create_walkable_matrix(map_obj, block_size)
    penalty_matrix = compute_penalty_matrix(walkable_matrix, block_size)

    # Gestione goals
    goals = load_goals(MAP_ID)
    if not goals:
        print("Seleziona i punti goal sulla mappa.")
        img_path = os.path.join(DATA_FOLDER, f"{MAP_ID}.png")
        goals = select_goals(img_path, walkable_matrix, block_size)
        if goals:
            save_goals(MAP_ID, goals)
        else:
            print("Nessun goal selezionato.")
            return
    else:
        print(f"Caricati {len(goals)} goals: {goals}")
        plot_all_goals(MAP_ID, walkable_matrix, goals, block_size)

    # Carica Q-table
    q_table = load_q_table(MAP_ID, walkable_matrix)

    # Training opzionale
    if input("Vuoi addestrare l'agente? (s/n): ").lower() == 's':
        print("Inizio training...")
        start_time = time.time()
        q_table = train_q_learning(walkable_matrix, goals, penalty_matrix)
        print(f"Training completato in {time.time() - start_time:.2f} secondi")
        save_q_table(q_table, MAP_ID)

    # Test del pathfinding
    print("\n" + "="*50)
    print("TEST DEL PATHFINDING")
    print("="*50)

    while True:
        # Selezione manuale del punto di partenza
        img_path = os.path.join(DATA_FOLDER, f"{MAP_ID}.png")
        start = select_point_on_map(img_path, walkable_matrix, block_size, 
                                   "Seleziona punto di partenza")
        if not start:
            print("Nessun punto di partenza selezionato.")
            break

        print(f"Punto di partenza: {start}")

        # Trova il percorso
        print("Calcolo del percorso...")
        start_time = time.time()
        path, chosen_goal = find_path_q_learning(walkable_matrix, start, goals, 
                                                q_table, penalty_matrix)
        pathfinding_time = time.time() - start_time

        # Risultati
        if path and path[-1] == chosen_goal:
            print(f"Percorso trovato: {len(path)} passi")
            print(f"Goal raggiunto: {chosen_goal}")
            print(f"Tempo: {pathfinding_time:.3f} secondi")
            
            distance = np.hypot(start[0] - chosen_goal[0], start[1] - chosen_goal[1])
            efficiency = distance / len(path)
            print(f"Efficienza: {efficiency:.3f}")
        else:
            print("Percorso non completato.")

        # Visualizzazione
        if input("Visualizzare il percorso? (s/n): ").lower() == 's':
            plot_result(MAP_ID, path, walkable_matrix, penalty_matrix, 
                       start, chosen_goal, block_size)

        if input("Testare un altro percorso? (s/n): ").lower() != 's':
            break

    print("Esecuzione completata!")


main()
