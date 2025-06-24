import os
import sys
import json
import heapq
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import distance_transform_edt
import time

# Imposta i percorsi delle cartelle
CARTELLA_BASE = os.path.dirname(__file__)
CARTELLA_DATI = os.path.join(CARTELLA_BASE, "data")
sys.path.append(CARTELLA_BASE)

from models import Map

def carica_mappa(id_mappa):
    """Carica la mappa e i dati di calpestabilità associati."""
    with open(os.path.join(CARTELLA_DATI, "maps.json"), encoding="utf-8") as f:
        mappe = {m["id"]: Map.from_dict(m) for m in json.load(f)}

    if id_mappa not in mappe:
        raise ValueError(f"ID mappa {id_mappa} non trovato.")

    mappa = mappe[id_mappa]
    percorso_bin = os.path.join(CARTELLA_DATI, f"walkable-{id_mappa}.bin")
    with open(percorso_bin, "rb") as f:
        mappa.walkable_image_bytes = f.read()

    return mappa

def crea_matrice_calpestabile(mappa, dimensione_blocco):
    """Crea una matrice binaria dove 1 indica blocchi completamente calpestabili."""
    righe = mappa.height // dimensione_blocco
    colonne = mappa.width // dimensione_blocco
    matrice = np.zeros((righe, colonne), dtype=np.uint8)

    for riga in range(righe):
        for colonna in range(colonne):
            x0, y0 = colonna * dimensione_blocco, riga * dimensione_blocco
            if all(mappa.pixel_is_walkable(x0 + dx, y0 + dy)
                   for dx in range(dimensione_blocco)
                   for dy in range(dimensione_blocco)):
                matrice[riga, colonna] = 1
    return matrice

def calcola_matrice_penalità(matrice_calpestabile, dimensione_blocco, fattore_scalabilità=4):
    """Calcola penalità per blocchi vicini ai muri, con raggio di penalità adattato alla dimensione del blocco."""
    h, w = matrice_calpestabile.shape
    raggio_penalità = dimensione_blocco * fattore_scalabilità
    distanza = distance_transform_edt(matrice_calpestabile)
    penalità = np.where(distanza < raggio_penalità, (raggio_penalità - distanza) ** 2, 0)
    penalità = 10 * penalità / penalità.max() if penalità.max() > 0 else penalità
    return penalità

def prendi_punto_da_click(percorso_immagine):
    """Permette all'utente di cliccare sul punto di START su un'immagine."""
    immagine = np.array(Image.open(percorso_immagine))
    coordinate = []

    def al_click(evento):
        if evento.xdata and evento.ydata:
            coordinate.append((int(evento.xdata), int(evento.ydata)))
            plt.plot(evento.xdata, evento.ydata, 'ro')
            plt.draw()
            if len(coordinate) == 1:
                plt.close()

    fig, ax = plt.subplots()
    ax.imshow(immagine)
    fig.canvas.mpl_connect('button_press_event', al_click)
    print("Clicca su START (un click).")
    plt.show()

    return coordinate[0] if len(coordinate) == 1 else None

def pixel_a_blocco(x, y, dimensione_blocco):
    """Converti coordinate pixel in coordinate blocco."""
    return y // dimensione_blocco, x // dimensione_blocco

def blocco_a_pixel(riga, colonna, dimensione_blocco):
    """Converti coordinate blocco nel centro in pixel."""
    return colonna * dimensione_blocco + dimensione_blocco // 2, riga * dimensione_blocco + dimensione_blocco // 2

def dijkstra(matrice_calpestabile, inizio, obiettivo, matrice_penalità=None):
    """Algoritmo di Dijkstra con supporto per penalità personalizzate per zona."""
    h, w = matrice_calpestabile.shape
    da_esplorare = [(0, inizio)]
    da_dove = {}
    costo_g = {inizio: 0}

    def vicini(pos):
        r, c = pos
        direzioni = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in direzioni:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and matrice_calpestabile[nr, nc]:
                yield (nr, nc)

    while da_esplorare:
        costo_attuale, attuale = heapq.heappop(da_esplorare)
        if attuale == obiettivo:
            return ricostruisci_percorso(da_dove, attuale)

        for vicino in vicini(attuale):
            costo_movimento = np.hypot(vicino[0] - attuale[0], vicino[1] - attuale[1])
            penalità = matrice_penalità[vicino] if matrice_penalità is not None else 0
            tentativo_g = costo_g[attuale] + costo_movimento + penalità

            if tentativo_g < costo_g.get(vicino, float("inf")):
                da_dove[vicino] = attuale
                costo_g[vicino] = tentativo_g
                heapq.heappush(da_esplorare, (tentativo_g, vicino))

    return []

def ricostruisci_percorso(da_dove, attuale):
    """Ricostruisce il percorso una volta raggiunto l'obiettivo."""
    percorso = [attuale]
    while attuale in da_dove:
        attuale = da_dove[attuale]
        percorso.append(attuale)
    return percorso[::-1]

def mostra_percorso_e_penalità(id_mappa, percorso, matrice_penalità, dimensione_blocco):
    """Visualizza il percorso e la mappa delle penalità."""
    img = np.array(Image.open(os.path.join(CARTELLA_DATI, f"{id_mappa}.png")))
    penalità_img = np.kron(matrice_penalità, np.ones((dimensione_blocco, dimensione_blocco)))
    penalità_img = penalità_img[:img.shape[0], :img.shape[1]]

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.imshow(penalità_img, cmap="RdYlGn_r", alpha=0.5)

    x, y = zip(*[(c * dimensione_blocco + dimensione_blocco // 2, r * dimensione_blocco + dimensione_blocco // 2) for r, c in percorso])
    plt.plot(x, y, color="black", linewidth=2)
    plt.scatter(x[0], y[0], color="green", label="Start")
    plt.scatter(x[-1], y[-1], color="blue", label="Goal")

    plt.axis("off")
    plt.legend()
    plt.title("Percorso Dijkstra e penalità (verde = calpestabile, rosso = non calpestabile)")
    plt.show()

def seleziona_uscite(id_mappa, dimensione_blocco, matrice_calpestabile):
    """Seleziona l'uscita più vicina al punto di partenza."""
    uscita_file = os.path.join(CARTELLA_DATI, f"uscite-{id_mappa}.json")
    with open(uscita_file, "r") as f:
        uscite_pixel = json.load(f)

    uscite_blocco = [pixel_a_blocco(x, y, dimensione_blocco) for x, y in uscite_pixel]
    uscite_valide = [usc for usc in uscite_blocco if matrice_calpestabile[usc] == 1]

    if not uscite_valide:
        raise ValueError("Nessuna uscita valida trovata nella mappa.")
    
    return min(uscite_valide, key=lambda u: np.hypot(blocco_inizio[0] - u[0], blocco_inizio[1] - u[1]))

def main():
    mappe_disponibili = [703368, 703326, 703323, 703372, 703381, 549362, 703428]
    print(f"Mappe disponibili: {mappe_disponibili}")
    try:
        id_mappa = int(input("Inserisci ID della mappa: "))
        if id_mappa not in mappe_disponibili:
            raise ValueError("ID mappa non valido.")
    except ValueError as e:
        print(e)
        return

    dimensione_blocco = 5
    mappa = carica_mappa(id_mappa)
    matrice_calpestabile = crea_matrice_calpestabile(mappa, dimensione_blocco)

    punto_click = prendi_punto_da_click(os.path.join(CARTELLA_DATI, f"{id_mappa}.png"))
    if not punto_click:
        print("Errore nella selezione del punto di partenza.")
        return

    global blocco_inizio
    blocco_inizio = pixel_a_blocco(*punto_click, dimensione_blocco)
    uscita_selezionata = seleziona_uscite(id_mappa, dimensione_blocco, matrice_calpestabile)

    print(f"Partenza: {blocco_inizio}, Uscita selezionata: {uscita_selezionata}")

    matrice_penalità = calcola_matrice_penalità(matrice_calpestabile, dimensione_blocco)

    percorso = dijkstra(matrice_calpestabile, blocco_inizio, uscita_selezionata, matrice_penalità)
    
    
    start_time = time.time()
    percorso = dijkstra(matrice_calpestabile, blocco_inizio, uscita_selezionata, matrice_penalità)
    tempo_esecuzione = time.time() - start_time  
    
    if not percorso:
        print("Nessun percorso trovato.")
        return
    
    lunghezza_percorso = len(percorso)
    costo_totale = 0
    for i in range(1, len(percorso)):
       r1, c1 = percorso[i - 1]
       r2, c2 = percorso[i]
       distanza = np.hypot(r2 - r1, c2 - c1)
       penalità = matrice_penalità[r2, c2]
       costo_totale += distanza + penalità
   
    print("\n=== METRICHE PERCORSO (Dijkstra) ===")
    print(f"Lunghezza (blocchi): {lunghezza_percorso}")
    print(f"Costo totale (distanza + penalità): {costo_totale:.2f}")
    print(f"Tempo di esecuzione: {tempo_esecuzione:.4f} secondi")
    
    mostra_percorso_e_penalità(id_mappa, percorso, matrice_penalità, dimensione_blocco)


main()
 
