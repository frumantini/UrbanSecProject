import os
import sys
import json
import heapq
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import distance_transform_edt

CARTELLA_BASE = os.path.dirname(__file__)
CARTELLA_DATI = os.path.join(CARTELLA_BASE, "data")
sys.path.append(CARTELLA_BASE)

from models import Map

def carica_mappa(id_mappa):
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
    righe = mappa.height // dimensione_blocco
    colonne = mappa.width // dimensione_blocco
    matrice = np.zeros((righe, colonne), dtype=np.uint8)
    for r in range(righe):
        for c in range(colonne):
            x0, y0 = c * dimensione_blocco, r * dimensione_blocco
            if all(mappa.pixel_is_walkable(x0 + dx, y0 + dy)
                   for dx in range(dimensione_blocco)
                   for dy in range(dimensione_blocco)):
                matrice[r, c] = 1
    return matrice

def calcola_matrice_penalità(matrice_calpestabile, dimensione_blocco, fattore_scalabilità=4):
    h, w = matrice_calpestabile.shape
    raggio_penalità = dimensione_blocco * fattore_scalabilità
    distanza = distance_transform_edt(matrice_calpestabile)
    penalità = np.where(distanza < raggio_penalità, (raggio_penalità - distanza) ** 2, 0)
    penalità = 10 * penalità / penalità.max() if penalità.max() > 0 else penalità
    return penalità

def prendi_punto_da_click(percorso_immagine, numero_click=1, messaggio="clicca"):
    immagine = np.array(Image.open(percorso_immagine))
    coordinate = []
    def al_click(evento):
        if evento.xdata is not None and evento.ydata is not None:
            coordinate.append((int(evento.xdata), int(evento.ydata)))
            plt.plot(evento.xdata, evento.ydata, 'ro')
            plt.draw()
            if len(coordinate) == numero_click:
                plt.close()
    fig, ax = plt.subplots()
    ax.imshow(immagine)
    fig.canvas.mpl_connect('button_press_event', al_click)
    print(f"{messaggio} ({numero_click} click attesi).")
    plt.show()
    return coordinate

def pixel_a_blocco(x, y, dimensione_blocco):
    return y // dimensione_blocco, x // dimensione_blocco

def blocco_a_pixel(riga, colonna, dimensione_blocco):
    return colonna * dimensione_blocco + dimensione_blocco // 2, riga * dimensione_blocco + dimensione_blocco // 2

def euristica(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])

def ricostruisci_percorso(da_dove, attuale):
    percorso = [attuale]
    while attuale in da_dove:
        attuale = da_dove[attuale]
        percorso.append(attuale)
    return percorso[::-1]

def a_star(matrice_calpestabile, inizio, obiettivo, matrice_penalità=None):
    h, w = matrice_calpestabile.shape
    da_esplorare = [(0, inizio)]
    da_dove = {}
    costo_g = {inizio: 0}
    costo_f = {inizio: euristica(inizio, obiettivo)}

    def vicini(pos):
        r, c = pos
        direzioni = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in direzioni:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and matrice_calpestabile[nr, nc]:
                yield (nr, nc)

    while da_esplorare:
        _, attuale = heapq.heappop(da_esplorare)
        if attuale == obiettivo:
            return ricostruisci_percorso(da_dove, attuale)
        for vicino in vicini(attuale):
            costo_movimento = np.hypot(vicino[0] - attuale[0], vicino[1] - attuale[1])
            penalità_base = matrice_penalità[vicino] if matrice_penalità is not None else 0
            tentativo_g = costo_g[attuale] + costo_movimento + penalità_base
            if tentativo_g < costo_g.get(vicino, float("inf")):
                da_dove[vicino] = attuale
                costo_g[vicino] = tentativo_g
                costo_f[vicino] = tentativo_g + euristica(vicino, obiettivo)
                heapq.heappush(da_esplorare, (costo_f[vicino], vicino))
    return []

def seleziona_uscite(id_mappa, dimensione_blocco, matrice_calpestabile):
    uscita_file = os.path.join(CARTELLA_DATI, f"uscite-{id_mappa}.json")
    with open(uscita_file, "r") as f:
        uscite_pixel = json.load(f)
    uscite_blocco = [pixel_a_blocco(x, y, dimensione_blocco) for x, y in uscite_pixel]
    h, w = matrice_calpestabile.shape
    uscite_valide = [u for u in uscite_blocco if 0 <= u[0] < h and 0 <= u[1] < w and matrice_calpestabile[u] == 1]
    if not uscite_valide:
        raise ValueError("Nessuna uscita valida trovata nella mappa.")
    return uscite_valide

def trova_percorso_sicuro(matrice_calpestabile, blocco_inizio, uscite, matrice_penalità, zone_pericolo=None):
    blocco_inizio = tuple(blocco_inizio)
    zone_pericolo_set = set(tuple(z) for z in zone_pericolo) if zone_pericolo else set()
    uscite_ordinate = sorted(uscite, key=lambda u: euristica(blocco_inizio, u))
    matrice_temp = np.copy(matrice_calpestabile)
    for blocco in zone_pericolo_set:
        matrice_temp[blocco] = 0
    for uscita in uscite_ordinate:
        percorso = a_star(matrice_temp, blocco_inizio, uscita, matrice_penalità)
        if percorso:
            return percorso, uscita, True
    for uscita in uscite_ordinate:
        percorso = a_star(matrice_calpestabile, blocco_inizio, uscita, matrice_penalità)
        if percorso:
            return percorso, uscita, False
    return None, None, False

def mostra_percorso_e_penalità(id_mappa, percorso, matrice_penalità, dimensione_blocco, zone_pericolo_blocchi=None):
    """Visualizza il percorso, le penalità e le zone di pericolo (verde = calpestabile, rosso = non calpestabile, rosso intenso = pericolo)."""
    img = np.array(Image.open(os.path.join(CARTELLA_DATI, f"{id_mappa}.png")))
    
    penalità_img = np.kron(matrice_penalità, np.ones((dimensione_blocco, dimensione_blocco)))
    penalità_img = penalità_img[:img.shape[0], :img.shape[1]]

    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    
    plt.imshow(penalità_img, cmap="RdYlGn_r", alpha=0.6)

    
    if zone_pericolo_blocchi:
        zone_pericolo_img = np.zeros((img.shape[0], img.shape[1]))
        for r, c in zone_pericolo_blocchi:
            r_start = r * dimensione_blocco
            c_start = c * dimensione_blocco
            r_end = min(r_start + dimensione_blocco, img.shape[0])
            c_end = min(c_start + dimensione_blocco, img.shape[1])
            zone_pericolo_img[r_start:r_end, c_start:c_end] = 1
        
        masked_danger = np.ma.masked_where(zone_pericolo_img == 0, zone_pericolo_img)
        plt.imshow(masked_danger, cmap="Reds", alpha=0.8, vmin=0, vmax=1)

    # Trova zone di pericolo attraversate dal percorso
    zone_attraversate = []
    if zone_pericolo_blocchi:
        zone_pericolo_set = set(zone_pericolo_blocchi)
        zone_attraversate = [blocco for blocco in percorso if blocco in zone_pericolo_set]

    # Disegna il percorso
    x, y = zip(*[(c * dimensione_blocco + dimensione_blocco // 2, 
                  r * dimensione_blocco + dimensione_blocco // 2) for r, c in percorso])
    plt.plot(x, y, color="black", linewidth=3)
    
    # Marca start e goal
    plt.scatter(x[0], y[0], color="green", s=150, label="Start", edgecolors='black', linewidth=1)
    plt.scatter(x[-1], y[-1], color="blue", s=150, label="Goal", edgecolors='black', linewidth=1)

    # Evidenzia i punti dove il percorso attraversa zone di pericolo
    if zone_attraversate:
        x_danger, y_danger = zip(*[(c * dimensione_blocco + dimensione_blocco // 2,
                                   r * dimensione_blocco + dimensione_blocco // 2)
                                  for r, c in zone_attraversate])
        plt.scatter(x_danger, y_danger, color="red", s=120, marker="X", 
                   linewidths=2, label="Attraversamento pericolo", edgecolors='white')

    plt.axis("off")
    plt.legend(loc='upper right', fontsize=12)
    
    # Titolo che indica la presenza di zone di pericolo
    if zone_pericolo_blocchi:
        plt.title("Percorso A* e penalità (verde = calpestabile, rosso = non calpestabile, ROSSO INTENSO = zone di pericolo)", fontsize=12)
    else:
        plt.title("Percorso A* e penalità (verde = calpestabile, rosso = non calpestabile)", fontsize=12)
    
    plt.tight_layout()
    plt.show()

    # Stampa statistiche
    if zone_pericolo_blocchi:
        print(f"\n--- RISULTATI ---")
        print(f"Zone di pericolo totali: {len(zone_pericolo_blocchi)}")
        print(f"Zone di pericolo attraversate: {len(zone_attraversate)}")
        if zone_attraversate:
            print(f"Blocchi di pericolo attraversati: {zone_attraversate}")
            print(" ATTENZIONE: Il percorso attraversa zone di pericolo!")
        else:
            print("Percorso completamente sicuro!")
        print(f"Lunghezza percorso: {len(percorso)} blocchi")
        print("-" * 50)

def blocchi_area_circolare(blocco_centro, raggio_blocchi, h, w):
    r0, c0 = blocco_centro
    blocchi = set()
    for dr in range(-raggio_blocchi, raggio_blocchi + 1):
        for dc in range(-raggio_blocchi, raggio_blocchi + 1):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < h and 0 <= c < w and np.hypot(dr, dc) <= raggio_blocchi:
                blocchi.add((r, c))
    return blocchi

def main():
    mappe_disponibili = [703368, 703326, 703323, 703372, 703381, 703428]
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

    percorso_immagine = os.path.join(CARTELLA_DATI, f"{id_mappa}.png")
    punto_click = prendi_punto_da_click(percorso_immagine, 1, "Clicca su START (un click).")
    if not punto_click:
        print("Errore nella selezione del punto di partenza.")
        return

    x_start, y_start = punto_click[0]
    blocco_inizio = pixel_a_blocco(x_start, y_start, dimensione_blocco)
    print(f"Punto di partenza selezionato: pixel ({x_start}, {y_start}), blocco {blocco_inizio}")

    h, w = matrice_calpestabile.shape
    r_start, c_start = blocco_inizio
    if r_start < 0 or r_start >= h or c_start < 0 or c_start >= w or matrice_calpestabile[r_start, c_start] == 0:
        print(f"ERRORE: Il punto di partenza ({r_start}, {c_start}) non è calpestabile o è fuori dalla mappa.")
        return

    print("Seleziona zone di pericolo (click sinistro multipli). Quando hai finito, chiudi la finestra o fai click destro.")
    immagine = np.array(Image.open(percorso_immagine))
    fig, ax = plt.subplots()
    ax.imshow(immagine)
    zone_pericolo_blocchi = set()

    def on_click(event):
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            blocco = pixel_a_blocco(x, y, dimensione_blocco)
            raggio_blocchi = 3  #  aumentare/diminuire l'area (3=circa 7x7 blocchi)
            blocchi_area = blocchi_area_circolare(blocco, raggio_blocchi, h, w)
            blocchi_area = {b for b in blocchi_area if matrice_calpestabile[b] == 1}
            zone_pericolo_blocchi.update(blocchi_area)
            for b in blocchi_area:
                x_plot, y_plot = blocco_a_pixel(b[0], b[1], dimensione_blocco)
                ax.plot(x_plot, y_plot, 'rx')
            plt.draw()
            print(f"Zona di pericolo aggiunta: pixel ({x}, {y}), blocco centrale {blocco}, area di {len(blocchi_area)} blocchi")
        elif event.button == 3:
            plt.close()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.title("Clicca per selezionare zone di pericolo, tasto destro per terminare.")
    plt.show()

    print(f"Numero di zone di pericolo selezionate: {len(zone_pericolo_blocchi)}")

    matrice_penalità = calcola_matrice_penalità(matrice_calpestabile, dimensione_blocco)
    uscite = seleziona_uscite(id_mappa, dimensione_blocco, matrice_calpestabile)
    print(f"Numero di uscite valide trovate: {len(uscite)}")

    zone_pericolo_sicure = set(zone_pericolo_blocchi)
    if blocco_inizio in zone_pericolo_sicure:
        zone_pericolo_sicure.remove(blocco_inizio)
        print("Punto di partenza rimosso dalle zone di pericolo.")
    for uscita in uscite:
        if uscita in zone_pericolo_sicure:
            zone_pericolo_sicure.remove(uscita)
            print(f"Punto di uscita {uscita} rimosso dalle zone di pericolo.")

    percorso, uscita_scelta, è_sicuro = trova_percorso_sicuro(
        matrice_calpestabile, blocco_inizio, uscite, matrice_penalità, zone_pericolo_sicure
    )

    if not percorso:
        print("Nessun percorso trovato, neanche passando per zone di pericolo.")
        return

    if è_sicuro:
        print(f"Trovato un percorso completamente sicuro verso l'uscita {uscita_scelta}")
    else:
        print(f"Trovato un percorso verso l'uscita {uscita_scelta} che attraversa zone di pericolo.")

    mostra_percorso_e_penalità(id_mappa, percorso, matrice_penalità, dimensione_blocco, zone_pericolo_sicure)
    


main()
