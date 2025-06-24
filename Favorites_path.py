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

def carica_preferenze(id_mappa):
    """Carica i punti del percorso preferibile se disponibili."""
    file_preferenze = os.path.join(CARTELLA_BASE, "preferenze.json")
    if not os.path.exists(file_preferenze):
        return []
    
    try:
        with open(file_preferenze, "r") as f:
            preferenze = json.load(f)
            return preferenze.get(str(id_mappa), [])
    except (json.JSONDecodeError, KeyError):
        return []

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

def calcola_matrice_penalità(matrice_calpestabile, dimensione_blocco, fattore_scalabilità=1.5):
    """Calcola penalità per blocchi vicini ai muri, con raggio di penalità adattato alla dimensione del blocco."""
    h, w = matrice_calpestabile.shape
    raggio_penalità = dimensione_blocco * fattore_scalabilità
    distanza = distance_transform_edt(matrice_calpestabile)
    
    # Creiamo la penalità in base alla distanza ai muri
    penalità = np.where(distanza < raggio_penalità, (raggio_penalità - distanza) ** 2, 0)
    
    # Normalizziamo la penalità per non farla andare fuori scala
    penalità = 10 * penalità / penalità.max() if penalità.max() > 0 else penalità
    return penalità

def calcola_matrice_incentivi(matrice_calpestabile, punti_preferibili, dimensione_blocco, raggio_incentivo=2, bonus_incentivo=3):
    """Calcola gli incentivi per i percorsi preferibili (versione ottimizzata)."""
    h, w = matrice_calpestabile.shape
    matrice_incentivi = np.zeros((h, w))
    
    if not punti_preferibili:
        return matrice_incentivi
    
    print(f"🔄 Calcolo incentivi per {len(punti_preferibili)} punti preferibili...")
    
    # Converti i punti pixel in coordinate blocco
    punti_blocco = []
    for x, y in punti_preferibili:
        r, c = pixel_a_blocco(x, y, dimensione_blocco)
        if 0 <= r < h and 0 <= c < w:
            punti_blocco.append((r, c))
    
    print(f"   Punti blocco validi: {len(punti_blocco)}")
    
    # Usa meshgrid per calcolo vettorizzato più efficiente
    rows, cols = np.meshgrid(range(h), range(w), indexing='ij')
    
    # Applica incentivi per ogni punto
    for i, (riga_p, colonna_p) in enumerate(punti_blocco):
        if i % 5 == 0:  # Progress ogni 5 punti
            print(f"   Processando punto {i+1}/{len(punti_blocco)}")
            
        # Calcola distanza da questo punto per tutta la matrice
        distanze = np.sqrt((rows - riga_p)**2 + (cols - colonna_p)**2)
        
        # Applica incentivo dove la distanza è <= raggio e il blocco è calpestabile
        mask = (distanze <= raggio_incentivo) & (matrice_calpestabile == 1)
        incentivo = bonus_incentivo * (1 - distanze / raggio_incentivo)
        
        # Aggiorna solo dove il nuovo incentivo è maggiore
        matrice_incentivi = np.where(mask & (incentivo > matrice_incentivi), 
                                   incentivo, matrice_incentivi)
    
    # Crea incentivi lungo i segmenti (versione semplificata)
    print("   Collegamento segmenti...")
    for i in range(len(punti_blocco) - 1):
        r1, c1 = punti_blocco[i]
        r2, c2 = punti_blocco[i + 1]
        
        # Usa l'algoritmo di Bresenham semplificato
        steps = max(abs(r2 - r1), abs(c2 - c1))
        if steps > 0:
            for step in range(0, steps + 1, 2):  # Skip alcuni punti per velocità
                t = step / steps
                r_interp = int(r1 + t * (r2 - r1))
                c_interp = int(c1 + t * (c2 - c1))
                
                if 0 <= r_interp < h and 0 <= c_interp < w:
                    # Applica incentivo in area ridotta attorno al punto
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            nr, nc = r_interp + dr, c_interp + dc
                            if (0 <= nr < h and 0 <= nc < w and 
                                matrice_calpestabile[nr, nc]):
                                matrice_incentivi[nr, nc] = max(
                                    matrice_incentivi[nr, nc], bonus_incentivo * 0.5)
    
    print(f" Matrice incentivi calcolata. Max incentivo: {matrice_incentivi.max():.2f}")
    return matrice_incentivi

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

def a_star(matrice_calpestabile, inizio, obiettivo, matrice_penalità=None, matrice_incentivi=None, max_iterazioni=50000):
    """Algoritmo A* ottimizzato con supporto per penalità e incentivi personalizzati."""
    h, w = matrice_calpestabile.shape
    da_esplorare = [(0, inizio)]
    esplorati = set()
    da_dove = {}
    costo_g = {inizio: 0}
    costo_f = {inizio: euristica(inizio, obiettivo)}
    
    iterazioni = 0
    
    def vicini(pos):
        r, c = pos
        # Prima direzioni cardinali, poi diagonali per efficienza
        direzioni = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in direzioni:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and matrice_calpestabile[nr, nc] and (nr, nc) not in esplorati:
                yield (nr, nc)

    print(f"   Partenza: {inizio}, Obiettivo: {obiettivo}")
    print(f"   Distanza euclidea: {euristica(inizio, obiettivo):.1f}")

    while da_esplorare and iterazioni < max_iterazioni:
        if iterazioni % 5000 == 0 and iterazioni > 0:
            print(f"   Iterazione {iterazioni}, nodi da esplorare: {len(da_esplorare)}")
        
        _, attuale = heapq.heappop(da_esplorare)
        
        if attuale in esplorati:
            continue
            
        esplorati.add(attuale)
        
        if attuale == obiettivo:
            print(f"Percorso trovato in {iterazioni} iterazioni!")
            return ricostruisci_percorso(da_dove, attuale)

        for vicino in vicini(attuale):
            # Calcolo costo ottimizzato
            dr, dc = vicino[0] - attuale[0], vicino[1] - attuale[1]
            costo_movimento = 1.0 if (dr == 0 or dc == 0) else 1.414  # Approssimazione sqrt(2)
            
            # Aggiungi penalità e sottrai incentivi
            penalità = matrice_penalità[vicino] if matrice_penalità is not None else 0
            incentivo = matrice_incentivi[vicino] if matrice_incentivi is not None else 0
            
            tentativo_g = costo_g[attuale] + costo_movimento + penalità - incentivo

            if tentativo_g < costo_g.get(vicino, float("inf")):
                da_dove[vicino] = attuale
                costo_g[vicino] = tentativo_g
                costo_f[vicino] = tentativo_g + euristica(vicino, obiettivo)
                heapq.heappush(da_esplorare, (costo_f[vicino], vicino))
        
        iterazioni += 1

    if iterazioni >= max_iterazioni:
        print(f" Timeout dopo {max_iterazioni} iterazioni")
    else:
        print(f" Nessun percorso trovato dopo {iterazioni} iterazioni")
    
    return []

def euristica(a, b):
    """Stima della distanza tra due punti (distanza euclidea)."""
    return np.hypot(a[0] - b[0], a[1] - b[1])

def ricostruisci_percorso(da_dove, attuale):
    """Ricostruisce il percorso una volta raggiunto l'obiettivo."""
    percorso = [attuale]
    while attuale in da_dove:
        attuale = da_dove[attuale]
        percorso.append(attuale)
    return percorso[::-1]

def mostra_percorso_e_preferenze(id_mappa, percorso, matrice_penalità, matrice_incentivi, 
                                dimensione_blocco, punti_preferibili):
    """Visualizza il percorso, la mappa delle penalità e i punti preferibili."""
    img = np.array(Image.open(os.path.join(CARTELLA_DATI, f"{id_mappa}.png")))
    
    # Combina penalità e incentivi per la visualizzazione
    mappa_costi = matrice_penalità - matrice_incentivi
    costi_img = np.kron(mappa_costi, np.ones((dimensione_blocco, dimensione_blocco)))
    costi_img = costi_img[:img.shape[0], :img.shape[1]]

    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    plt.imshow(costi_img, cmap="RdYlGn_r", alpha=0.5)

    # Mostra il percorso calcolato
    if percorso:
        x, y = zip(*[(c * dimensione_blocco + dimensione_blocco // 2, 
                     r * dimensione_blocco + dimensione_blocco // 2) for r, c in percorso])
        plt.plot(x, y, color="black", linewidth=3, label="Percorso A*")
        plt.scatter(x[0], y[0], color="green", s=100, label="Start", zorder=5)
        plt.scatter(x[-1], y[-1], color="blue", s=100, label="Goal", zorder=5)

    # Mostra i punti del percorso preferibile
    if punti_preferibili:
        pref_x, pref_y = zip(*punti_preferibili)
        plt.plot(pref_x, pref_y, color="cyan", linewidth=2, marker='o', 
                markersize=4, label="Percorso Preferibile", alpha=0.8)

    plt.axis("off")
    plt.legend()
    plt.title("Percorso A* con Preferenze (verde=incentivi, giallo=neutro, rosso=penalità)")
    plt.tight_layout()
    plt.show()

def seleziona_uscite(id_mappa, dimensione_blocco, matrice_calpestabile, blocco_inizio):
    """Seleziona l'uscita più vicina al punto di partenza."""
    uscita_file = os.path.join(CARTELLA_DATI, f"uscite-{id_mappa}.json")
    with open(uscita_file, "r") as f:
        uscite_pixel = json.load(f)

    uscite_blocco = [pixel_a_blocco(x, y, dimensione_blocco) for x, y in uscite_pixel]
    uscite_valide = [usc for usc in uscite_blocco if matrice_calpestabile[usc] == 1]

    if not uscite_valide:
        raise ValueError("Nessuna uscita valida trovata nella mappa.")
    
    # Seleziona l'uscita più vicina al punto di partenza
    return min(uscite_valide, key=lambda u: euristica(blocco_inizio, u))

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

    print(" Caricamento mappa...")
    dimensione_blocco = 5
    mappa = carica_mappa(id_mappa)
    
    print(" Creazione matrice calpestabile...")
    matrice_calpestabile = crea_matrice_calpestabile(mappa, dimensione_blocco)
    print(f"   Dimensioni matrice: {matrice_calpestabile.shape}")

    # Carica i punti del percorso preferibile
    punti_preferibili = carica_preferenze(id_mappa)
    if punti_preferibili:
        print(f" Caricati {len(punti_preferibili)} punti del percorso preferibile.")
    else:
        print(" Nessun percorso preferibile trovato per questa mappa.")

    # Ottieni il punto di partenza dal click dell'utente
    print(" Selezione punto di partenza...")
    punto_click = prendi_punto_da_click(os.path.join(CARTELLA_DATI, f"{id_mappa}.png"))
    if not punto_click:
        print("Errore nella selezione del punto di partenza.")
        return

    blocco_inizio = pixel_a_blocco(*punto_click, dimensione_blocco)
    print(" Selezione uscita...")
    uscita_selezionata = seleziona_uscite(id_mappa, dimensione_blocco, matrice_calpestabile, blocco_inizio)

    print(f"Partenza: {blocco_inizio}, Uscita selezionata: {uscita_selezionata}")

    # Calcola penalità e incentivi
    print(" Calcolo penalità...")
    matrice_penalità = calcola_matrice_penalità(matrice_calpestabile, dimensione_blocco)
    
    print(" Calcolo incentivi...")
    matrice_incentivi = calcola_matrice_incentivi(matrice_calpestabile, punti_preferibili, dimensione_blocco)

    # Ricerca percorso con A*
    print(" Ricerca percorso con A*...")
    percorso = a_star(matrice_calpestabile, blocco_inizio, uscita_selezionata, 
                     matrice_penalità, matrice_incentivi)
    if not percorso:
        print(" Nessun percorso trovato.")
        return

    print(f" Percorso trovato con {len(percorso)} punti.")
    print(" Creazione visualizzazione...")
    mostra_percorso_e_preferenze(id_mappa, percorso, matrice_penalità, matrice_incentivi, 
                                dimensione_blocco, punti_preferibili)

main()
    