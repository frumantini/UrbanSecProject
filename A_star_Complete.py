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

def calcola_matrice_penalità(matrice_calpestabile, dimensione_blocco, fattore_scalabilità=2.0):
    """Calcola penalità per blocchi vicini ai muri."""
    h, w = matrice_calpestabile.shape
    raggio_penalità = dimensione_blocco * fattore_scalabilità
    distanza = distance_transform_edt(matrice_calpestabile)
    
    penalità = np.where(distanza < raggio_penalità, (raggio_penalità - distanza) ** 2, 0)
    penalità = 10 * penalità / penalità.max() if penalità.max() > 0 else penalità
    return penalità

def calcola_matrice_incentivi(matrice_calpestabile, punti_preferibili, dimensione_blocco, 
                            raggio_incentivo=2, bonus_incentivo=5):
    """Calcola gli incentivi per i percorsi preferibili."""
    h, w = matrice_calpestabile.shape
    matrice_incentivi = np.zeros((h, w))
    
    if not punti_preferibili:
        return matrice_incentivi
    
    print(f" Calcolo incentivi per {len(punti_preferibili)} punti preferibili...")
    
    # Converti i punti pixel in coordinate blocco
    punti_blocco = []
    for x, y in punti_preferibili:
        r, c = pixel_a_blocco(x, y, dimensione_blocco)
        if 0 <= r < h and 0 <= c < w:
            punti_blocco.append((r, c))
    
    print(f"   Punti blocco validi: {len(punti_blocco)}")
    
    # Usa meshgrid per calcolo vettorizzato
    rows, cols = np.meshgrid(range(h), range(w), indexing='ij')
    
    # Applica incentivi per ogni punto
    for i, (riga_p, colonna_p) in enumerate(punti_blocco):
        if i % 10 == 0:
            print(f"   Processando punto {i+1}/{len(punti_blocco)}")
        
        distanze = np.sqrt((rows - riga_p)**2 + (cols - colonna_p)**2)
        mask = (distanze <= raggio_incentivo) & (matrice_calpestabile == 1)
        incentivo = bonus_incentivo * (1 - distanze / raggio_incentivo)
        
        matrice_incentivi = np.where(mask & (incentivo > matrice_incentivi), 
                                   incentivo, matrice_incentivi)
    
    # Crea incentivi lungo i segmenti
    print("   Collegamento segmenti...")
    for i in range(len(punti_blocco) - 1):
        r1, c1 = punti_blocco[i]
        r2, c2 = punti_blocco[i + 1]
        
        steps = max(abs(r2 - r1), abs(c2 - c1))
        if steps > 0:
            for step in range(0, steps + 1, 2):
                t = step / steps
                r_interp = int(r1 + t * (r2 - r1))
                c_interp = int(c1 + t * (c2 - c1))
                
                if 0 <= r_interp < h and 0 <= c_interp < w:
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            nr, nc = r_interp + dr, c_interp + dc
                            if (0 <= nr < h and 0 <= nc < w and 
                                matrice_calpestabile[nr, nc]):
                                matrice_incentivi[nr, nc] = max(
                                    matrice_incentivi[nr, nc], bonus_incentivo * 0.5)
    
    print(f" Matrice incentivi calcolata. Max incentivo: {matrice_incentivi.max():.2f}")
    return matrice_incentivi

def blocchi_area_circolare(blocco_centro, raggio_blocchi, h, w):
    """Calcola i blocchi in un'area circolare."""
    r0, c0 = blocco_centro
    blocchi = set()
    for dr in range(-raggio_blocchi, raggio_blocchi + 1):
        for dc in range(-raggio_blocchi, raggio_blocchi + 1):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < h and 0 <= c < w and np.hypot(dr, dc) <= raggio_blocchi:
                blocchi.add((r, c))
    return blocchi

def seleziona_zone_pericolo(percorso_immagine, dimensione_blocco, matrice_calpestabile):
    """Interfaccia per selezionare zone di pericolo cliccando sull'immagine."""
    print("Seleziona zone di pericolo (click sinistro). Tasto destro per terminare.")
    immagine = np.array(Image.open(percorso_immagine))
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(immagine)
    
    h, w = matrice_calpestabile.shape
    zone_pericolo_blocchi = set()
    
    def on_click(event):
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            blocco = pixel_a_blocco(x, y, dimensione_blocco)
            
            # Area circolare di pericolo
            raggio_blocchi = 3
            blocchi_area = blocchi_area_circolare(blocco, raggio_blocchi, h, w)
            blocchi_area = {b for b in blocchi_area if matrice_calpestabile[b] == 1}
            
            zone_pericolo_blocchi.update(blocchi_area)
            
            # Visualizza l'area selezionata
            for b in blocchi_area:
                x_plot, y_plot = blocco_a_pixel(b[0], b[1], dimensione_blocco)
                ax.plot(x_plot, y_plot, 'rx', markersize=3)
            
            plt.draw()
            print(f"Zona pericolo: pixel ({x}, {y}), {len(blocchi_area)} blocchi aggiunti")
            
        elif event.button == 3:  # Tasto destro per terminare
            plt.close()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.title("Click sinistro: aggiungi zona pericolo | Tasto destro: termina")
    plt.show()
    
    return zone_pericolo_blocchi

def prendi_punto_da_click(percorso_immagine, messaggio="Clicca su START"):
    """Permette all'utente di cliccare per selezionare un punto."""
    immagine = np.array(Image.open(percorso_immagine))
    coordinate = []
    
    def al_click(evento):
        if evento.xdata and evento.ydata:
            coordinate.append((int(evento.xdata), int(evento.ydata)))
            plt.plot(evento.xdata, evento.ydata, 'go', markersize=10)
            plt.draw()
            plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(immagine)
    fig.canvas.mpl_connect('button_press_event', al_click)
    print(f"{messaggio} (un click).")
    plt.show()
    
    return coordinate[0] if coordinate else None

def pixel_a_blocco(x, y, dimensione_blocco):
    """Converti coordinate pixel in coordinate blocco."""
    return y // dimensione_blocco, x // dimensione_blocco

def blocco_a_pixel(riga, colonna, dimensione_blocco):
    """Converti coordinate blocco nel centro in pixel."""
    return colonna * dimensione_blocco + dimensione_blocco // 2, riga * dimensione_blocco + dimensione_blocco // 2

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

def a_star(matrice_calpestabile, inizio, obiettivo, matrice_penalità=None, 
          matrice_incentivi=None, zone_pericolo=None, max_iterazioni=100000):
    """Algoritmo A* ottimizzato con supporto per penalità, incentivi e zone di pericolo."""
    h, w = matrice_calpestabile.shape
    da_esplorare = [(0, inizio)]
    esplorati = set()
    da_dove = {}
    costo_g = {inizio: 0}
    costo_f = {inizio: euristica(inizio, obiettivo)}
    
    # Se ci sono zone di pericolo, crea una matrice modificata
    matrice_navigabile = np.copy(matrice_calpestabile)
    if zone_pericolo:
        zone_pericolo_set = set(tuple(z) for z in zone_pericolo)
        for blocco in zone_pericolo_set:
            if blocco != inizio and blocco != obiettivo:  # Non bloccare start/goal
                matrice_navigabile[blocco] = 0
    
    iterazioni = 0
    
    def vicini(pos):
        r, c = pos
        direzioni = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in direzioni:
            nr, nc = r + dr, c + dc
            if (0 <= nr < h and 0 <= nc < w and 
                matrice_navigabile[nr, nc] and (nr, nc) not in esplorati):
                yield (nr, nc)
    
    print(f"   Partenza: {inizio}, Obiettivo: {obiettivo}")
    print(f"   Distanza euclidea: {euristica(inizio, obiettivo):.1f}")
    
    while da_esplorare and iterazioni < max_iterazioni:
        if iterazioni % 10000 == 0 and iterazioni > 0:
            print(f"   Iterazione {iterazioni}, nodi da esplorare: {len(da_esplorare)}")
        
        _, attuale = heapq.heappop(da_esplorare)
        
        if attuale in esplorati:
            continue
        
        esplorati.add(attuale)
        
        if attuale == obiettivo:
            print(f"Percorso trovato in {iterazioni} iterazioni!")
            return ricostruisci_percorso(da_dove, attuale)
        
        for vicino in vicini(attuale):
            # Calcolo costo base
            dr, dc = vicino[0] - attuale[0], vicino[1] - attuale[1]
            costo_movimento = 1.0 if (dr == 0 or dc == 0) else 1.414
            
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
        print(f"Nessun percorso trovato dopo {iterazioni} iterazioni")
    
    return []

def seleziona_uscite(id_mappa, dimensione_blocco, matrice_calpestabile, blocco_inizio):
    """Seleziona l'uscita più vicina al punto di partenza."""
    uscita_file = os.path.join(CARTELLA_DATI, f"uscite-{id_mappa}.json")
    with open(uscita_file, "r") as f:
        uscite_pixel = json.load(f)
    
    uscite_blocco = [pixel_a_blocco(x, y, dimensione_blocco) for x, y in uscite_pixel]
    h, w = matrice_calpestabile.shape
    uscite_valide = [u for u in uscite_blocco 
                    if 0 <= u[0] < h and 0 <= u[1] < w and matrice_calpestabile[u] == 1]
    
    if not uscite_valide:
        raise ValueError("Nessuna uscita valida trovata nella mappa.")
    
    # Seleziona l'uscita più vicina al punto di partenza
    return min(uscite_valide, key=lambda u: euristica(blocco_inizio, u))

def trova_percorso_completo(matrice_calpestabile, blocco_inizio, uscita, 
                          matrice_penalità, matrice_incentivi, zone_pericolo):
    """Trova il percorso considerando tutte le opzioni: sicuro -> con pericoli -> fallback."""
    zone_pericolo_set = set(tuple(z) for z in zone_pericolo) if zone_pericolo else set()
    
    # Rimuovi start e goal dalle zone di pericolo se presenti
    zone_pericolo_sicure = zone_pericolo_set.copy()
    if blocco_inizio in zone_pericolo_sicure:
        zone_pericolo_sicure.remove(blocco_inizio)
    if uscita in zone_pericolo_sicure:
        zone_pericolo_sicure.remove(uscita)
    
    print(" Tentativo 1: Percorso sicuro (evitando zone di pericolo)...")
    percorso = a_star(matrice_calpestabile, blocco_inizio, uscita, 
                     matrice_penalità, matrice_incentivi, zone_pericolo_sicure)
    
    if percorso:
        print("Trovato percorso sicuro!")
        return percorso, True
    
    print("Tentativo 2: Percorso attraversando zone di pericolo...")
    percorso = a_star(matrice_calpestabile, blocco_inizio, uscita, 
                     matrice_penalità, matrice_incentivi)
    
    if percorso:
        print(" Trovato percorso che attraversa zone di pericolo!")
        return percorso, False
    
    print(" Nessun percorso trovato!")
    return None, False

def mostra_percorso_completo(id_mappa, percorso, matrice_penalità, matrice_incentivi, 
                            dimensione_blocco, punti_preferibili, zone_pericolo, è_sicuro):
    """Visualizza il risultato finale con stile elegante e minimalista."""
    img = np.array(Image.open(os.path.join(CARTELLA_DATI, f"{id_mappa}.png")))
    
    plt.figure(figsize=(12, 10))
    plt.imshow(img)

    # Calcolo della heatmap normalizzata
    mappa_costi = matrice_penalità - matrice_incentivi
    if mappa_costi.max() > mappa_costi.min():
        mappa_costi_norm = (mappa_costi - mappa_costi.min()) / (mappa_costi.max() - mappa_costi.min())
    else:
        mappa_costi_norm = mappa_costi

    costi_img = np.kron(mappa_costi_norm, np.ones((dimensione_blocco, dimensione_blocco)))
    costi_img = costi_img[:img.shape[0], :img.shape[1]]
    plt.imshow(costi_img, cmap="RdYlGn_r", alpha=0.7, vmin=0, vmax=1)

    ha_zone_pericolo = False
    if zone_pericolo:
        h, w = matrice_penalità.shape
        maschera_pericolo = np.zeros((h, w), dtype=bool)
        for r, c in zone_pericolo:
            if 0 <= r < h and 0 <= c < w:
                maschera_pericolo[r, c] = True

        if np.any(maschera_pericolo):
            ha_zone_pericolo = True
            pericolo_img = np.kron(maschera_pericolo.astype(float), np.ones((dimensione_blocco, dimensione_blocco)))
            pericolo_img = pericolo_img[:img.shape[0], :img.shape[1]]
            plt.imshow(pericolo_img, cmap="Reds", alpha=0.3)

    if punti_preferibili:
        pref_x, pref_y = zip(*punti_preferibili)
        plt.plot(pref_x, pref_y, color="cyan", linewidth=3, alpha=0.7, 
                 label="Percorso Preferibile", marker='o', markersize=4)

    if percorso:
        x_percorso = []
        y_percorso = []
        for r, c in percorso:
            x_pixel = c * dimensione_blocco + dimensione_blocco // 2
            y_pixel = r * dimensione_blocco + dimensione_blocco // 2
            x_percorso.append(x_pixel)
            y_percorso.append(y_pixel)

        colore_percorso = "lime" if è_sicuro else "orange"
        tipo_percorso_label = f"Percorso {'Sicuro' if è_sicuro else 'Con Rischi'}"
        plt.plot(x_percorso, y_percorso, color=colore_percorso, linewidth=4, 
                 label=tipo_percorso_label, alpha=0.7)

        plt.scatter(x_percorso[0], y_percorso[0], color="green", s=200, 
                    label="Start", zorder=5, edgecolor='white', linewidth=2, marker='o')
        plt.scatter(x_percorso[-1], y_percorso[-1], color="blue", s=200, 
                    label="Goal", zorder=5, edgecolor='white', linewidth=2, marker='o')

        if zone_pericolo and not è_sicuro:
            zone_pericolo_set = set(tuple(z) for z in zone_pericolo)
            punti_pericolo = [blocco for blocco in percorso if tuple(blocco) in zone_pericolo_set]
            if punti_pericolo:
                x_danger = []
                y_danger = []
                for r, c in punti_pericolo:
                    x_pixel = c * dimensione_blocco + dimensione_blocco // 2
                    y_pixel = r * dimensione_blocco + dimensione_blocco // 2
                    x_danger.append(x_pixel)
                    y_danger.append(y_pixel)
                plt.scatter(x_danger, y_danger, color="red", s=120, marker="X", 
                            label="Attraversamento Pericolo", zorder=6, 
                            edgecolor='white', linewidth=1)

    plt.axis("off")
    if ha_zone_pericolo:
        plt.scatter([], [], color="red", s=100, alpha=0.6, 
                    label="Zone di Pericolo", marker='s')

    legend = plt.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                        fancybox=True, shadow=True, borderpad=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')

    tipo_percorso = "Sicuro" if è_sicuro else "Con Rischi"
    plt.title(f"Mappa {id_mappa} - Percorso A* ({tipo_percorso})", 
              fontsize=14, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Output testuale semplificato
    print("\n" + "="*50)
    print("RISULTATI DEL PERCORSO")
    print("="*50)
    
    if percorso:
        print(f"Lunghezza percorso: {len(percorso)} blocchi")
        print(f"Tipo: {'Sicuro (evita pericoli)' if è_sicuro else 'Con rischi (attraversa pericoli)'}")
        
        # Distanza euclidea
        distanza_euclidea = sum(np.hypot(percorso[i+1][0] - percorso[i][0], 
                                        percorso[i+1][1] - percorso[i][1]) 
                               for i in range(len(percorso)-1))
        print(f"Distanza euclidea: {distanza_euclidea:.2f} blocchi")
        
        # Analisi zone di pericolo
        if zone_pericolo:
            zone_pericolo_set = set(tuple(z) for z in zone_pericolo)
            zone_attraversate = [b for b in percorso if tuple(b) in zone_pericolo_set]
            print(f"Zone di pericolo attraversate: {len(zone_attraversate)}/{len(zone_pericolo)}")
            if zone_attraversate:
                print("  Blocchi di pericolo nel percorso:", zone_attraversate[:5], 
                      "..." if len(zone_attraversate) > 5 else "")
    else:
        print("Nessun percorso trovato")
    
    print("="*50)

def main():
    """Funzione principale che orchestriona tutto il processo."""
    mappe_disponibili = [703368, 703326, 703323, 703372, 703381, 703428]
    
    print(" PATHFINDING INTEGRATO - A* con Preferenze e Pericoli")
    print("="*60)
    print(f"Mappe disponibili: {mappe_disponibili}")
    
    # Selezione mappa
    try:
        id_mappa = int(input("Inserisci ID della mappa: "))
        if id_mappa not in mappe_disponibili:
            raise ValueError("ID mappa non valido.")
    except ValueError as e:
        print(f"Errore: {e}")
        return
    
    print(f"\nInizializzazione mappa {id_mappa}...")
    dimensione_blocco = 5
    
    # Caricamento dati
    mappa = carica_mappa(id_mappa)
    matrice_calpestabile = crea_matrice_calpestabile(mappa, dimensione_blocco)
    print(f"   Dimensioni matrice: {matrice_calpestabile.shape}")
    
    # Carica preferenze
    punti_preferibili = carica_preferenze(id_mappa)
    if punti_preferibili:
        print(f" Caricati {len(punti_preferibili)} punti del percorso preferibile.")
    else:
        print(" Nessun percorso preferibile trovato per questa mappa.")
    
    # Selezione punto di partenza
    percorso_immagine = os.path.join(CARTELLA_DATI, f"{id_mappa}.png")
    print("\n Selezione punto di partenza...")
    punto_click = prendi_punto_da_click(percorso_immagine, "Clicca su START")
    if not punto_click:
        print("Errore nella selezione del punto di partenza.")
        return
    
    blocco_inizio = pixel_a_blocco(*punto_click, dimensione_blocco)
    print(f"   Punto selezionato: pixel {punto_click}, blocco {blocco_inizio}")
    
    # Validazione punto di partenza
    h, w = matrice_calpestabile.shape
    r_start, c_start = blocco_inizio
    if (r_start < 0 or r_start >= h or c_start < 0 or c_start >= w or 
        matrice_calpestabile[r_start, c_start] == 0):
        print(f"ERRORE: Il punto di partenza non è calpestabile!")
        return
    
    # Selezione zone di pericolo
    print("\n Selezione zone di pericolo...")
    zone_pericolo = seleziona_zone_pericolo(percorso_immagine, dimensione_blocco, matrice_calpestabile)
    print(f"   {len(zone_pericolo)} blocchi di pericolo selezionati.")
    
    # Selezione uscita
    print("\n Selezione uscita ottimale...")
    uscita = seleziona_uscite(id_mappa, dimensione_blocco, matrice_calpestabile, blocco_inizio)
    print(f"   Uscita selezionata: {uscita}")
    
    # Calcolo matrici di costo
    print("\n Calcolo matrici di costo...")
    print("   Calcolando penalità...")
    matrice_penalità = calcola_matrice_penalità(matrice_calpestabile, dimensione_blocco)
    
    print("   Calcolando incentivi...")
    matrice_incentivi = calcola_matrice_incentivi(matrice_calpestabile, punti_preferibili, dimensione_blocco)
    
    # Ricerca percorso
    print("\n Ricerca percorso ottimale...")
    percorso, è_sicuro = trova_percorso_completo(
        matrice_calpestabile, blocco_inizio, uscita, 
        matrice_penalità, matrice_incentivi, zone_pericolo
    )
    
    # Visualizzazione risultati
    print("\nGenerazione visualizzazione...")
    mostra_percorso_completo(
        id_mappa, percorso, matrice_penalità, matrice_incentivi, 
        dimensione_blocco, punti_preferibili, zone_pericolo, è_sicuro
    )

main()