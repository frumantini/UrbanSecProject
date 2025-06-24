import os
import sys
import json
import heapq
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import distance_transform_edt
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd

# Configurazione percorsi
CARTELLA_BASE = os.path.dirname(__file__)
CARTELLA_DATI = os.path.join(CARTELLA_BASE, "data")
sys.path.append(CARTELLA_BASE)

from models import Map

@dataclass
class TestResult:
    """Classe per memorizzare i risultati di un singolo test."""
    algorithm: str
    map_id: int
    start_pos: Tuple[int, int]
    end_pos: Tuple[int, int]
    path_length: int
    total_cost: float
    execution_time: float
    path_found: bool
    nodes_explored: int

@dataclass
class ComparisonResult:
    """Classe per memorizzare il confronto tra due algoritmi."""
    map_id: int
    test_point: int
    astar_result: TestResult
    dijkstra_result: TestResult
    winner_time: str
    winner_cost: str
    winner_path: str
    time_difference: float
    cost_difference: float

class PathfindingTester:
    def __init__(self):
        self.results = []
        self.comparison_results = []
    
    def carica_mappa(self, id_mappa):
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

    def crea_matrice_calpestabile(self, mappa, dimensione_blocco):
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

    def calcola_matrice_penalità(self, matrice_calpestabile, dimensione_blocco, fattore_scalabilità=1.5):
        """Calcola penalità per blocchi vicini ai muri."""
        h, w = matrice_calpestabile.shape
        raggio_penalità = dimensione_blocco * fattore_scalabilità
        distanza = distance_transform_edt(matrice_calpestabile)
        penalità = np.where(distanza < raggio_penalità, (raggio_penalità - distanza) ** 2, 0)
        penalità = 10 * penalità / penalità.max() if penalità.max() > 0 else penalità
        return penalità

    def genera_punti_test_casuali(self, matrice_calpestabile, num_punti=10):
        """Genera punti di test casuali su aree calpestabili."""
        h, w = matrice_calpestabile.shape
        punti_validi = [(r, c) for r in range(h) for c in range(w) if matrice_calpestabile[r, c] == 1]
        
        if len(punti_validi) < 2:
            raise ValueError("Non ci sono abbastanza punti calpestabili per i test")
        
        punti_test = []
        for _ in range(num_punti):
            start, end = random.sample(punti_validi, 2)
            # Assicuriamoci che i punti siano abbastanza distanti
            if self.euristica(start, end) > 10:  # Distanza minima
                punti_test.append((start, end))
        
        return punti_test

    def carica_uscite(self, id_mappa, dimensione_blocco, matrice_calpestabile):
        """Carica le uscite dalla mappa."""
        try:
            uscita_file = os.path.join(CARTELLA_DATI, f"uscite-{id_mappa}.json")
            with open(uscita_file, "r") as f:
                uscite_pixel = json.load(f)

            uscite_blocco = [self.pixel_a_blocco(x, y, dimensione_blocco) for x, y in uscite_pixel]
            uscite_valide = [usc for usc in uscite_blocco if matrice_calpestabile[usc] == 1]
            return uscite_valide
        except FileNotFoundError:
            return []

    def pixel_a_blocco(self, x, y, dimensione_blocco):
        """Converti coordinate pixel in coordinate blocco."""
        return y // dimensione_blocco, x // dimensione_blocco

    def euristica(self, a, b):
        """Stima della distanza tra due punti (distanza euclidea)."""
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def ricostruisci_percorso(self, da_dove, attuale):
        """Ricostruisce il percorso una volta raggiunto l'obiettivo."""
        percorso = [attuale]
        while attuale in da_dove:
            attuale = da_dove[attuale]
            percorso.append(attuale)
        return percorso[::-1]

    def a_star(self, matrice_calpestabile, inizio, obiettivo, matrice_penalità=None):
        """Algoritmo A* con conteggio nodi esplorati."""
        h, w = matrice_calpestabile.shape
        da_esplorare = [(0, inizio)]
        da_dove = {}
        costo_g = {inizio: 0}
        costo_f = {inizio: self.euristica(inizio, obiettivo)}
        nodi_esplorati = 0

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
            nodi_esplorati += 1
            
            if attuale == obiettivo:
                return self.ricostruisci_percorso(da_dove, attuale), nodi_esplorati

            for vicino in vicini(attuale):
                costo_movimento = np.hypot(vicino[0] - attuale[0], vicino[1] - attuale[1])
                penalità = matrice_penalità[vicino] if matrice_penalità is not None else 0
                tentativo_g = costo_g[attuale] + costo_movimento + penalità

                if tentativo_g < costo_g.get(vicino, float("inf")):
                    da_dove[vicino] = attuale
                    costo_g[vicino] = tentativo_g
                    costo_f[vicino] = tentativo_g + self.euristica(vicino, obiettivo)
                    heapq.heappush(da_esplorare, (costo_f[vicino], vicino))

        return [], nodi_esplorati

    def dijkstra(self, matrice_calpestabile, inizio, obiettivo, matrice_penalità=None):
        """Algoritmo Dijkstra con conteggio nodi esplorati."""
        h, w = matrice_calpestabile.shape
        da_esplorare = [(0, inizio)]
        da_dove = {}
        costo_g = {inizio: 0}
        nodi_esplorati = 0

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
            nodi_esplorati += 1
            
            if attuale == obiettivo:
                return self.ricostruisci_percorso(da_dove, attuale), nodi_esplorati

            for vicino in vicini(attuale):
                costo_movimento = np.hypot(vicino[0] - attuale[0], vicino[1] - attuale[1])
                penalità = matrice_penalità[vicino] if matrice_penalità is not None else 0
                tentativo_g = costo_g[attuale] + costo_movimento + penalità

                if tentativo_g < costo_g.get(vicino, float("inf")):
                    da_dove[vicino] = attuale
                    costo_g[vicino] = tentativo_g
                    heapq.heappush(da_esplorare, (tentativo_g, vicino))

        return [], nodi_esplorati

    def calcola_costo_percorso(self, percorso, matrice_penalità):
        """Calcola il costo totale di un percorso."""
        if len(percorso) < 2:
            return 0
        
        costo_totale = 0
        for i in range(1, len(percorso)):
            p1 = percorso[i - 1]
            p2 = percorso[i]
            distanza = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            penalità = matrice_penalità[p2] if matrice_penalità is not None else 0
            costo_totale += distanza + penalità
        
        return costo_totale

    def esegui_test_singolo(self, id_mappa, start, end, algoritmo, matrice_calpestabile, matrice_penalità):
        """Esegue un singolo test con un algoritmo specifico."""
        start_time = time.time()
        
        if algoritmo == "A*":
            percorso, nodi_esplorati = self.a_star(matrice_calpestabile, start, end, matrice_penalità)
        elif algoritmo == "Dijkstra":
            percorso, nodi_esplorati = self.dijkstra(matrice_calpestabile, start, end, matrice_penalità)
        else:
            raise ValueError(f"Algoritmo non supportato: {algoritmo}")
        
        execution_time = time.time() - start_time
        
        if percorso:
            path_length = len(percorso)
            total_cost = self.calcola_costo_percorso(percorso, matrice_penalità)
            path_found = True
        else:
            path_length = 0
            total_cost = float('inf')
            path_found = False
        
        return TestResult(
            algorithm=algoritmo,
            map_id=id_mappa,
            start_pos=start,
            end_pos=end,
            path_length=path_length,
            total_cost=total_cost,
            execution_time=execution_time,
            path_found=path_found,
            nodes_explored=nodi_esplorati
        )

    def esegui_test_completo(self, mappe_id, num_test_per_mappa=10, dimensione_blocco=5):
        """Esegue test completi su multiple mappe."""
        print("Inizio test automatizzato A* vs Dijkstra...")
        print("=" * 60)
        
        for id_mappa in mappe_id:
            print(f"\nTestando mappa {id_mappa}...")
            
            try:
                # Carica mappa e prepara dati
                mappa = self.carica_mappa(id_mappa)
                matrice_calpestabile = self.crea_matrice_calpestabile(mappa, dimensione_blocco)
                matrice_penalità = self.calcola_matrice_penalità(matrice_calpestabile, dimensione_blocco)
                
                # Genera punti di test
                punti_test = self.genera_punti_test_casuali(matrice_calpestabile, num_test_per_mappa)
                
                # Aggiungi test con le uscite se disponibili
                uscite = self.carica_uscite(id_mappa, dimensione_blocco, matrice_calpestabile)
                if uscite:
                    punti_validi = [(r, c) for r in range(matrice_calpestabile.shape[0]) 
                                   for c in range(matrice_calpestabile.shape[1]) 
                                   if matrice_calpestabile[r, c] == 1]
                    
                    # Aggiungi alcuni test verso le uscite
                    for i, uscita in enumerate(uscite[:3]):  # Massimo 3 uscite
                        if punti_validi:
                            start_random = random.choice(punti_validi)
                            punti_test.append((start_random, uscita))
                
                print(f"   Generati {len(punti_test)} punti di test")
                
                # Esegui test per ogni punto
                for i, (start, end) in enumerate(punti_test):
                    print(f"   Test {i+1}/{len(punti_test)}: {start} → {end}")
                    
                    # Test A*
                    astar_result = self.esegui_test_singolo(
                        id_mappa, start, end, "A*", matrice_calpestabile, matrice_penalità
                    )
                    
                    # Test Dijkstra
                    dijkstra_result = self.esegui_test_singolo(
                        id_mappa, start, end, "Dijkstra", matrice_calpestabile, matrice_penalità
                    )
                    
                    # Salva risultati
                    self.results.extend([astar_result, dijkstra_result])
                    
                    # Confronta risultati
                    if astar_result.path_found and dijkstra_result.path_found:
                        winner_time = "A*" if astar_result.execution_time < dijkstra_result.execution_time else "Dijkstra"
                        winner_cost = "A*" if astar_result.total_cost < dijkstra_result.total_cost else "Dijkstra"
                        winner_path = "A*" if astar_result.path_length < dijkstra_result.path_length else "Dijkstra"
                        
                        time_diff = abs(astar_result.execution_time - dijkstra_result.execution_time)
                        cost_diff = abs(astar_result.total_cost - dijkstra_result.total_cost)
                        
                        comparison = ComparisonResult(
                            map_id=id_mappa,
                            test_point=i+1,
                            astar_result=astar_result,
                            dijkstra_result=dijkstra_result,
                            winner_time=winner_time,
                            winner_cost=winner_cost,
                            winner_path=winner_path,
                            time_difference=time_diff,
                            cost_difference=cost_diff
                        )
                        
                        self.comparison_results.append(comparison)
                
                print(f"   Test completata!")
                
            except Exception as e:
                print(f"   Errore nella mappa {id_mappa}: {e}")
                continue
        
        print("\nTest completati!")
        self.genera_report()

    def genera_report(self):
        """Genera un report dei risultati."""
        if not self.comparison_results:
            print("Nessun risultato da analizzare!")
            return
        
        print("\n" + "="*80)
        print("REPORT  A* vs DIJKSTRA")
        print("="*80)
        
        # Statistiche generali
        astar_wins_time = sum(1 for r in self.comparison_results if r.winner_time == "A*")
        dijkstra_wins_time = sum(1 for r in self.comparison_results if r.winner_time == "Dijkstra")
        
        astar_wins_cost = sum(1 for r in self.comparison_results if r.winner_cost == "A*")
        dijkstra_wins_cost = sum(1 for r in self.comparison_results if r.winner_cost == "Dijkstra")
        
        # Calcola vittorie per nodi esplorati
        astar_wins_nodes = sum(1 for r in self.comparison_results 
                              if r.astar_result.nodes_explored < r.dijkstra_result.nodes_explored)
        dijkstra_wins_nodes = sum(1 for r in self.comparison_results 
                                 if r.dijkstra_result.nodes_explored < r.astar_result.nodes_explored)
        
        total_tests = len(self.comparison_results)
        
        print(f"\nSTATISTICHE GENERALI ({total_tests} test)")
        print("-" * 40)
        print(f"TEMPO DI ESECUZIONE:")
        print(f"   A* vince: {astar_wins_time}/{total_tests} ({astar_wins_time/total_tests*100:.1f}%)")
        print(f"   Dijkstra vince: {dijkstra_wins_time}/{total_tests} ({dijkstra_wins_time/total_tests*100:.1f}%)")
        
        print(f"\nCOSTO DEL PERCORSO:")
        print(f"   A* vince: {astar_wins_cost}/{total_tests} ({astar_wins_cost/total_tests*100:.1f}%)")
        print(f"   Dijkstra vince: {dijkstra_wins_cost}/{total_tests} ({dijkstra_wins_cost/total_tests*100:.1f}%)")
        
        print(f"\nEFFICIENZA (nodi esplorati):")
        print(f"   A* vince: {astar_wins_nodes}/{total_tests} ({astar_wins_nodes/total_tests*100:.1f}%)")
        print(f"   Dijkstra vince: {dijkstra_wins_nodes}/{total_tests} ({dijkstra_wins_nodes/total_tests*100:.1f}%)")
        
        # Tempi medi
        astar_times = [r.astar_result.execution_time for r in self.comparison_results if r.astar_result.path_found]
        dijkstra_times = [r.dijkstra_result.execution_time for r in self.comparison_results if r.dijkstra_result.path_found]
        
        if astar_times and dijkstra_times:
            print(f"\nTEMPI MEDI:")
            print(f"   A*: {np.mean(astar_times):.4f}s (±{np.std(astar_times):.4f}s)")
            print(f"   Dijkstra: {np.mean(dijkstra_times):.4f}s (±{np.std(dijkstra_times):.4f}s)")
        
        # Nodi esplorati
        astar_nodes = [r.astar_result.nodes_explored for r in self.comparison_results if r.astar_result.path_found]
        dijkstra_nodes = [r.dijkstra_result.nodes_explored for r in self.comparison_results if r.dijkstra_result.path_found]
        
        if astar_nodes and dijkstra_nodes:
            print(f"\nNODI ESPLORATI MEDI:")
            print(f"   A*: {np.mean(astar_nodes):.0f} (±{np.std(astar_nodes):.0f})")
            print(f"   Dijkstra: {np.mean(dijkstra_nodes):.0f} (±{np.std(dijkstra_nodes):.0f})")
            
            efficiency_ratio = np.mean(astar_nodes) / np.mean(dijkstra_nodes)
            if efficiency_ratio < 1:
                print(f"   A* esplora {(1-efficiency_ratio)*100:.1f}% nodi in meno di Dijkstra")
            else:
                print(f"   Dijkstra esplora {((1/efficiency_ratio)-1)*100:.1f}% nodi in meno di A*")
        
        # Analisi per mappa
        print(f"\nANALISI PER MAPPA:")
        print("-" * 40)
        mappe_unique = set(r.map_id for r in self.comparison_results)
        
        for map_id in sorted(mappe_unique):
            map_results = [r for r in self.comparison_results if r.map_id == map_id]
            map_astar_wins_time = sum(1 for r in map_results if r.winner_time == "A*")
            map_astar_wins_nodes = sum(1 for r in map_results 
                                      if r.astar_result.nodes_explored < r.dijkstra_result.nodes_explored)
            map_total = len(map_results)
            
            print(f"   Mappa {map_id}: A* vince {map_astar_wins_time}/{map_total} in tempo, {map_astar_wins_nodes}/{map_total} in efficienza")
        
        # Calcolo punteggio complessivo considerando anche i nodi esplorati
        print(f"\nCONCLUSIONI:")
        print("-" * 40)
        
        # Punteggio pesato: tempo 33%, costo 33%, efficienza 33%
        astar_score = (astar_wins_time * 0.33 + astar_wins_cost * 0.33 + astar_wins_nodes * 0.33) / total_tests
        dijkstra_score = (dijkstra_wins_time * 0.33 + dijkstra_wins_cost * 0.33 + dijkstra_wins_nodes * 0.33) / total_tests
        
        print(f"Punteggio A*: {astar_score:.3f} (tempo: 33%, costo: 33%, efficienza: 33%)")
        print(f"Punteggio Dijkstra: {dijkstra_score:.3f}")
        
        if astar_wins_time > dijkstra_wins_time:
            print("A* è generalmente PIU' VELOCE di Dijkstra")
        elif dijkstra_wins_time > astar_wins_time:
            print("Dijkstra è generalmente PIU' VELOCE di A*")
        else:
            print("A* e Dijkstra hanno prestazioni SIMILI in velocità")
        
        if astar_wins_cost > dijkstra_wins_cost:
            print("A* trova percorsi con COSTO MINORE")
        elif dijkstra_wins_cost > astar_wins_cost:
            print("Dijkstra trova percorsi con COSTO MINORE")
        else:
            print("A* e Dijkstra trovano percorsi con COSTO SIMILE")
            
        if astar_wins_nodes > dijkstra_wins_nodes:
            print("A* è PIU' EFFICIENTE (esplora meno nodi)")
        elif dijkstra_wins_nodes > astar_wins_nodes:
            print("Dijkstra è PIU' EFFICIENTE (esplora meno nodi)")
        else:
            print("A* e Dijkstra hanno EFFICIENZA SIMILE")
        
        # Raccomandazione finale basata su punteggio complessivo
        print(f"\nRACCOMANDAZIONE:")
        if astar_score > dijkstra_score:
            margin = (astar_score - dijkstra_score) * 100
            print(f"A* è l'algoritmo RACCOMANDATO (vantaggio: {margin:.1f}%)")
            print("Motivi: migliori prestazioni complessive considerando velocità, costo ed efficienza")
        elif dijkstra_score > astar_score:
            margin = (dijkstra_score - astar_score) * 100
            print(f"Dijkstra è l'algoritmo RACCOMANDATO (vantaggio: {margin:.1f}%)")
            print("Motivi: migliori prestazioni complessive considerando velocità, costo ed efficienza")
        else:
            print("Entrambi gli algoritmi sono equivalenti per questo scenario")
            print("Scegli in base alle priorità specifiche del tuo caso d'uso")
        
        print("="*80)

    def salva_risultati_csv(self, filename="pathfinding_results.csv"):
        """Salva i risultati in un file CSV per analisi future."""
        if not self.results:
            print("Nessun risultato da salvare!")
            return
        
        # Converte i risultati in un DataFrame
        data = []
        for result in self.results:
            data.append({
                'algorithm': result.algorithm,
                'map_id': result.map_id,
                'start_pos': str(result.start_pos),
                'end_pos': str(result.end_pos),
                'path_length': result.path_length,
                'total_cost': result.total_cost,
                'execution_time': result.execution_time,
                'path_found': result.path_found,
                'nodes_explored': result.nodes_explored
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Risultati salvati in {filename}")

def main():
    """Funzione principale per eseguire i test."""
    # Lista delle mappe disponibili (dalla tua configurazione)
    mappe_disponibili = [703368, 703326, 703323, 703372, 703381, 549362, 703428]
    
    print("TEST AUTOMATIZZATO A* vs DIJKSTRA")
    print("Questo script confronterà le prestazioni dei due algoritmi")
    print(f"Mappe disponibili: {mappe_disponibili}")
    
    # Chiedi all'utente quali mappe testare
    print("\nScegli le mappe da testare:")
    print("1. Tutte le mappe")
    print("2. Seleziona mappe specifiche")
    print("3. Test rapido (solo 2 mappe)")
    
    scelta = input("Inserisci la tua scelta (1-3): ").strip()
    
    if scelta == "1":
        mappe_da_testare = mappe_disponibili
    elif scelta == "2":
        mappe_input = input(f"Inserisci gli ID delle mappe separati da virgola: ").strip()
        try:
            mappe_da_testare = [int(x.strip()) for x in mappe_input.split(",")]
            mappe_da_testare = [m for m in mappe_da_testare if m in mappe_disponibili]
        except ValueError:
            print("Input non valido, uso tutte le mappe")
            mappe_da_testare = mappe_disponibili
    else:  # Test rapido
        mappe_da_testare = mappe_disponibili[:2]
    
    num_test = int(input("Numero di test per mappa (consigliato: 5-10): ") or "5")
    
    # Crea il tester ed esegui i test
    tester = PathfindingTester()
    tester.esegui_test_completo(mappe_da_testare, num_test)
    
    # Salva i risultati
    if input("\nVuoi salvare i risultati in CSV? (y/n): ").lower().startswith('y'):
        filename = input("Nome file (default: pathfinding_results.csv): ").strip() or "pathfinding_results.csv"
        tester.salva_risultati_csv(filename)

main()