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
class HeuristicTestResult:
    """Classe per memorizzare i risultati di un singolo test con euristica."""
    heuristic: str
    map_id: int
    start_pos: Tuple[int, int]
    end_pos: Tuple[int, int]
    path_length: int
    total_cost: float
    execution_time: float
    path_found: bool
    nodes_explored: int
    heuristic_calls: int

@dataclass
class HeuristicComparisonResult:
    """Classe per memorizzare il confronto tra euristiche."""
    map_id: int
    test_point: int
    euclidean_result: HeuristicTestResult
    manhattan_result: HeuristicTestResult
    chebyshev_result: HeuristicTestResult
    winner_time: str
    winner_cost: str
    winner_efficiency: str
    time_differences: dict
    cost_differences: dict
    efficiency_differences: dict

class HeuristicTester:
    def __init__(self):
        self.results = []
        self.comparison_results = []
        self.heuristic_call_count = 0
    
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
            if self.euristica_euclidea(start, end) > 10:  # Distanza minima
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

    # === EURISTICHE ===
    def euristica_euclidea(self, a, b):
        """Distanza euclidea (diagonale libera)."""
        self.heuristic_call_count += 1
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def euristica_manhattan(self, a, b):
        """Distanza Manhattan (solo movimenti ortogonali)."""
        self.heuristic_call_count += 1
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def euristica_chebyshev(self, a, b):
        """Distanza Chebyshev (massimo tra differenze coordinate)."""
        self.heuristic_call_count += 1
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def ricostruisci_percorso(self, da_dove, attuale):
        """Ricostruisce il percorso una volta raggiunto l'obiettivo."""
        percorso = [attuale]
        while attuale in da_dove:
            attuale = da_dove[attuale]
            percorso.append(attuale)
        return percorso[::-1]

    def a_star_con_euristica(self, matrice_calpestabile, inizio, obiettivo, euristica_func, matrice_penalità=None):
        """Algoritmo A* con euristica personalizzabile e conteggio nodi esplorati."""
        self.heuristic_call_count = 0  # Reset contatore
        h, w = matrice_calpestabile.shape
        da_esplorare = [(0, inizio)]
        da_dove = {}
        costo_g = {inizio: 0}
        costo_f = {inizio: euristica_func(inizio, obiettivo)}
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
                return self.ricostruisci_percorso(da_dove, attuale), nodi_esplorati, self.heuristic_call_count

            for vicino in vicini(attuale):
                costo_movimento = np.hypot(vicino[0] - attuale[0], vicino[1] - attuale[1])
                penalità = matrice_penalità[vicino] if matrice_penalità is not None else 0
                tentativo_g = costo_g[attuale] + costo_movimento + penalità

                if tentativo_g < costo_g.get(vicino, float("inf")):
                    da_dove[vicino] = attuale
                    costo_g[vicino] = tentativo_g
                    costo_f[vicino] = tentativo_g + euristica_func(vicino, obiettivo)
                    heapq.heappush(da_esplorare, (costo_f[vicino], vicino))

        return [], nodi_esplorati, self.heuristic_call_count

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

    def esegui_test_euristica(self, id_mappa, start, end, euristica_name, euristica_func, matrice_calpestabile, matrice_penalità):
        """Esegue un singolo test con una specifica euristica."""
        start_time = time.time()
        
        percorso, nodi_esplorati, heuristic_calls = self.a_star_con_euristica(
            matrice_calpestabile, start, end, euristica_func, matrice_penalità
        )
        
        execution_time = time.time() - start_time
        
        if percorso:
            path_length = len(percorso)
            total_cost = self.calcola_costo_percorso(percorso, matrice_penalità)
            path_found = True
        else:
            path_length = 0
            total_cost = float('inf')
            path_found = False
        
        return HeuristicTestResult(
            heuristic=euristica_name,
            map_id=id_mappa,
            start_pos=start,
            end_pos=end,
            path_length=path_length,
            total_cost=total_cost,
            execution_time=execution_time,
            path_found=path_found,
            nodes_explored=nodi_esplorati,
            heuristic_calls=heuristic_calls
        )

    def esegui_test_completo(self, mappe_id, num_test_per_mappa=10, dimensione_blocco=5):
        """Esegue test completi confrontando le tre euristiche."""
        print("Inizio test automatizzato EURISTICHE A*...")
        print("Confronto: Euclidea vs Manhattan vs Chebyshev")
        print("=" * 60)
        
        euristiche = {
            "Euclidea": self.euristica_euclidea,
            "Manhattan": self.euristica_manhattan,
            "Chebyshev": self.euristica_chebyshev
        }
        
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
                    
                    # Test con tutte le euristiche
                    risultati_test = {}
                    for nome_euristica, func_euristica in euristiche.items():
                        risultato = self.esegui_test_euristica(
                            id_mappa, start, end, nome_euristica, func_euristica, 
                            matrice_calpestabile, matrice_penalità
                        )
                        risultati_test[nome_euristica] = risultato
                        self.results.append(risultato)
                    
                    # Confronta risultati se tutti hanno trovato un percorso
                    if all(r.path_found for r in risultati_test.values()):
                        # Determina i vincitori
                        winner_time = min(risultati_test.keys(), 
                                        key=lambda k: risultati_test[k].execution_time)
                        winner_cost = min(risultati_test.keys(), 
                                        key=lambda k: risultati_test[k].total_cost)
                        winner_efficiency = min(risultati_test.keys(), 
                                              key=lambda k: risultati_test[k].nodes_explored)
                        
                        # Calcola differenze
                        time_diffs = {}
                        cost_diffs = {}
                        efficiency_diffs = {}
                        
                        euclidean_result = risultati_test["Euclidea"]
                        for nome, risultato in risultati_test.items():
                            if nome != "Euclidea":
                                time_diffs[nome] = risultato.execution_time - euclidean_result.execution_time
                                cost_diffs[nome] = risultato.total_cost - euclidean_result.total_cost
                                efficiency_diffs[nome] = risultato.nodes_explored - euclidean_result.nodes_explored
                        
                        comparison = HeuristicComparisonResult(
                            map_id=id_mappa,
                            test_point=i+1,
                            euclidean_result=risultati_test["Euclidea"],
                            manhattan_result=risultati_test["Manhattan"],
                            chebyshev_result=risultati_test["Chebyshev"],
                            winner_time=winner_time,
                            winner_cost=winner_cost,
                            winner_efficiency=winner_efficiency,
                            time_differences=time_diffs,
                            cost_differences=cost_diffs,
                            efficiency_differences=efficiency_diffs
                        )
                        
                        self.comparison_results.append(comparison)
                
                print(f"   Test completata!")
                
            except Exception as e:
                print(f"   Errore nella mappa {id_mappa}: {e}")
                continue
        
        print("\nTest completati!")
        self.genera_report()

    def genera_report(self):
        """Genera un report dettagliato dei risultati."""
        if not self.comparison_results:
            print("Nessun risultato da analizzare!")
            return
        
        print("\n" + "="*80)
        print("REPORT CONFRONTO EURISTICHE A*")
        print("="*80)
        
        euristiche = ["Euclidea", "Manhattan", "Chebyshev"]
        total_tests = len(self.comparison_results)
        
        # Conteggio vittorie
        wins_time = {e: 0 for e in euristiche}
        wins_cost = {e: 0 for e in euristiche}
        wins_efficiency = {e: 0 for e in euristiche}
        
        for result in self.comparison_results:
            wins_time[result.winner_time] += 1
            wins_cost[result.winner_cost] += 1
            wins_efficiency[result.winner_efficiency] += 1
        
        print(f"\nSTATISTICHE GENERALI ({total_tests} test)")
        print("-" * 40)
        
        print("TEMPO DI ESECUZIONE:")
        for euristica in euristiche:
            pct = wins_time[euristica] / total_tests * 100
            print(f"   {euristica}: {wins_time[euristica]}/{total_tests} vittorie ({pct:.1f}%)")
        
        print("\nCOSTO DEL PERCORSO:")
        for euristica in euristiche:
            pct = wins_cost[euristica] / total_tests * 100
            print(f"   {euristica}: {wins_cost[euristica]}/{total_tests} vittorie ({pct:.1f}%)")
        
        print("\nEFFICIENZA (nodi esplorati):")
        for euristica in euristiche:
            pct = wins_efficiency[euristica] / total_tests * 100
            print(f"   {euristica}: {wins_efficiency[euristica]}/{total_tests} vittorie ({pct:.1f}%)")
        
        # Statistiche dettagliate
        stats_per_euristica = {}
        for euristica in euristiche:
            risultati = [r for r in self.results if r.heuristic == euristica and r.path_found]
            if risultati:
                stats_per_euristica[euristica] = {
                    'tempo_medio': np.mean([r.execution_time for r in risultati]),
                    'tempo_std': np.std([r.execution_time for r in risultati]),
                    'costo_medio': np.mean([r.total_cost for r in risultati]),
                    'costo_std': np.std([r.total_cost for r in risultati]),
                    'nodi_medio': np.mean([r.nodes_explored for r in risultati]),
                    'nodi_std': np.std([r.nodes_explored for r in risultati]),
                    'chiamate_euristica_media': np.mean([r.heuristic_calls for r in risultati]),
                    'chiamate_euristica_std': np.std([r.heuristic_calls for r in risultati])
                }
        
        print(f"\nSTATISTICHE DETTAGLIATE:")
        print("-" * 40)
        for euristica, stats in stats_per_euristica.items():
            print(f"\n{euristica.upper()}:")
            print(f"   Tempo: {stats['tempo_medio']:.4f}s (±{stats['tempo_std']:.4f}s)")
            print(f"   Costo: {stats['costo_medio']:.2f} (±{stats['costo_std']:.2f})")
            print(f"   Nodi esplorati: {stats['nodi_medio']:.0f} (±{stats['nodi_std']:.0f})")
            print(f"   Chiamate euristica: {stats['chiamate_euristica_media']:.0f} (±{stats['chiamate_euristica_std']:.0f})")
        
        # Analisi efficienza relativa
        print(f"\nANALISI EFFICIENZA RELATIVA (vs Euclidea):")
        print("-" * 40)
        
        euclidean_stats = stats_per_euristica.get("Euclidea")
        if euclidean_stats:
            for euristica, stats in stats_per_euristica.items():
                if euristica != "Euclidea":
                    tempo_diff_pct = ((stats['tempo_medio'] - euclidean_stats['tempo_medio']) / euclidean_stats['tempo_medio']) * 100
                    nodi_diff_pct = ((stats['nodi_medio'] - euclidean_stats['nodi_medio']) / euclidean_stats['nodi_medio']) * 100
                    
                    print(f"{euristica}:")
                    print(f"   Tempo: {tempo_diff_pct:+.1f}% rispetto a Euclidea")
                    print(f"   Nodi esplorati: {nodi_diff_pct:+.1f}% rispetto a Euclidea")
        
        # Analisi per mappa
        print(f"\nANALISI PER MAPPA:")
        print("-" * 40)
        mappe_unique = set(r.map_id for r in self.comparison_results)
        
        for map_id in sorted(mappe_unique):
            map_results = [r for r in self.comparison_results if r.map_id == map_id]
            map_total = len(map_results)
            
            map_wins_time = {e: sum(1 for r in map_results if r.winner_time == e) for e in euristiche}
            map_wins_efficiency = {e: sum(1 for r in map_results if r.winner_efficiency == e) for e in euristiche}
            
            print(f"Mappa {map_id} ({map_total} test):")
            print(f"   Tempo - " + ", ".join([f"{e}: {map_wins_time[e]}" for e in euristiche]))
            print(f"   Efficienza - " + ", ".join([f"{e}: {map_wins_efficiency[e]}" for e in euristiche]))
        
        # Raccomandazioni finali
        print(f"\nRACCOMANDAZIONI:")
        print("-" * 40)
        
        # Punteggio complessivo
        scores = {}
        for euristica in euristiche:
            score = (wins_time[euristica] * 0.3 + wins_cost[euristica] * 0.3 + wins_efficiency[euristica] * 0.4) / total_tests
            scores[euristica] = score
        
        best_heuristic = max(scores.keys(), key=lambda k: scores[k])
        
        for euristica, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"{euristica}: {score:.3f} (tempo: 30%, costo: 30%, efficienza: 40%)")
        
        print(f"\nCONCLUSIONI:")
        print(f"Euristica RACCOMANDATA: {best_heuristic}")
        
        # Analisi caratteristiche
        if best_heuristic == "Euclidea":
            print("✓ Bilancia bene distanza reale e performance")
            print("✓ Ottima per movimenti diagonali liberi")
        elif best_heuristic == "Manhattan":
            print("✓ Efficace quando i movimenti diagonali sono costosi")
            print("✓ Più conservativa, esplora meno nodi in certi scenari")
        elif best_heuristic == "Chebyshev":
            print("✓ Ottima per griglie dove movimenti diagonali = ortogonali")
            print("✓ Può essere più efficiente in labirinti aperti")
        
        print("\nNOTE:")
        print("- Euclidea: ideale per movimento libero in tutte le direzioni")
        print("- Manhattan: migliore quando i movimenti diagonali sono penalizzati")
        print("- Chebyshev: ottima quando tutti i movimenti hanno costo uguale")
        
        print("="*80)

    def salva_risultati_csv(self, filename="heuristic_results.csv"):
        """Salva i risultati in un file CSV per analisi future."""
        if not self.results:
            print("Nessun risultato da salvare!")
            return
        
        # Converte i risultati in un DataFrame
        data = []
        for result in self.results:
            data.append({
                'heuristic': result.heuristic,
                'map_id': result.map_id,
                'start_pos': str(result.start_pos),
                'end_pos': str(result.end_pos),
                'path_length': result.path_length,
                'total_cost': result.total_cost,
                'execution_time': result.execution_time,
                'path_found': result.path_found,
                'nodes_explored': result.nodes_explored,
                'heuristic_calls': result.heuristic_calls
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Risultati salvati in {filename}")

def main():
    """Funzione principale per eseguire i test delle euristiche."""
    # Lista delle mappe disponibili
    mappe_disponibili = [703368, 703326, 703323, 703372, 703381, 549362, 703428]
    
    print("TEST CONFRONTO EURISTICHE A*")
    print("Questo script confronterà Euclidea vs Manhattan vs Chebyshev")
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
    tester = HeuristicTester()
    tester.esegui_test_completo(mappe_da_testare, num_test)
    
    # Salva i risultati
    if input("\nVuoi salvare i risultati in CSV? (y/n): ").lower().startswith('y'):
        filename = input("Nome file (default: heuristic_results.csv): ").strip() or "heuristic_results.csv"
        tester.salva_risultati_csv(filename)

main()