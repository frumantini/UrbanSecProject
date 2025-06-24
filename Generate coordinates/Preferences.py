import os
import json
import cv2
import matplotlib.pyplot as plt

def seleziona_strada_matplotlib(img):
    """Permette di selezionare punti sulla mappa per definire il percorso preferibile."""
    punti = []

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title("Seleziona i punti del percorso preferibile", fontsize=14)

    print("👉 Clicca per selezionare i punti del percorso preferibile.")
    print("   I punti verranno collegati nell'ordine di selezione.")
    print("   Chiudi la finestra quando hai finito.")

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            punti.append([x, y])
            
            # Disegna il punto
            ax.plot(x, y, 'go', markersize=8)
            ax.annotate(str(len(punti)), (x, y), xytext=(5, 5), 
                       textcoords='offset points', color='white', 
                       fontweight='bold', fontsize=10)
            
            # Disegna la linea se ci sono almeno 2 punti
            if len(punti) > 1:
                x_prev, y_prev = punti[-2]
                ax.plot([x_prev, x], [y_prev, y], 'g-', linewidth=2, alpha=0.7)
            
            fig.canvas.draw()
            print(f"   Punto {len(punti)}: ({x}, {y})")

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return punti

def carica_o_aggiorna_preferenze(mappa_id, nuovi_punti):
    """Carica le preferenze esistenti e aggiunge/aggiorna quelle per la mappa corrente."""
    cartella_base = os.path.dirname(os.path.abspath(__file__))
    file_json = os.path.join(cartella_base, "preferenze.json")
    
    # Carica preferenze esistenti se il file esiste
    preferenze = {}
    if os.path.exists(file_json):
        try:
            with open(file_json, "r") as f:
                preferenze = json.load(f)
            print(f" Preferenze esistenti caricate da {file_json}")
        except json.JSONDecodeError:
            print("File preferenze corrotto, verrà ricreato.")
    
    # Aggiorna con i nuovi punti
    preferenze[mappa_id] = nuovi_punti
    
    # Salva le preferenze aggiornate
    with open(file_json, "w") as f:
        json.dump(preferenze, f, indent=2)
    
    return file_json

def main():
    # Lista delle mappe disponibili (puoi modificarla secondo le tue necessità)
    mappe_disponibili = ["703368", "703326", "703323", "703372", "703381", "703428"]
    
    print("Selettore Percorsi Preferibili")
    print(f"Mappe disponibili: {', '.join(mappe_disponibili)}")
    
    mappa_id = input("Inserisci ID della mappa: ").strip()
    
    if mappa_id not in mappe_disponibili:
        print(f"Mappa {mappa_id} non trovata nella lista. Procedo comunque...")
    
    cartella_base = os.path.dirname(os.path.abspath(__file__))
    cartella_data = os.path.join(cartella_base, "data")
    file_img = os.path.join(cartella_data, f"{mappa_id}.png")

    if not os.path.isfile(file_img):
        print(f"Immagine per la mappa {mappa_id} non trovata in {file_img}")
        return

    print(f"Seleziona il percorso preferibile per la mappa {mappa_id}")
    img = cv2.imread(file_img)
    
    if img is None:
        print(f" Impossibile caricare l'immagine {file_img}")
        return
    
    punti = seleziona_strada_matplotlib(img)

    if not punti:
        print(f"Nessun punto selezionato per la mappa {mappa_id}")
        return

    print(f" Selezionati {len(punti)} punti per il percorso preferibile")
    
    # Salva le preferenze
    file_json = carica_o_aggiorna_preferenze(mappa_id, punti)
    print(f" Percorso preferibile salvato in '{file_json}'")
    
    # Mostra un riepilogo
    print("\nRiepilogo punti selezionati:")
    for i, (x, y) in enumerate(punti, 1):
        print(f"   {i}. ({x}, {y})")

if __name__ == "__main__":
    main()