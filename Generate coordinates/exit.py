import os
import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Imposta il percorso ai dati
CARTELLA_BASE = os.path.dirname(__file__)
CARTELLA_DATI = os.path.join(CARTELLA_BASE, "data")

def seleziona_uscite(percorso_immagine):
    """Permette all'utente di selezionare cliccando una o più uscite su una mappa."""
    immagine = np.array(Image.open(percorso_immagine))
    uscite = []

    def al_click(evento):
        if evento.xdata and evento.ydata:
            x, y = int(evento.xdata), int(evento.ydata)
            uscite.append((x, y))
            plt.plot(x, y, 'bo')  # cerchio blu
            plt.draw()

    fig, ax = plt.subplots()
    ax.imshow(immagine)
    fig.canvas.mpl_connect('button_press_event', al_click)
    print("Clicca su tutte le uscite di sicurezza. Chiudi la finestra quando hai finito.")
    plt.title("Clicca per selezionare le uscite")
    plt.show()

    return uscite

if __name__ == "__main__":
    id_mappa = input("Inserisci l'ID della mappa (703368, 703326, 703323, 703372, 703381, 703428): ")
    percorso_img = os.path.join(CARTELLA_DATI, f"{id_mappa}.png")

    if not os.path.exists(percorso_img):
        print("Immagine non trovata.")
        exit()

    uscite = seleziona_uscite(percorso_img)

    if not uscite:
        print("Nessuna uscita selezionata.")
        exit()

    # Salvataggio opzionale su file JSON
    percorso_output = os.path.join(CARTELLA_DATI, f"uscite-{id_mappa}.json")
    with open(percorso_output, "w", encoding="utf-8") as f:
        json.dump(uscite, f, indent=2)

    print(f"\nUscite selezionate (pixel):\n{uscite}")
    print(f"\nCoordinate salvate in {percorso_output}")
