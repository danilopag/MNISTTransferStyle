"""
analyze_tensorboard.py

Questo script utilizza l'EventAccumulator di TensorBoard per leggere i log
salvati dal SummaryWriter durante il training (i log sono generati nella directory specificata).
Vengono estratte le curve di Loss/Content, Loss/Style e Loss/Total, e vengono tracciate in un grafico.

Come usarlo:
    python analyze_tensorboard.py <log_directory>

Ad esempio:
    python analyze_tensorboard.py runs/exp_lr0.01_content1_style50_epochs5
"""

import os
import sys
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def load_scalar_data(log_dir, tag):
    """
    Carica i dati scalari per un dato tag dalla directory di log di TensorBoard.
    
    Args:
        log_dir (str): La directory dove sono salvati i file di log.
        tag (str): Il tag del valore scalare da estrarre (es. "Loss/Content").
    
    Returns:
        steps (list): Lista degli step (iterazioni) corrispondenti.
        values (list): Lista dei valori scalari.
    """
    # Imposta size_guidance per avere una buona copertura dei dati (in questo caso 1000 valori)
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={event_accumulator.SCALARS: 1000})
    ea.Reload()  # Carica tutti gli eventi dal file di log
    # Estrae i dati scalari per il tag specificato
    scalar_list = ea.Scalars(tag)
    steps = [s.step for s in scalar_list]
    values = [s.value for s in scalar_list]
    return steps, values

def plot_scalars(log_dir, tags):
    """
    Traccia le curve per i tag specificati.
    
    Args:
        log_dir (str): Directory dei log.
        tags (list): Lista dei tag da tracciare (es. ["Loss/Content", "Loss/Style", "Loss/Total"]).
    
    Returns:
        fig, ax: Figura e asse matplotlib.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    for tag in tags:
        steps, values = load_scalar_data(log_dir, tag)
        ax.plot(steps, values, label=tag)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Andamento delle Loss dal training\n(Log directory: {log_dir})")
    ax.legend()
    ax.grid(True)
    return fig, ax

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_tensorboard.py <log_directory>")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    
    # Definiamo i tag che vogliamo analizzare
    tags = ["Loss/Content", "Loss/Style", "Loss/Total"]
    
    # Tracciamo le curve delle loss
    fig, ax = plot_scalars(log_dir, tags)
    
    # Salviamo il grafico in una immagine nella stessa directory dei log
    output_path = os.path.join(log_dir, "tensorboard_loss_plot.png")
    fig.savefig(output_path)
    print(f"Plot salvato in: {output_path}")
    
    # Visualizza il grafico
    plt.show()

if __name__ == "__main__":
    main()
