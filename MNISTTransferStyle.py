import os
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torchvision.models import VGG19_Weights
from torch.utils.tensorboard import SummaryWriter

# FUNZIONI DI UTILITÀ E PREPROCESSING
def load_image(image, max_size=280, shape=None):
    """
    Carica un'immagine PIL, la ridimensiona e la converte in un tensore PyTorch normalizzato.
    Se 'shape' viene specificato, viene usato per il ridimensionamento.
    """
    size = max_size if max(image.size) > max_size else max(image.size)
    if shape is not None:
        size = shape
    transform_pipeline = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # Normalizzazione per 3 canali (RGB)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform_pipeline(image).unsqueeze(0)

def augment_image(image):
    """
    Applica trasformazioni casuali a un'immagine PIL:
      - Rotazione casuale (fino a 20°)
      - Ritaglio e zoom casuale
      - Variazione di luminosità e contrasto
      - Flip orizzontale casuale
    Restituisce la nuova immagine PIL.
    """
    augmentation = transforms.Compose([
        transforms.RandomRotation(degrees=(0, 20)),
        transforms.RandomResizedCrop(size=(56, 56), scale=(0.75, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(p=0.5)
    ])
    return augmentation(image)

def get_content_image(path, nr_img=10):
    """
    Combina nr_img x nr_img immagini prese casualmente dalla cartella 'path'
    in un'unica immagine composita (PIL) ridimensionata a 280x280.
    """
    rows = []
    for _ in range(nr_img):
        row_images = []
        for _ in range(nr_img):
            all_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not all_files:
                raise ValueError("La cartella non contiene immagini valide.")
            random_file = random.choice(all_files)
            img = Image.open(os.path.join(path, random_file)).convert('RGB')
            img_aug = augment_image(img)
            row_images.append(np.array(img_aug))
        row_combined = np.hstack(row_images)
        rows.append(row_combined)
    full_image = np.vstack(rows)
    full_image = Image.fromarray(full_image).convert('RGB')
    return full_image.resize((280, 280), resample=Image.Resampling.LANCZOS)

def get_style_image_mnist(mnist_data, nr_img=10):
    """
    Combina nr_img x nr_img immagini dal dataset MNIST in un'unica immagine composita (PIL)
    ridimensionata a 280x280.
    """
    mnist_array = mnist_data.data.numpy()
    rows = []
    for _ in range(nr_img):
        row_images = []
        for _ in range(nr_img):
            idx = random.randint(0, len(mnist_array) - 1)
            single_img = mnist_array[idx]  # Immagine 28x28 in scala di grigi
            row_images.append(single_img)
        row_combined = np.hstack(row_images)
        rows.append(row_combined)
    full_image = np.vstack(rows)
    full_image = Image.fromarray(full_image).convert('RGB')
    return full_image.resize((280, 280), resample=Image.Resampling.LANCZOS)

def im_convert(tensor):
    """
    Converte un tensore PyTorch (con batch=1) normalizzato in un array NumPy (H, W, C)
    con valori compresi tra 0 e 1.
    """
    image = tensor.detach().cpu().clone().numpy()
    image = image.squeeze()  # Rimuove la dimensione batch
    image = image.transpose(1, 2, 0)  # Da (C, H, W) a (H, W, C)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    return np.clip(image, 0, 1)

def get_features(image, model):
    """
    Passa l'immagine attraverso alcuni layer chiave di VGG19 e restituisce un dizionario
    contenente le feature map. I layer selezionati sono:
      conv1_1, conv2_1, conv3_1, conv4_1, conv4_2 (per il contenuto) e conv5_1.
    """
    layers = {
        '0':  'conv1_1',
        '5':  'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',
        '28': 'conv5_1'
    }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    """
    Calcola la Gram Matrix di un tensore (assumendo batch_size=1) con shape [1, C, H, W].
    La Gram Matrix cattura le correlazioni tra le feature maps ed è usata per la style loss.
    """
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t())

def overlay_images(background, overlay, alpha=0.5):
    """
    Crea un’immagine ottenuta sovrapponendo 'overlay' su 'background' mediante blending.
    L'alpha controlla la trasparenza dell'immagine overlay.
    """
    return Image.blend(background, overlay, alpha)

# FUNZIONI DI LOSS 
def custom_content_loss(target_features, content_features, layer='conv4_2'):
    """
    Calcola la content loss (MSE) tra le feature map del layer 'conv4_2'
    dell'immagine target e quelle dell'immagine di content.
    """
    return torch.mean((target_features[layer] - content_features[layer]) ** 2)

def custom_style_loss(target_features, style_grams, style_weights):
    """
    Calcola la style loss sommando il contributo dei layer definiti in 'style_weights'.
    Per ogni layer, confronta la Gram Matrix dell'immagine target con quella dell'immagine di stile.
    """
    loss = 0.0
    for layer in style_weights:
        t_feat = target_features[layer]
        t_gram = gram_matrix(t_feat)
        s_gram = style_grams[layer]
        layer_loss = style_weights[layer] * torch.mean((t_gram - s_gram) ** 2)
        _, d, h, w = t_feat.shape
        loss += layer_loss / (d * h * w)
    return loss

# FUNZIONI PER SALVATAGGIO E RESUME
def save_checkpoint(model, optimizer, epoch, path, target):
    """
    Salva un checkpoint che include lo stato del modello, dell'ottimizzatore e l'epoca corrente.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'target_state': target.detach().cpu(),  # Salva il target corrente
        'epoch': epoch
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint salvato in {path}")

def load_checkpoint(model, optimizer, path, device):
    """
    Carica un checkpoint e ripristina lo stato del modello e dell'ottimizzatore.
    Restituisce l'epoca da cui riprendere l'allenamento.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    target = checkpoint['target_state'].to(device).requires_grad_(True)
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint caricato da {path}. Riprendi dall'epoca {start_epoch}")
    return start_epoch, target

def save_generated_image(image, folder, filename):
    """
    Salva l'immagine (formato PIL) nella cartella 'folder' con il nome 'filename'.
    Se la cartella non esiste, la crea.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    image.save(os.path.join(folder, filename))
    print(f"Immagine salvata in {os.path.join(folder, filename)}")

# FUNZIONE DI STYLE TRANSFER
def run_style_transfer(content, style, model, config, device, exp_name):
    """
    Esegue lo style transfer ottimizzando direttamente l'immagine target.
    Se esiste un checkpoint per questo esperimento, riprende dall'epoca salvata.
    """
    # Parametri
    n_epochs = config.get("n_epochs", 10)
    steps_per_epoch = config.get("steps_per_epoch", 1000)
    learning_rate = config.get("learning_rate", 0.01)
    optimizer_choice = config.get("optimizer", "adam")
    content_weight = config.get("content_weight", 1.0)
    style_weight = config.get("style_weight", 50.0)
    style_weights = config.get("style_weights", {
        'conv1_1': 0.2,
        'conv2_1': 0.2,
        'conv3_1': 0.2,
        'conv4_1': 0.2,
        'conv5_1': 0.2
    })
    show_every = config.get("show_every", 300)

    writer = SummaryWriter(log_dir=f"runs/{exp_name}")

    # Estrai le feature (e stacca i tensori dal grafo)
    content_features = {k: v.detach() for k, v in get_features(content, model).items()}
    style_features = {k: v.detach() for k, v in get_features(style, model).items()}
    style_grams = {layer: gram_matrix(style_features[layer]).detach() for layer in style_weights}

    # Inizializza il target come clone dell'immagine di contenuto
    target = content.clone().requires_grad_(True).to(device)

    # Check se fine-tuning di VGG è abilitato
    finetune_vgg = config.get("finetune_vgg", False)
    if finetune_vgg:
        optimizer = optim.Adam([
            {'params': [target], 'lr': learning_rate},
            {'params': model.parameters(), 'lr': config.get("vgg_learning_rate", 0.0001)}
        ])
        print("Fine-tuning abilitato: i pesi di VGG verranno aggiornati con un lr ridotto.")
    else:
        optimizer = optim.Adam([target], lr=learning_rate)

    # Verifica l'esistenza di un checkpoint
    checkpoint_path = f"checkpoint_{exp_name}.pth" # Checkpoint di default
    start_epoch = 0
    print(f"Verifica esistenza di checkpoint in {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        # Carica checkpoint esistente
        start_epoch, target = load_checkpoint(model, optimizer, checkpoint_path, device)
        print(f"Ripreso da epoca {start_epoch}/{n_epochs}")
    else:
        print("Nessun checkpoint trovato, avvio da epoca 0")

    print(f"{exp_name}: ottimizzazione con {optimizer_choice}, lr={learning_rate}")

    global_step = start_epoch * steps_per_epoch
    # Riprendi l'allenamento da start_epoch
    for epoch in range(start_epoch, n_epochs):
        for step in range(1, steps_per_epoch + 1):
            target_features = get_features(target, model)
            # Content Loss
            c_loss = custom_content_loss(target_features, content_features, layer='conv4_2')
            # Style Loss
            s_loss = custom_style_loss(target_features, style_grams, style_weights)
            total_loss = content_weight * c_loss + style_weight * s_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            global_step += 1
            if step % show_every == 0:
                print(f"Exp: {exp_name} - Epoch [{epoch+1}/{n_epochs}], Step [{step}/{steps_per_epoch}]"
                      f" - Total Loss: {total_loss.item():.4f}")
                writer.add_scalar("Loss/Content", c_loss.item(), global_step)
                writer.add_scalar("Loss/Style", s_loss.item(), global_step)
                writer.add_scalar("Loss/Total", total_loss.item(), global_step)
                # Salva anteprima immagine in TensorBoard
                preview_img = im_convert(target).transpose(2,0,1)
                writer.add_image("Image/Target", preview_img, global_step)

        # Salva un checkpoint e un'anteprima a ogni epoca
        checkpoint_img = im_convert(target)
        checkpoint_pil = Image.fromarray((checkpoint_img * 255).astype(np.uint8))
        # Salva una jpg per riferimento visivo
        checkpoint_filename = f'checkpoint_{exp_name}_epoch_{epoch+1}.jpg'
        checkpoint_pil.save(checkpoint_filename)
        print(f"Salvato checkpoint immagine: {checkpoint_filename}")
        # Salva lo stato del training
        save_checkpoint(model, optimizer, epoch, checkpoint_path, target)

    writer.close()
    print(f"{exp_name} completato.\n")
    return target

# SINGLE-IMAGE VERSION (OPZIONALE)
def apply_style_transfer_to_single_image(content_path, mnist_data, model, config, device):
    """
    Applicazione Style Transfer su una singola immagine.
    """

    # Scegle una singola immagine di contenuto
    all_files = [f for f in os.listdir(content_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not all_files:
        raise ValueError("La cartella non contiene immagini valide.")
    random_file = random.choice(all_files)
    content_image = Image.open(os.path.join(content_path, random_file)).convert('RGB')
    content_tensor = load_image(content_image).to(device)

    # Scegle una singola immagine di stile da MNIST
    idx = random.randint(0, len(mnist_data) - 1)
    style_image = mnist_data.data[idx].numpy()
    style_image = Image.fromarray(style_image).convert('RGB')
    style_tensor = load_image(style_image, shape=content_tensor.shape[-2:]).to(device)

    # Visualizza input
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Content Image")
    plt.imshow(content_image)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Style Image (MNIST)")
    plt.imshow(style_image)
    plt.axis('off')
    plt.show()
    exp_name ="single_exp"
    final_target = run_style_transfer(content_tensor, style_tensor, model, config, device, exp_name)
    final_img_array = im_convert(final_target)
    final_pil = Image.fromarray((final_img_array * 255).astype(np.uint8))
    # Mostra risultato
    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1)
    plt.title("Content Image")
    plt.imshow(content_image)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title("Style Image")
    plt.imshow(style_image)
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.title("Stylized Image")
    plt.imshow(final_pil)
    plt.axis('off')
    plt.show()

    # Salva risultato
    save_generated_image(final_pil, "results", "styled_single_image.jpg")
    print("Trasferimento di stile completato per singola immagine.")

# MAIN
def main():
    # Carica la configurazione dal file YAML
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)  # Se non presente, usa 42 come default
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Imposta il dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carica VGG19 pre-addestrato
    vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
    if not config.get("finetune_vgg", False):
        for param in vgg.parameters():
            param.requires_grad_(False)
    vgg.to(device)

    # Carica dataset MNIST
    mnist_data = datasets.MNIST(root="./", train=True, download=True, transform=transforms.ToTensor())

    if config.get("single_image", False) is False:
        print("Esecuzione dello Style Transfer su immagini multiple.")
        # Immagini composte (10x10) per content e style
        content_full = get_content_image(config.get("content_folder", "numbers"), nr_img=10)
        style_full = get_style_image_mnist(mnist_data, nr_img=10)

        # Mostra content e style
        print("Visualizzazione delle immagini di Content e Style:")
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title("Content")
        plt.imshow(content_full)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.title("Style (MNIST)")
        plt.imshow(style_full)
        plt.axis('off')
        plt.show()

        # Converte in tensori
        content_tensor = load_image(content_full).to(device)
        style_tensor = load_image(style_full, shape=content_tensor.shape[-2:]).to(device)

        # Esegui lo style transfer per ogni esperimento definito nel file di configurazione
        experiments = config.get("experiments", [])
        results_folder = config.get("results_folder", "results")

        for exp_config in experiments:
            exp_name = exp_config.get("exp_name", "default_exp")
            print(f"\nAvvio: {exp_name}")
            final_target = run_style_transfer(content_tensor, style_tensor, vgg, exp_config, device, exp_name)

            # Salva risultato finale
            final_img_array = im_convert(final_target)
            final_pil = Image.fromarray((final_img_array * 255).astype(np.uint8))
            final_filename = f'final_result_{exp_name}.jpg'
            save_generated_image(final_pil, results_folder, final_filename)

            # Crea overlay
            overlay = overlay_images(content_full, final_pil, alpha=0.5)
            overlay_filename = f'overlay_{exp_name}.jpg'
            save_generated_image(overlay, results_folder, overlay_filename)

            # Mostra a schermo
            print(f"{exp_name} completato e immagini salvate in '{results_folder}'")
            fig, axs = plt.subplots(1,4, figsize=(20,5))
            axs[0].imshow(content_full); axs[0].set_title("Content"); axs[0].axis('off')
            axs[1].imshow(style_full); axs[1].set_title("Style"); axs[1].axis('off')
            axs[2].imshow(final_img_array); axs[2].set_title("Result"); axs[2].axis('off')
            axs[3].imshow(overlay); axs[3].set_title("Overlay"); axs[3].axis('off')
            plt.show()
    else:
        # Se si vuoi lavorare su singola immagine
        print("Esecuzione dello Style Transfer su singola immagine.")
        apply_style_transfer_to_single_image(
            content_path=config.get("content_folder", "numbers"),
            mnist_data=mnist_data,
            model=vgg,
            config=config,
            device=device
        )

if __name__ == "__main__":
    main()
