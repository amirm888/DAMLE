# Cancer Classification using ViT with Validation, Regularization, and Feature Extraction
# Optimized for HPCC execution with all outputs saved to files

import os
from PIL import Image
from collections import Counter
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for HPCC
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(filename=os.path.join(output_path, 'training.log'), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Python version: {sys.version}")
logging.info(f"NumPy version: {np.__version__}")
logging.info(f"Matplotlib version: {matplotlib.__version__}")
print("Test GH")
# --- Configuration ---
data_path = r"/......./new_patches/new_patches/C_0/"  # change the path accordingly
output_path = r"/....../new_patches/new_patches/C_0/"  # change the path accordingly
os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset Class ---
class PathologyDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Load and Split Data ---
def load_and_split_data():
    samples = []
    for label_folder in ['L_0', 'L_1']:
        class_dir = os.path.join(data_path, label_folder)
        label = 0 if label_folder == 'L_0' else 1
        for dirpath, _, filenames in os.walk(class_dir):
            for fname in filenames:
                if fname.lower().endswith('.png'):
                    samples.append((os.path.join(dirpath, fname), label))

    print(f"Total samples collected: {len(samples)}")

    # Save label distribution
    label_counts = Counter(label for _, label in samples)
    with open(os.path.join(output_path, 'label_distribution.txt'), 'w') as f:
        f.write(str(dict(label_counts)))
    print(f"Label distribution saved to: {output_path}/label_distribution.txt")

    # Save sample paths and labels
    with open(os.path.join(output_path, 'sample_labels.txt'), 'w') as f:
        for image_path, label in samples:
            f.write(f"{image_path} {label}\n")
    print(f"Sample labels saved to: {output_path}/sample_labels.txt")

    # Split data: 80% train, 10% validation, 10% test
    labels = [label for _, label in samples]
    train_idx, temp_idx = train_test_split(range(len(samples)), test_size=0.2, stratify=labels, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=[labels[i] for i in temp_idx], random_state=42)

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    print(f"Training samples: {len(train_samples)}, Validation samples: {len(val_samples)}, Test samples: {len(test_samples)}")

    # Save split indices
    pd.DataFrame({'Index': train_idx}).to_csv(os.path.join(output_path, 'train_indices.csv'), index=False)
    pd.DataFrame({'Index': val_idx}).to_csv(os.path.join(output_path, 'val_indices.csv'), index=False)
    pd.DataFrame({'Index': test_idx}).to_csv(os.path.join(output_path, 'test_indices.csv'), index=False)

    # Save split image names
    pd.DataFrame([os.path.basename(p) for p, _ in train_samples], columns=["Image Name"]).to_excel(
        os.path.join(output_path, 'train_samples.xlsx'), index=False)
    pd.DataFrame([os.path.basename(p) for p, _ in val_samples], columns=["Image Name"]).to_excel(
        os.path.join(output_path, 'val_samples.xlsx'), index=False)
    pd.DataFrame([os.path.basename(p) for p, _ in test_samples], columns=["Image Name"]).to_excel(
        os.path.join(output_path, 'test_samples.xlsx'), index=False)

    return train_samples, val_samples, test_samples

# --- Define Transforms ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Initialize Model ---
def initialize_model():
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=2)
    # Freeze ViT backbone
    for param in model.vit.parameters():
        param.requires_grad = False
    # Keep classifier head trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
    model.to(device)
    print("Model initialized and moved to device.")
    return model

# --- Unfreeze ViT Layers ---
def unfreeze_last_vit_layers(model, num_layers=2):
    for name, param in model.vit.named_parameters():
        if any(f"layer.{11 - i}" in name for i in range(num_layers)):
            param.requires_grad = True
    print(f"Unfroze the last {num_layers} layers of the ViT backbone.")

# --- Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# --- Save Checkpoint ---
def save_model_checkpoint(model, optimizer, epoch, val_loss):
    checkpoint_path = os.path.join(output_path, f"checkpoint_epoch{epoch+1}_valloss{val_loss:.4f}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

# --- Training Loop ---
def train_model(model, train_loader, val_loader, epochs=10):
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    early_stopping = EarlyStopping(patience=5)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Unfreeze layers at epoch 3
        if epoch == 3:
            unfreeze_last_vit_layers(model, num_layers=2)

        # Training
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        # Validation
        model.eval()
        correct, total = 0, 0
        val_loss_total = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                loss = F.cross_entropy(outputs, labels)
                val_loss_total += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss_total / len(val_loader)
        val_acc = 100 * correct / total
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save checkpoint if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model_checkpoint(model, optimizer, epoch, avg_val_loss)

        # Scheduler and early stopping
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Save training history
    pd.DataFrame(history).to_csv(os.path.join(output_path, 'training_history.csv'), index=False)
    print(f"Training history saved to: {output_path}/training_history.csv")

    # Plot and save training curves
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label='Train Loss', marker='o')
    plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label='Val Loss', marker='x')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history['val_acc']) + 1), history['val_acc'], label='Val Accuracy', marker='s')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'training_curves.png'))
    plt.close()
    print(f"Training curves saved to: {output_path}/training_curves.png")

    return history

# --- Evaluate on Test Set ---
def evaluate_test_set(model, test_loader):
    model.eval()
    test_loss = 0.0
    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = F.cross_entropy(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100 * test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Accuracy: {test_acc:.2f}%, Avg Test Loss: {avg_test_loss:.4f}")

    # Save test metrics
    with open(os.path.join(output_path, 'test_metrics.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.2f}%\nAvg Test Loss: {avg_test_loss:.4f}\n")
    print(f"Test metrics saved to: {output_path}/test_metrics.txt")

    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix on Test Set")
    plt.savefig(os.path.join(output_path, 'confusion_matrix.png'))
    plt.close()
    print(f"Confusion matrix saved to: {output_path}/confusion_matrix.png")

    # Save classification report
    report = classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1"], digits=3)
    with open(os.path.join(output_path, 'classification_report.txt'), 'w') as f:
        f.write(report)
    print(f"Classification report saved to: {output_path}/classification_report.txt")

    return all_preds, all_labels

# --- Visualize Sample Predictions ---
def visualize_sample_predictions(model, test_loader):
    model.eval()
    sample_images, sample_labels, sample_preds = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)
            sample_images.extend(images.cpu())
            sample_labels.extend(labels.cpu())
            sample_preds.extend(predicted.cpu())
            if len(sample_images) >= 9:
                break

    # Plot and save sample predictions
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axs.flatten()):
        img = sample_images[i].permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)  # Denormalize
        ax.imshow(img)
        ax.set_title(f"True: {sample_labels[i]}, Pred: {sample_preds[i]}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'sample_predictions.png'))
    plt.close()
    print(f"Sample predictions saved to: {output_path}/sample_predictions.png")

# --- Extract Features for t-SNE ---
def extract_features(loader, model):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            outputs = model.vit(images).last_hidden_state[:, 0, :]  # [CLS] token
            features.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)

# --- t-SNE Visualization ---
def tsne_visualization(val_loader, model):
    val_features, val_labels = extract_features(val_loader, model)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_features = tsne.fit_transform(val_features)

    # Save t-SNE plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=val_labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Class')
    plt.title('t-SNE Visualization of Validation Set Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'tsne_visualization.png'))
    plt.close()
    print(f"t-SNE visualization saved to: {output_path}/tsne_visualization.png")

    # Save t-SNE features
    np.save(os.path.join(output_path, 'tsne_features.npy'), tsne_features)
    np.save(os.path.join(output_path, 'tsne_labels.npy'), val_labels)
    print(f"t-SNE features and labels saved to: {output_path}/tsne_features.npy, {output_path}/tsne_labels.npy")

# --- Extract and Save All Features ---
def extract_and_save_all_features(samples, model):
    all_dataset = PathologyDataset(samples, transform=val_transform)
    all_loader = DataLoader(all_dataset, batch_size=32, shuffle=False)
    all_features, all_labels = extract_features(all_loader, model)

    # Save features and labels
    np.save(os.path.join(output_path, 'all_features.npy'), all_features)
    np.save(os.path.join(output_path, 'all_labels.npy'), all_labels)
    print(f"All dataset features saved to: {output_path}/all_features.npy")
    print(f"All dataset labels saved to: {output_path}/all_labels.npy")

    # Save metadata
    metadata = pd.DataFrame({
        'Image_Name': [os.path.basename(sample[0]) for sample in samples],
        'Label': all_labels
    })
    metadata.to_excel(os.path.join(output_path, 'all_features_metadata.xlsx'), index=False)
    print(f"All dataset feature metadata saved to: {output_path}/all_features_metadata.xlsx")

# --- Main Execution ---
def main():
    # Load and split data
    train_samples, val_samples, test_samples = load_and_split_data()

    # Create datasets and loaders
    train_dataset = PathologyDataset(train_samples, transform=train_transform)
    val_dataset = PathologyDataset(val_samples, transform=val_transform)
    test_dataset = PathologyDataset(test_samples, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print("Data loaders created successfully.")

    # Initialize model
    model = initialize_model()

    # Train model
    history = train_model(model, train_loader, val_loader, epochs=10)

    # Evaluate on test set
    evaluate_test_set(model, test_loader)

    # Visualize sample predictions
    visualize_sample_predictions(model, test_loader)

    # t-SNE visualization
    tsne_visualization(val_loader, model)

    # Extract and save all features
    extract_and_save_all_features(train_samples + val_samples + test_samples, model)

if __name__ == "__main__":
    main()