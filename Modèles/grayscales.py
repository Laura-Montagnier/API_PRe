import os
import glob
import pickle
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

# 1. Définir le chemin vers les images
image_dir = '/mnt/c/Users/monta/Desktop/BODMAS2/Images_Grayscale'

# 2. Obtenir la liste des images et des étiquettes
image_paths = glob.glob(os.path.join(image_dir, '*', '*.png'))  # assuming images are in .png format
labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]
label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
idx_labels = [label_to_idx[label] for label in labels]

# 3. Diviser les données en ensembles d'entraînement, de validation et de test
images_tv, images_test, y_tv, y_test = train_test_split(image_paths, idx_labels, shuffle=True, test_size=0.2, random_state=123)
images_train, images_val, y_train, y_val = train_test_split(images_tv, y_tv, shuffle=True, test_size=0.25, random_state=123)

class CT_Dataset(Dataset):
    def __init__(self, img_path, img_labels, img_transforms=None, grayscale=True):
        self.img_path = img_path
        self.img_labels = torch.tensor(img_labels, dtype=torch.long)  # Use torch.long for CrossEntropyLoss
        
        if img_transforms is None:
            if grayscale:
                self.transforms = transforms.Compose([
                    transforms.Grayscale(), 
                    transforms.Resize((128, 128)),  # Reduce image size
                    transforms.ToTensor()
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.Resize((128, 128)),  # Reduce image size
                    transforms.ToTensor()
                ])
        else:
            self.transforms = img_transforms

    def __getitem__(self, index):
        cur_path = self.img_path[index]
        cur_img = Image.open(cur_path).convert('RGB')
        cur_img = self.transforms(cur_img)
        return cur_img, self.img_labels[index]

    def __len__(self):
        return len(self.img_path)

# Define CNN model
class Convnet(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(Convnet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        
        # Compute the flattened size for the first linear layer
        with torch.no_grad():
            self.flattened_size = self._get_flattened_size()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=self.flattened_size, out_features=512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes)  # Change to num_classes
        )

    def _get_flattened_size(self):
        sample_input = torch.zeros(1, 1, 128, 128)  # Change to (1, 3, 128, 128) if using RGB images
        sample_output = self.convnet(sample_input)
        return sample_output.numel()

    def forward(self, x):
        x = self.convnet(x)
        x = self.classifier(x)
        return x

# Define training function
def train_model(model, train_dataset, val_dataset, test_dataset, device, lr=0.0001, epochs=30, batch_size=16, l2=0.00001, gamma=0.5, patience=7):  # Reduce batch size
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multiclass classification
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=patience, gamma=gamma)

    print("Training Start:")
    for epoch in range(epochs):
        model.train()
        
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_acc += (preds == labels).sum().item() / batch_size

        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_acc += (preds == labels).sum().item() / batch_size

        scheduler.step()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}, lr: {optimizer.param_groups[0]['lr']:.5f}, train loss: {train_loss:.5f}, train acc: {train_acc:.5f}, val loss: {val_loss:.5f}, val acc: {val_acc:.5f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

    test_acc = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            test_acc += (preds == labels).sum().item()

    print(f'Test Accuracy: {test_acc / len(test_loader):.5f}')

    return history

num_classes = len(set(idx_labels))
train_dataset = CT_Dataset(img_path=images_train, img_labels=y_train)
val_dataset = CT_Dataset(img_path=images_val, img_labels=y_val)
test_dataset = CT_Dataset(img_path=images_test, img_labels=y_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn_model_grayscale = Convnet(num_classes=num_classes, dropout=0.5)
history = train_model(cnn_model_grayscale, train_dataset, val_dataset, test_dataset, device, lr=0.0002, batch_size=16, epochs=5, l2=0.09, patience=5)

# Save the trained model
torch.save(cnn_model_grayscale.state_dict(), 'cnn_model_grayscale.pth')

# Save the trained model to a file
with open('cnn_model_grayscale.pkl', 'wb') as f:
    pickle.dump(cnn_model_grayscale, f)





