import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = './trashnet_dataset' 
BATCH_SIZE = 32
NUM_EPOCHS = 5
NUM_CLASSES = 6 # Plastic, Paper, Organic, Metal, Glass, Trash
LEARNING_RATE = 0.001 # <-- Added the missing variable here!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware selected for training: {device}")

# ==========================================
# 2. DATA LOADERS (Reading the images)
# ==========================================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# IMPORTANT: num_workers=0 is required for Windows to prevent memory crashing during training
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0) for x in ['train', 'val']}

# ==========================================
# 3. BUILD MODEL
# ==========================================
def build_model():
    print("Loading pre-trained MobileNetV2 brain...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    return model.to(device)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train_model(model, criterion, optimizer, num_epochs=5):
    print("Starting Training Process...")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return model

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == '__main__':
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=LEARNING_RATE)
    
    trained_model = train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS)
    
    torch.save(trained_model.state_dict(), 'waste_classifier_mobilenet.pth')
    print("✅ Model fully trained and saved as 'waste_classifier_mobilenet.pth'!")