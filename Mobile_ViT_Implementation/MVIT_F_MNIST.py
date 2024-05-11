import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Define the model
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride):
        super(MobileViTBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))

        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileViT(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileViT, self).__init__()
        self.conv1 = ConvBNReLU(1, 16, kernel_size=3, stride=2, padding=1)
        self.blocks = nn.Sequential(
            MobileViTBlock(16, 16, expand_ratio=1, stride=1),
            MobileViTBlock(16, 24, expand_ratio=6, stride=2),
            MobileViTBlock(24, 24, expand_ratio=6, stride=1),
            MobileViTBlock(24, 32, expand_ratio=6, stride=2),
            MobileViTBlock(32, 32, expand_ratio=6, stride=1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)  # Adjusted output size to match 32x32 input

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Hyperparameters
batch_size = 128
learning_rate = 0.001
num_epochs = 10

# Load FashionMNIST dataset
transform = transforms.Compose([
    transforms.Resize(32),  # Resize images to 32x32
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = datasets.FashionMNIST(root='./FMdata', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./FMdata', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Loaded")

# Initialize model
model = MobileViT(num_classes=10).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

import matplotlib.pyplot as plt

def plot_losses(train_losses, test_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Losses')
    plt.legend()
    plt.savefig("MNISTF.jpg")
    plt.close()

# Train the model
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, targets) in tqdm(enumerate(train_loader)):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)
        running_loss += loss.item()

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or Adam step
        optimizer.step()
    train_losses.append(running_loss / len(train_loader))
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Forward
            scores = model(data)
            loss = criterion(scores, targets)
            running_test_loss += loss.item()

    # Calculate average test loss for the epoch
    test_loss = running_test_loss / len(test_loader)
    test_losses.append(test_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
plot_losses(train_losses, test_losses)

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

# Save the trained model
torch.save(model.state_dict(), 'fashionmnist_mobilevit_model_32x32.pth')

def display_and_test_image(model, device, loader, classes):
    model.eval()
    dataiter = iter(loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    # Get predictions
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Select the first image in the batch
    img = images[0].cpu()
    label = labels[0].cpu()
    predicted_label = predicted[0].cpu()

    # Inverse normalize the image
    img = img.permute(1, 2, 0).numpy() * 0.5 + 0.5
    img = np.clip(img, 0, 1)

    # Plot and save the image
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f'Actual: {classes[label]} - Predicted: {classes[predicted_label]}')
    plt.savefig('fashionmnist_test_image_prediction.png')
    plt.show()

    # Print confidence scores
    softmax = nn.Softmax(dim=1)
    conf_scores = softmax(outputs)[0] * 100
    print(f"Confidence Scores: {conf_scores}")

# Assign class labels for FashionMNIST
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# After training the model and running the accuracy checks, call this function
display_and_test_image(model, device, test_loader, classes)
