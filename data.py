from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.endpoints import PlayerGameLog
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.nn.functional as F


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import kagglehub
import os
import cv2
import torch


"""Lebron's Career Statistics"""
career_stats = playercareerstats.PlayerCareerStats(player_id=2544).get_data_frames()[0]

x = []
y = []

for i, szn in career_stats.iterrows():
    x.append(i)
    y.append(szn["PTS"] / szn["GP"])

plt.plot(x, y, marker='o')
plt.show()


"""create a neural network to predict matchups"""
nba_teams = teams.get_teams()
team_abbr_to_id = {team["abbreviation"]: team["id"] for team in nba_teams}
"""DataFrame is...."""
all_games = pd.DataFrame()


"""for team in nba_teams:
    id = team["id"]
    gamefinder = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=id, season_type_nullable="Regular Season"
    )
    games = gamefinder.get_data_frames()[0]
    all_games = pd.concat([all_games, games], ignore_index=True)
    print(all_games)
"""

"""path = kagglehub.dataset_download("samithsachidanandan/human-face-emotions")"""


label_types = os.listdir('Data/')
print(label_types)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
"""We want to be able to parse live video and provide analysis on emotions state based on frame data. 
This may also be used for basketball related statistics on play-by-play events in prediction markets. """

"""took this from a online article on designing machine learning models"""
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        acc = accuracy(out, labels)  
        return loss,acc
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['train_accuracy'], result['val_loss'], result['val_acc']))


"""output: (batch_size, num_classes)"""
class EmotionEncoding(nn.Module):
    def __init__(self, ):
        super(EmotionEncoding, self).__init__()
        """Initialize internal Module state, shared by both nn.Module and ScriptModule."""
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128->64

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64->32

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 32->16
        )

        # 128 channels * 6 * 6 feature map. Pooling returns a feature map which is necessary to flatten.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            #helps prevent overfitting cases
            nn.Dropout(0.3),
            #numclasses = 5
            nn.Linear(256, 5) 
        )


    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x

model = EmotionEncoding()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])
"""train_loader = DataLoader(train_set, batch_size=32, shuffle=True)"""

dataset = ImageFolder("Data/", transform=transform)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
criterion = nn.CrossEntropyLoss()

train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds, batch_size=32, num_workers=0, pin_memory=False)
best_val_acc = 0.0


for epoch in range(50):
    model.train()
    train_loss = 0
    train_acc = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_acc += torch.sum(preds == labels).item()
        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss {loss.item():.4f}")


    train_loss /= len(train_loader)
    train_acc /= len(train_ds)

    # Validation for model
    model.eval()
    val_loss = 0
    val_acc = 0
    """"non calculating gradients"""
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_acc += torch.sum(preds == labels).item()

    val_loss /= len(val_loader)
    val_acc /= len(val_ds)

    print(f"Epoch {epoch+1}: "
          f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_emotion_model.pth")
        print(f"Saved new best model with val_acc={best_val_acc:.4f}")

