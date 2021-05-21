
import torch
import torch.nn as nn
from matplotlib.path import Path
from torch.utils.data import DataLoader

epochs = 7
learning_rate = 0.001
k_fold_accuracy =[]
accuracy_total = []
loss_total =[]
ratio=[]
aggregate=[]
def train_dataloader(train_df) -> DataLoader:
    x = DataLoader(train_df, batch_size=32, shuffle=True, num_workers=0)
    return x


class FaceMaskDetectorCNN(nn.Module):
    def __init__(self):
        super(FaceMaskDetectorCNN, self).__init__()
        self.conv_layer = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(8 * 8 * 32, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x


# model = FaceMaskDetectorCNN(5,3,3)

def train_model(t, model,size, fold_instance,train_total=0, train_correct=0,):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Training the model on ",size,"images ")
    for epoch in range(epochs):
        loaded_data =train_dataloader(t)
        iter_num = len(loaded_data)
        for i, data in enumerate(loaded_data):
            inputs, labels = data['image'], data['class']

            labels = labels.flatten()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_total.append(loss)

            # Backpropogation and performing Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            train_total += total
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            train_correct +=correct
            #print("Actual class:", labels, "Predicted class :", predicted)
            accuracy_total.append(100*correct/total)

            ratio.append(loss/100*correct/total)
            print('Epoch [{}], Iteration [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, i,iter_num, loss.item(),correct/total * 100))
    acc = float(100 * train_correct / train_total)
    #print(acc)
    aggregate.append(acc)
    #print("Training accuracy {:.2f}%".format( 100 * train_correct / train_total))
    k_fold_accuracy.append(" %d Fold accuracy: %f"%(fold_instance,acc))
    # torch.save(Path('Face Mask Dataset 3/module.pt'))
    # model_t = torch.load(Path('Face Mask Dataset 3/module.pt'))



