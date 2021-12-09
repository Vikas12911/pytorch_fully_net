# import 
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.datasets as datasets

# Model
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# HyperParameter
in_channels = 1
num_classes = 10
learning_rate = 0.001 
batch_size = 64
num_epoch = 3

# Load Data
train_dataset = datasets.MNIST(root = 'dataset2' , train = True ,transform = transforms.ToTensor(), download = True)
test_dataset = datasets.MNIST(root = 'dataset2' , train = False , transform = transforms.ToTensor() , download = True)
train_dl = DataLoader(dataset = train_dataset,batch_size = batch_size  )
test_dl = DataLoader(dataset = test_dataset,batch_size = batch_size  )

# Initilize Network
model = CNN(in_channels = in_channels , num_classes = num_classes).to(device)

# Loss and optimizer
criterian = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Accuracy Function
def check_accuracy( model , loader):
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


    model.train()
    return print(f"Accuracy of the model: {(num_correct/num_samples)*100}")	

# Train Network
for i in range(num_epoch):
	for idx ,( data ,targets) in enumerate(tqdm(train_dl)):
		data = data.to(device)
		targets = targets.to(device)
		score = model(data)
		loss = criterian(score , targets)
		optimizer.zero_grad()	
		loss.backward()
		optimizer.step()

	check_accuracy(model , test_dl)

		
def save_checkpoint(state,filename = "mymodel.pth.tar"):
	print("Saving Checkpoint")
	torch.save(state , filename)

def load_checkpoint(checkpoint , model , optimizer):
	print("Loading checkpoint")
	model.load_state_dict(checkpoint["state_dict"])
	optimizer.load_state_dict(checkpoint['optimizer'])

def main():



	checkpoint = { 'state_dict': model.state_dict() , 'optimizer':optimizer.state_dict()}
	
	save_checkpoint(checkpoint)
	
	load_checkpoint(checkpoint , model , optimizer)

if __name__=="__main__":
	main()
	 
