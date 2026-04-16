import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import configparser
import torch.nn.functional as F
from admm import (
    ADMM_SpikingLinear, ADMM_Flatten, ADMM_SpatialPool, ADMM_SpikingConv2d, 
    ADMM_Conv2d, ADMM_Linear, ADMM, ADMM_Heaviside, ADMM_ReLU, ADMM_Metrics, ADMM_GAP
)
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = configparser.ConfigParser()

config.read('config/main.ini') 

seed = config.getint('config','seed')
#random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False

batch_size = config.getint('config', 'batch_size')
epochs = config.getint('config', 'epochs')
hidden_size = config.getint('config', 'hidden_size')

lr = config.getfloat('config', 'learning_rate')

rho= config.getfloat('config', 'rho')
beta= config.getfloat('config', 'beta')

input_size=784

class GDLinearNet(nn.Module):
    def __init__(self, input_size=input_size, hidden_size=hidden_size, num_classes=10):
        super(GDLinearNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

layers = nn.ModuleList([ADMM_Linear(in_f=input_size, out_f=hidden_size, h=ADMM_ReLU(), init='zeros', bias=True),
                            ADMM_Linear(in_f=hidden_size, out_f=10, h=ADMM_ReLU(), init='zeros', bias=True)])

admm_model = ADMM(layers, rho=rho, beta=beta, init='zeros', bias=True, train_method='vectorized').to(device)

# --- Data ---
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

images, labels = next(iter(dataloader))
images = images.view(images.size(0), -1)

# --- Setup ---
model = GDLinearNet(hidden_size=hidden_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

images, labels = images.to(device), labels.to(device)
labels_one_hot = F.one_hot(labels, num_classes=10).float()

gd_steps = []
gd_mses = []
gd_accs = []
# --- Training Loop ---
print("Training Linear model with gradient descent")
for epoch in range(epochs):
    outputs = model(images)
    loss = criterion(outputs, labels_one_hot)
    _, predictions = torch.max(outputs, 1)
    correct = (predictions == labels).sum().item()
    accuracy = (correct / labels.size(0))*100
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    gd_steps.append(epoch)
    gd_mses.append(loss.item())
    gd_accs.append(accuracy)
    
    print(f"Step {epoch + 1} | MSE: {loss.item():.4f} | Acc: {accuracy:.4f}")
    

m = ADMM_Metrics(admm_model) 
admm_model._init_states(images)
admm_steps = []
admm_mses = []
admm_accs = []

print("\nTraining ADMM model...")
for epoch in range(epochs + 1):
    # Fit using one-hot labels
    admm_model.fit(images, labels_one_hot, warming=False)             
    
    with torch.no_grad():
        raw_outputs, firing_rates = admm_model.forward_model(images)
        flat_outputs = raw_outputs.view(batch_size, -1) 
        _, predictions = torch.max(flat_outputs, 1)
        correct = (predictions == labels).sum().item()
        accuracy = (correct / labels.size(0))*100
        current_metrics = m.get_all_metrics(images, labels_one_hot)
            
        mse = current_metrics["mse"]
        admm_steps.append(epoch)
        admm_mses.append(mse)
        admm_accs.append(accuracy)
           
        print(f"Epoch [{epoch:3d}/{epochs}]" f"| MSE: {mse:.4f}" f"| Acc: {accuracy:6.2f}%")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(gd_steps, gd_mses, label='Gradient Descent', color='blue', linewidth=2)
plt.plot(admm_steps, admm_mses, label='ADMM', color='orange', linewidth=2)
plt.title('Mean Squared Error vs. Time')
plt.xlabel('Steps / Epochs')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, linestyle='dotted', alpha=0.7)

# Plot 2: Accuracy
plt.subplot(1, 2, 2)
plt.plot(gd_steps, gd_accs, label='Gradient Descent', color='blue', linewidth=2)
plt.plot(admm_steps, admm_accs, label='ADMM', color='orange', linewidth=2)
plt.title('Accuracy vs. Time')
plt.xlabel('Steps / Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot
plt.tight_layout()
plot_filename = "training_comparison.png"
plt.savefig(plot_filename, dpi=300)
print(f"Plot successfully saved as '{plot_filename}' in your current directory!")
plt.show()
