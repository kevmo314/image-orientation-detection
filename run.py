import torch
import torchvision
import torch.nn.functional as F
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(1, 3, 5)
        self.mp1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(3, 4, 5)
        self.mp2 = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(20, 20)
        self.fc2 = torch.nn.Linear(20, 5)
        self.fc3 = torch.nn.Linear(5, 2)
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x):
        x = self.mp1(F.relu(self.conv1(x)))
        x = self.mp2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class Images(torch.utils.data.Dataset):
    def __init__(self, root):
        self.imgs = [os.path.join(root, f) for f in os.listdir(root)]
        self.imgs = [i for i in self.imgs if os.path.isfile(i) and os.path.getsize(i) > 0]

    def __len__(self):
        return 2 * len(self.imgs)

    def __getitem__(self, index):
        if index < len(self.imgs):
            img = torchvision.io.read_image(self.imgs[index], torchvision.io.ImageReadMode.GRAY)
            return (img, torch.tensor(0))
        # flip the image
        img, cls = self[index - len(self)]
        img = torchvision.transforms.functional.hflip(img)
        img = torchvision.transforms.functional.vflip(img)
        return (img, cls + 1)

def test(model, root='./test'):
    correct = 0
    with torch.no_grad():
        images = Images(root)
        for i in range(len(images)):
            image, label = images[i]
            image, label = image.to(device), label.to(device)
            output = model(image.unsqueeze(0).float() / 256)
            if torch.squeeze(output).argmax() == label:
                correct += 1
    print('%s Accuracy: %d/%d' % (root, correct, len(images)))

def train(num_epochs, model):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    test(model, root='./test')

    for epoch in range(num_epochs):
        train_loader = torch.utils.data.DataLoader(Images('./train'), batch_size=128, shuffle=True, num_workers=4)
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images.float() / 256)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('[%d] loss: %.3f' % (epoch + 1, running_loss))
        torch.save(model, 'model.pt')
        if epoch % 10 == 0:
            test(model, root='./train')
            test(model, root='./test')

if __name__ == '__main__':
    train(1500, torch.load('model.pt').to(device))
