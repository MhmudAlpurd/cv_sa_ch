import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import ResNet50_Weights
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "./data"
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 1
LR = 1e-4
NUM_CLASSES = 3  # background, pet, border



class PetSegTransform:
    def __init__(self, img_size=256):
        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.mask_resize = transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.NEAREST
        )

    def __call__(self, image, mask):
        image = self.img_tf(image)

        mask = self.mask_resize(mask)
        mask = torch.as_tensor(
            torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
        ).view(mask.size[1], mask.size[0]).long()

        # Oxford Pet trimap values are usually:
        # 1 = pet, 2 = border, 3 = background
        # تبدیل به کلاس‌های 0,1,2
        mask = mask - 1

        return image, mask



train_dataset = OxfordIIITPet(
    root=DATA_DIR,
    split="trainval",
    target_types="segmentation",
    download=True,
    transforms=PetSegTransform(IMG_SIZE)
)

test_dataset = OxfordIIITPet(
    root=DATA_DIR,
    split="test",
    target_types="segmentation",
    download=True,
    transforms=PetSegTransform(IMG_SIZE)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)



model = deeplabv3_resnet50(
    weights=None,
    weights_backbone=ResNet50_Weights.DEFAULT,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)

#cuda ---> gpu ---> model + data ---> vram
#cpu ---> matplotlib , opencv ---> ram (x vram)



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)



def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for images, masks in tqdm(loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)["out"]

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)



def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

            total_loss += loss.item()

    return total_loss / len(loader)



for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, test_loader, criterion)

    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")



torch.save(model.state_dict(), "deeplabv3_pet_segmentation.pth")
print("Model saved.")


model.eval()

image, mask = test_dataset[0]
input_image = image.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(input_image)["out"]

pred = torch.argmax(output.squeeze(), dim=0).cpu()

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Ground Truth")
plt.imshow(mask)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Prediction")
plt.imshow(pred)
plt.axis("off")

plt.show()