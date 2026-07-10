import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import ResNet50_Weights
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 256

NUM_CLASSES = 3


model = deeplabv3_resnet50(
    weights=None,
    weights_backbone=ResNet50_Weights.DEFAULT,
    num_classes=NUM_CLASSES
)


model.load_state_dict(
    torch.load(
        "deeplabv3_pet_segmentation.pth",
        map_location=DEVICE
    )
)


model = model.to(DEVICE)

model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


image = Image.open("cat.jpg").convert("RGB")

input_tensor = transform(image)
input_tensor = input_tensor.unsqueeze(0)



input_tensor = input_tensor.to(DEVICE)


with torch.no_grad():

    output = model(input_tensor)["out"]




prediction = torch.argmax(output, dim=1)



prediction = prediction.squeeze(0)

prediction = prediction.cpu().numpy()



plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(prediction)
plt.title("Predicted Mask")
plt.axis("off")

plt.show()