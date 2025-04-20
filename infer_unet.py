import os
from PIL import Image
import torch
from torchvision import transforms
from unet_model import UNetClassifier

# === Config ===
inf_dir = "/home/bay/codes/unet/xray/inf"
classes = ['normal', 'pneumonia']  # Adjust based on your dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model ===
model = UNetClassifier(in_channels=3, num_classes=2)
model.load_state_dict(torch.load("unet_classifier.pth", map_location=device))
model.to(device)
model.eval()

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
])

# === Loop through images in folder ===
print(f"\n[ Predicting on images in: {inf_dir} ]\n")
predictions = []

with torch.no_grad():
    for filename in sorted(os.listdir(inf_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(inf_dir, filename)

            try:
                image = Image.open(filepath).convert("RGB")
                image = transform(image).unsqueeze(0).to(device)

                output = model(image)
                _, pred = torch.max(output, 1)
                predicted_class = classes[pred.item()]

                predictions.append((filename, predicted_class))
                print(f"{filename} --> {predicted_class}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# === Save predictions to a text file ===
output_path = os.path.join(inf_dir, "predictions.txt")
with open(output_path, "w") as f:
    for filename, pred_class in predictions:
        f.write(f"{filename}\t{pred_class}\n")

print(f"\nPredictions saved to: {output_path}")
