import torch as th
import torchvision.models as models
import requests
import numpy as np

model = models.resnet18(weights=None)
model.conv1 = th.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=0, bias=False)
model.fc = th.nn.Sequential(
    th.nn.Dropout(p=0.16),
    th.nn.Linear(model.fc.in_features, 9)
)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

model.load_state_dict(th.load("classifier.pt", map_location="cpu"))
model.to(device)

dataset = th.load("/sprint2/adam/natural_images.pt")x

# Example of conversion from pt containing adversarial examples to npz file containing perturbations

def pt_to_npz(base_data, part_data, npz_path):
    base_images = base_data["images"]
    part_images = part_data["images"]

    if not isinstance(base_images, th.Tensor) or not isinstance(part_images, th.Tensor):
        raise TypeError("base_data['images'] and part_data['images'] must be torch.Tensor")

    part_length = len(part_images)
    cut_base = base_images[:part_length]
    perturbations = (part_images - cut_base).cpu().numpy().astype(np.float16)

    if perturbations.dtype == np.object_:
        raise ValueError("Perturbations array has dtype=object, which is unsafe to save.")
    
    np.savez_compressed(npz_path, perturbations=perturbations, allow_pickle=False)

# Generate a pt file with random images and labels

random_labels = th.randint(0, 9, (len(dataset['images']), 1))
random_images = dataset['images'] + th.randn_like(dataset['images']) * 0.1
random_pt = {'images': random_images}

# Convert this to an npz file

pt_to_npz(dataset, random_pt, npz_path="perturbations.npz")


TOKEN = "efd51e3ccdb1fc0088748f5735a263b8"

th.save({'images': dataset["images"], 'labels': dataset["labels"]}, "test.pt")

response = requests.post("http://34.122.51.94:9000/10-adversarial-examples", files={"file": open("test.pt", "rb")}, headers={"token": TOKEN})
print(response.json())