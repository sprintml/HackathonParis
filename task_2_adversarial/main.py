import torch as th
import numpy as np

#Load the dataset here

dataset = th.load("natural_images.pt")

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

# Convert this to an npz file of perturbations
pt_to_npz(dataset, random_pt, npz_path="perturbations.npz")

# Use the generated .npz file to evaluate your results using sample_submission_task2.py
