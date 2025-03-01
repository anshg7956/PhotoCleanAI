import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import shutil
from concurrent.futures import ThreadPoolExecutor

def get_image_hash(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
    mean = resized.mean()
    hash_value = (resized > mean).astype(np.uint8)
    return hash_value

def hamming_distance(hash1, hash2):
    return np.count_nonzero(hash1 != hash2)

def calculate_metrics(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sharpness (Laplacian variance)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Brightness and Contrast
    mean_brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Color Balance (Standard deviation of color channels)
    color_balance = np.std(cv2.mean(image)[:3])
    
    # Exposure (Histogram analysis)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    overexposed = np.sum(hist[-10:])
    underexposed = np.sum(hist[:10])
    
    # Advanced Noise Detection
    denoised = cv2.fastNlMeansDenoising(gray, None, h=30, templateWindowSize=7, searchWindowSize=21)
    noise = np.std(gray - denoised)
    
    return sharpness, mean_brightness, contrast, color_balance, overexposed, underexposed, noise

def detect_smile(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        if len(smiles) > 0:
            return True
    return False

def score_image(image_path):
    image = cv2.imread(image_path)
    sharpness, brightness, contrast, color_balance, overexposed, underexposed, noise = calculate_metrics(image)
    
    # Score based on the metrics
    sharpness_score = sharpness / 1000
    exposure_score = (1 / (overexposed + 1)) + (1 / (underexposed + 1))
    contrast_score = contrast - noise
    color_balance_score = color_balance
    smile_score = 1 if detect_smile(image) else 0
    
    score = (
        0.4 * sharpness_score +
        0.2 * exposure_score +
        0.1 * contrast_score +
        0.1 * color_balance_score +
        0.1 * smile_score
    )
    
    return score

def group_similar_images(image_paths, distance_threshold=7):
    hashes = []
    for path in image_paths:
        image_hash = get_image_hash(path)
        hashes.append(image_hash)

    # Calculate pairwise distances
    distance_matrix = np.zeros((len(hashes), len(hashes)), dtype=int)
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            distance = hamming_distance(hashes[i], hashes[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    # Perform clustering
    db = DBSCAN(eps=distance_threshold, min_samples=2, metric='precomputed')
    labels = db.fit_predict(distance_matrix)

    # Group images by cluster
    grouped_images = defaultdict(list)
    for idx, label in enumerate(labels):
        grouped_images[label].append(image_paths[idx])

    return grouped_images

def find_best_in_cluster(image_paths):
    with ThreadPoolExecutor() as executor:
        scores = list(executor.map(score_image, image_paths))
    best_image_idx = np.argmax(scores)
    return image_paths[best_image_idx]

def organize_images_into_folders(grouped_images, base_dir):
    for group, images in grouped_images.items():
        if group != -1:  # Exclude noise points labeled as -1 by DBSCAN
            # Create a subfolder for each group
            group_folder = os.path.join(base_dir, f"group_{group}")
            os.makedirs(group_folder, exist_ok=True)

            # Find the best image in the cluster
            best_image = find_best_in_cluster(images)

            # Move images to the group folder
            for img in images:
                img_name = os.path.basename(img)
                if img == best_image:
                    # Prefix best image filename with "BEST_"
                    new_name = f"BEST_{img_name}"
                else:
                    new_name = img_name

                new_path = os.path.join(group_folder, new_name)
                shutil.move(img, new_path)

def optimize_image_processing(image_dir):
    # Gather all image paths
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.jpg', '.jpeg', '.png', '.heic'))]

    # Step 1: Group similar images into clusters
    grouped_images = group_similar_images(image_paths)

    # Step 2: Organize images into folders, with the best image in each cluster marked
    organize_images_into_folders(grouped_images, image_dir)

# Example usage:
image_dir = '/Users/ngupta2/Desktop/PhoneImages'
optimize_image_processing(image_dir)
