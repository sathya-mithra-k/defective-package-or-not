import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import os

def compute_sharpness(image_path):
    gray = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
    return np.var(cv2.Laplacian(gray, cv2.CV_64F))

def compute_entropy(image_path):
    gray = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entr = -np.sum(hist * np.log2(hist + 1e-10))
    return entr

def neutrosophic_decision(image_path, alpha=0.5, sharpness_scale=500, entropy_scale=8):
    T = compute_sharpness(image_path) / sharpness_scale
    F = 1 - T
    I = compute_entropy(image_path) / entropy_scale
    score = T - F - alpha * I
    return score

def roboflow_inference(image_path, client, model_id):
    """Run Roboflow API inference on an image using the official SDK"""
    try:
        result = client.infer(image_path, model_id=model_id)
        return result
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

# Initialize the Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="AtpFozXxspWNOWzLJcd6"
)

# Configuration
model_id = "defective-or-not-c9b4m/1"
threshold = -2  # Neutrosophic threshold
image_folder = r"C:\Users\sathy\yolo\data\test\images"

# Get all images
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]

print("Image Name\t\tScore\t\tStatus\t\tRoboflow Result")
print("-" * 80)

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    score = neutrosophic_decision(img_path)
    status = "PASS" if score >= threshold else "FAIL"
    
    # Only run Roboflow inference on PASS images
    roboflow_result = "N/A"
    if status == "PASS":
        result = roboflow_inference(img_path, CLIENT, model_id)
        if result and 'predictions' in result:
            predictions = result['predictions']
            if predictions:
                # Get the prediction with highest confidence
                best_pred = max(predictions, key=lambda x: x['confidence'])
                roboflow_result = f"{best_pred['class']} ({best_pred['confidence']:.2f})"
            else:
                roboflow_result = "No defects detected"
        else:
            roboflow_result = "API Error"
    
    print(f"{img_file}\t{score:.3f}\t\t{status}\t\t{roboflow_result}")

print(f"\nProcessing complete!") 