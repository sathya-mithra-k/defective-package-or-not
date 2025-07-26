from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

#image_path = r"C:\Users\sathy\yolo\data\test\images\images-15-_jpg.rf.9dc1da7054ed8d3b6448159c1c86174f.jpg"
image_path = r"C:\Users\sathy\yolo\data\test\images\istockphoto-615247940-612x612_jpg.rf.e8bbacee97800eac0407521d587c3c39.jpg"

try:
    result = CLIENT.infer(image_path, model_id="defective-or-not-c9b4m/1")
    print("Prediction successful:")
    print(result)
    
    if 'predictions' in result:
        for pred in result['predictions']:
            print(f"Class: {pred['class']}, Confidence: {pred['confidence']:.2f}")
    else:
        print("No predictions found in result")
        
except Exception as e:
    print(f"Error during inference: {e}")

try:
    import cv2
    import matplotlib.pyplot as plt


    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if 'predictions' in result:
        for pred in result['predictions']:
            x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
            label = pred['class']
            confidence = pred['confidence']
            
           
            cv2.rectangle(img, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x - w//2, y - h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
except Exception as e:
    print(f"Error during visualization: {e}")


