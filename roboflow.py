from inference_sdk import InferenceHTTPClient

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="AtpFozXxspWNOWzLJcd6"
)


image_path = r"C:\Users\sathy\yolo\data\test\images\images-34-_jpg.rf.b3fbbb65bfeb31013c1d05184ecb249b.jpg"


try:
    result = CLIENT.infer(image_path, model_id="defective-or-not-c9b4m/1")
    print("Prediction successful:")
    print(result)
    
    # Process predictions if available
    if 'predictions' in result:
        for pred in result['predictions']:
            print(f"Class: {pred['class']}, Confidence: {pred['confidence']:.2f}")
    else:
        print("No predictions found in result")
        
except Exception as e:
    print(f"Error during inference: {e}")

# Visualization code
try:
    import cv2
    import matplotlib.pyplot as plt

    # Load your image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw boxes using the 'result' variable
    if 'predictions' in result:
        for pred in result['predictions']:
            x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
            label = pred['class']
            confidence = pred['confidence']
            
            # Draw rectangle and label
            cv2.rectangle(img, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x - w//2, y - h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Show the image
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
except Exception as e:
    print(f"Error during visualization: {e}")


