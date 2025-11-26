from ultralytics import YOLO
import os

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"

def test_inference():
    """
    Loads a fine-tuned YOLO model and runs inference on a sample video
    to verify detection capabilities.
    """
    # Path to the fine-tuned model weights
    # Ensure 'best.pt' from training has been renamed/moved to this location
    model_path = 'models/yolo_garbage_colab_v2.pt' 
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        print("Please copy 'best.pt' from the runs folder to the 'models' directory.")
        return

    print(f"Loading model: {model_path}...")
    model = YOLO(model_path)

    # Path to the input video source for testing
    # Replace with a valid video path from the validation set
    source = 'data/videos/vid0071.mp4' 

    if not os.path.exists(source):
        print(f"Error: Source file not found at {source}")
        return

    print(f"Starting inference on: {source}")

   
    try:
        model.predict(
            source=source, 
            save=True, 
            conf=0.25, 
            project='results', 
            name='test_run'    
        )
        print("\nInference completed successfully.")
        print("Output saved to directory: results/test_run/")
        
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == '__main__':
    test_inference()