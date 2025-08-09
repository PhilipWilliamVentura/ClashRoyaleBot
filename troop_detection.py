import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from actions import Actions

# Load environment variables
load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")

class TroopDetection:
    def __init__(self):
        self.img = Actions()
        self.client = InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key=API_KEY
        )

    def run_troop_detection(self, image_name="troops_img.png"):
        # Capture screen region and get image path
        img_path = self.img.capture_area(image_name)

        # Run workflow on captured image
        result = self.client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id="detect-count-and-visualize",
            images={"image": img_path}
        )

        # Handle output format
        workflow_output = result[0] if isinstance(result, list) else result
        predictions_block = workflow_output.get("predictions", {})
        raw_preds = predictions_block.get("predictions", [])

        # Clean and return predictions
        allies = []
        enemies = []
        for pred in raw_preds:
            cx, cy = pred["x"], pred["y"]
            if pred["class"].startswith("ally"):
                allies.append((cx, cy))
            else:
                enemies.append((cx, cy))

        return allies, enemies

if __name__ == "__main__":
    x = TroopDetection()
    allies, enemies = x.run_troop_detection()
    print("Allies: ", allies)
    print("Enemies: ", enemies)
