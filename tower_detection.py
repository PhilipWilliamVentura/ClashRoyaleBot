"""
Ally tower -> EX: -10 points
Enemy tower -> EX: +10 points
Win -> +25 points
Loss -> -25 points
"""

import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from actions import Actions

# Load environment variables
load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")

class TowerDetection:
    def __init__(self):
        self.img = Actions()
        self.client = InferenceHTTPClient(
            api_url="http://localhost:9001", # use local inference server
            api_key=API_KEY
        )

    def run_tower_detection(self, image_name="tower_img.png"):
        img_path = self.img.capture_area(image_name)

        result = self.client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id="custom-workflow-2",
            images={
                "image": img_path
            }
        )

        workflow_output = result[0] if isinstance(result, list) else result
        predictions_block = workflow_output.get("predictions", {})
        raw_preds = predictions_block.get("predictions", [])

        ally_tower = 0
        enemy_tower = 0
        for pred in raw_preds:
            if pred["class"].startswith("ally"):
                ally_tower += 1
            else:
                enemy_tower += 1

        return ally_tower, enemy_tower
    
x = TowerDetection()
print(x.run_tower_detection())