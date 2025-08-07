import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from actions import Actions

load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")

class CardDetection:
    def __init__(self):
        self.img = Actions()
        self.img_path = self.img.capture_card_area("card_img.png")

        self.client = InferenceHTTPClient(
            api_url="http://localhost:9001", # use local inference server
            api_key=API_KEY
        )

    def run_card_detection(self):
        result = self.client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id="custom-workflow",
            images={
                "image": self.img_path
            }
        )

        # Handle output format
        workflow_output = result[0] if isinstance(result, list) else result
        count_objects = workflow_output.get("count_objects", 0)
        predictions_block = workflow_output.get("predictions", {})
        raw_preds = predictions_block.get("predictions", [])

        # Clean and return predictions
        clean_predictions = []
        for pred in raw_preds:
            clean_predictions.append({
                "class_name": pred.get("class") or pred.get("class_name"),
                "confidence": round(pred["confidence"], 4),
            })

        return clean_predictions, count_objects

x = CardDetection()
predictions, count = x.run_card_detection()
print(predictions)
print(count)

