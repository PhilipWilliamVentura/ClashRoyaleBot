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
        self.card1, self.card2, self.card3, self.card4 = self.img.capture_card_area("card1_img.png", "card2_img.png", "card3_img.png", "card4_img.png")

        self.client = InferenceHTTPClient(
            api_url="http://localhost:9001", # use local inference server
            api_key=API_KEY
        )

    def run_card_detection(self):
        cards = [self.card1, self.card2, self.card3, self.card4]
        clean_predictions = []
        for card in cards:
            result = self.client.run_workflow(
                workspace_name=WORKSPACE_NAME,
                workflow_id="custom-workflow",
                images={
                    "image": card
                }
            )

            workflow_output = result[0] if isinstance(result, list) else result
            predictions_block = workflow_output.get("predictions", {})
            raw_pred = predictions_block.get("predictions", [])

            for pred in raw_pred:
                clean_predictions.append(pred["class_id"])

        return clean_predictions

if __name__ == "__main__":
    x = CardDetection()
    print(x.run_card_detection())


