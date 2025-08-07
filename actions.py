import pyautogui
import os
from datetime import datetime
import time
import platform

class Actions:
    def __init__(self):
        self.os_type = platform.system()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.images_folder = os.path.join(self.script_dir, 'main_images')


        if self.os_type == "Darwin":
            self.TOP_LEFT_X = 1072
            self.TOP_LEFT_Y = 158
            self.BOTTOM_RIGHT_X = 1476
            self.BOTTOM_RIGHT_Y =709
            self.FIELD_AREA = (self.TOP_LEFT_X, self.TOP_LEFT_Y, self.BOTTOM_RIGHT_X, self.BOTTOM_RIGHT_Y)

            self.WIDTH = self.BOTTOM_RIGHT_X - self.TOP_LEFT_X
            self.HEIGHT = self.BOTTOM_RIGHT_Y - self.TOP_LEFT_Y


        self.card_key = {
            "card1" : "1",
            "card2" : "2",
            "card3" : "3",
            "card4" : "4", 
            "emote" : "e"
        }

        # Card name to position mapping (will be updated during detection)
        self.current_card_positions = {}

    def capture_area(self, save_path):
        screenshot = pyautogui.screenshot(region=(self.TOP_LEFT_X, self.TOP_LEFT_Y, self.WIDTH, self.HEIGHT))
        screenshot.save(save_path)
        return save_path



    
    #play_again_x = 
    #play_again_y = 

    #top_bridge = 421
    #bot_bridge = 450