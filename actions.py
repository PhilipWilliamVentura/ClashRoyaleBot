import pyautogui
import os
from datetime import datetime
import time
import platform

class Actions:
    def __init__(self):
        self.os_type = platform.system()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.elixer_folder = "elixir_screenshots"


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

    def count_elixir(self):
        elixir_count = 0
        for i in range(9,0,-1):
            full_filepath = os.path.join(self.elixer_folder, f"{i}elixir.png")
            try:
                location = pyautogui.locateOnScreen(full_filepath, confidence=0.75, grayscale=True)
                if location:
                    elixir_count = i
                    print("Elixir found:", elixir_count)
                    return elixir_count
            except pyautogui.ImageNotFoundException:
                continue
        return 0

    def start_game(self):
        pyautogui.moveTo(1450, 400, duration=1)
        pyautogui.click()
        while True:
            try:
                location = pyautogui.locateOnScreen("Battle-button.png", confidence=0.8)
            except pyautogui.ImageNotFoundException:
                location = None
            if location:
                x, y = pyautogui.center(location)
                print("Button appeared! Moving to:", x, y)
                pyautogui.moveTo(x / 2, y / 2, duration=0.5)  # adjust for Retina scaling if needed
                pyautogui.click()
                location = None
                break  # Exit the loop after clicking the button
            else:
                # No button found — continue normally, let user control mouse
                pyautogui.click(1270, 734)
                time.sleep(0.5)
    
    def play_again(self):
        pyautogui.moveTo(1348, 842, duration=1)  # chang coords for play again button
        pyautogui.click()  # click the mouse
        time.sleep(1)
        self.start_game() #if doing the OK button




    #play_again_x = 
    #play_again_y = 

    #top_bridge = 421
    #bot_bridge = 450