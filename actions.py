import pyautogui
import os
from datetime import datetime
import time
import platform

class Actions:
    def __init__(self):
        self.os_type = platform.system()
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.main_image_folder = "main_screenshots"
        self.elixer_folder = "elixir_screenshots"
        self.card_folder = "card_screenshots"


        if self.os_type == "Darwin":
            self.TOP_LEFT_X = 1035
            self.TOP_LEFT_Y = 150
            self.BOTTOM_RIGHT_X = 1445
            self.BOTTOM_RIGHT_Y =715
            self.BRIDGE_HEIGHT = 20
            self.FIELD_AREA = (self.TOP_LEFT_X, self.TOP_LEFT_Y, self.BOTTOM_RIGHT_X, self.BOTTOM_RIGHT_Y)

            self.WIDTH = self.BOTTOM_RIGHT_X - self.TOP_LEFT_X
            self.HEIGHT = self.BOTTOM_RIGHT_Y - self.TOP_LEFT_Y

            self.CARD_TOP_LEFT_X = 1100
            self.CARD_TOP_LEFT_Y = 770
            self.CARD_BAR_WIDTH = 1469 - 1100
            self.CARD_BAR_HEIGHT = 886 - 770
            self.CARD_WIDTH = self.CARD_BAR_WIDTH//4


        self.card_key = {
            0 : "1",
            1 : "2",
            2 : "3",
            3 : "4"
        }

    def capture_area(self, save_path):
        full = os.path.join(self.main_image_folder, save_path)
        screenshot = pyautogui.screenshot(region=(self.TOP_LEFT_X, self.TOP_LEFT_Y, self.WIDTH, self.HEIGHT))
        screenshot.save(full)
        return full
    
    def capture_card_area(self, path_card1, path_card2, path_card3, path_card4):
        full1 = os.path.join(self.card_folder, path_card1)
        full2 = os.path.join(self.card_folder, path_card2)
        full3 = os.path.join(self.card_folder, path_card3)
        full4 = os.path.join(self.card_folder, path_card4)

        pyautogui.screenshot(region=(self.CARD_TOP_LEFT_X, self.CARD_TOP_LEFT_Y, self.CARD_WIDTH, self.CARD_BAR_HEIGHT)).save(full1)
        pyautogui.screenshot(region=(self.CARD_TOP_LEFT_X + self.CARD_WIDTH, self.CARD_TOP_LEFT_Y, self.CARD_WIDTH, self.CARD_BAR_HEIGHT)).save(full2)
        pyautogui.screenshot(region=(self.CARD_TOP_LEFT_X + 2*self.CARD_WIDTH, self.CARD_TOP_LEFT_Y, self.CARD_WIDTH, self.CARD_BAR_HEIGHT)).save(full3)
        pyautogui.screenshot(region=(self.CARD_TOP_LEFT_X + 3*self.CARD_WIDTH, self.CARD_TOP_LEFT_Y, self.CARD_WIDTH, self.CARD_BAR_HEIGHT)).save(full4)

        return full1, full2, full3, full4

    def count_elixir(self):
        elixir_count = 0
        for i in range(9,0,-1):
            full_filepath = os.path.join(self.elixer_folder, f"{i}elixir.png")
            try:
                location = pyautogui.locateOnScreen(full_filepath, confidence=0.75, grayscale=True)
                if location:
                    elixir_count = i
                    return elixir_count
            except pyautogui.ImageNotFoundException:
                continue
        return 0

    def start_game(self):
        pyautogui.moveTo(1430, 500, duration=0.5)
        pyautogui.click()
        battlebutton = os.path.join(self.main_image_folder, "Battle-button.png")
        while True:
            try:
                location = pyautogui.locateOnScreen(battlebutton, confidence=0.8)
            except pyautogui.ImageNotFoundException:
                location = None
            if location:
                x, y = pyautogui.center(location)
                print("Button appeared! Moving to:", x, y)
                pyautogui.moveTo(x / 2, y / 2, duration=0.5)  # adjust for Retina scaling if needed
                pyautogui.click()
                location = None
                break
            else:
                # No button found — continue normally, let user control mouse
                pyautogui.click(1430, 500)
                time.sleep(0.5)
    
    def play_card(self, x, y, card_ind):
        if card_ind in self.card_key:
            pyautogui.press(self.card_key[card_ind])
            pyautogui.moveTo(x, y, duration=0.1)
            pyautogui.click()
        else:
            print("Not valid card index")
    
    def play_again(self):
        pyautogui.press("1")  # Change to 1 for play again button
        time.sleep(0.5)
        #self.start_game() #if doing the OK button

    def detect_winner(self):
        winner_screenshot = os.path.join(self.main_image_folder, "Winner.png")

        try:
            winner_location = pyautogui.locateOnScreen(winner_screenshot, confidence=0.8, grayscale=True)
        except pyautogui.ImageNotFoundException:
            winner_location = None
            
        if winner_location:
            _, y = pyautogui.center(winner_location)
            result = "victory" if y > 350 else "defeat"
            return result
        else:
            return None
        

    def start_friendly_match(self):
        pyautogui.moveTo(1430, 500, duration=0.5)
        pyautogui.click()
        friends_button = os.path.join(self.main_image_folder, "Friends.png")
        friendly_match_button = os.path.join(self.main_image_folder, "Friendly-battle.png")
        while True:
            try:
                location = pyautogui.locateOnScreen(friends_button, confidence=0.8)
            except pyautogui.ImageNotFoundException:
                location = None
            if location:
                x, y = pyautogui.center(location)
                print("Button appeared! Moving to:", x, y)
                pyautogui.moveTo(x / 2, y / 2, duration=0.5)  # adjust for Retina scaling if needed
                pyautogui.click()
                location = None
                break
            else:
                # No button found — continue normally, let user control mouse
                pyautogui.click(1430, 500)
                time.sleep(0.5)
        while True:
            try:
                location = pyautogui.locateOnScreen(friendly_match_button, confidence=0.8)
            except pyautogui.ImageNotFoundException:
                location = None
            if location:
                x, y = pyautogui.center(location)
                print("Button appeared! Moving to:", x, y)
                pyautogui.moveTo(x / 2, y / 2, duration=0.5)  # adjust for Retina scaling if needed
                pyautogui.click()
                location = None
                break
            else:
                # No button found — continue normally, let user control mouse
                pyautogui.click(1430, 215)
                time.sleep(0.5)




    #play_again_x = 
    #play_again_y = 


