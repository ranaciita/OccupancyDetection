import os
from datetime import datetime
import cv2

class file_save_handler:
    
    # constructor
    def __init__(self):
        self.base_directory = "results"
        os.makedirs(self.base_directory, exist_ok=True)
        self.current_folder = None
        self.txt_file_path = None

    # method creates a new folder under /results/
    def create_new_folder(self, folder_name=None):
        if self.current_folder is None:
            if folder_name is None:
                folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")

            self.current_folder = os.path.join(self.base_directory, folder_name)
            os.makedirs(self.current_folder, exist_ok=True)

            print(f"Created new folder for logs: {self.current_folder}")
            self.__create_txt_file()

    # private method - creates a .txt file in the current folder
    def __create_txt_file(self):
        if not self.current_folder:
            raise RuntimeError("No folder created yet.")
        
        self.txt_file_path = os.path.join(self.current_folder, "log.txt")
        open(self.txt_file_path, "w").close()  # create empty log.txt

    # adds a line of text to the txt file
    def add_log_to_txt(self, text: str):
        if not self.txt_file_path:
            raise RuntimeError("No txt file created yet.")

        with open(self.txt_file_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    # adds an image to the current folder
    def add_image_log(self, frame, image_name: str):
        if not self.current_folder:
            raise RuntimeError("No folder created yet.")
        
        if not image_name.lower().endswith((".jpg", ".png", ".jpeg")):
            image_name += ".jpg"
        
        save_path = os.path.join(self.current_folder, image_name)
        cv2.imwrite(save_path, frame)
        print(f"Saved image: {save_path}")