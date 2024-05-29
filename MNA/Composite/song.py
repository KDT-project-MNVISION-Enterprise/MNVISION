import pygame
import time
import threading

def play_music(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(1)

mp3_file = "Program/Audio/alarm_bell.mp3"
music_thread = threading.Thread(target=play_music, args=(mp3_file,))
music_thread.start()
print("Music is playing in a separate thread.")
