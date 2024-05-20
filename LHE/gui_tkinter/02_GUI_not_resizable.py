# ===============================================================
# imports
# ===============================================================
import tkinter as tk

# create instance
win = tk.Tk() # 인스턴스 생성

# Add a title
win.title("My GUI")

# disable resizing the GUI by passing in False/False
#win.resizable(False, False) # 가로 세로 사이즈 변경 차단

# Enable resizing x-dimension, disable y-dimension
#win.resizable(True, False) # 가로 변경 가능, 세로 변경 불가능.

# Enable resizing x-dimension, disable y-dimension
win.resizable(False, True) # 가로 변경 불가능, 세로 변경 가능.

# ===============================================================
# Start GUI
# ===============================================================
win.mainloop()