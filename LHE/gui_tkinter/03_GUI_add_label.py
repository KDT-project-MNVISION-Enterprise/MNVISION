# ===============================================================
# imports
# ===============================================================
import tkinter as tk
from tkinter import ttk # tkinter패키지의 ttk 모듈 import
#ttk = themed(특정테마의) tk

# create instance
win = tk.Tk() # 인스턴스 생성

# Add a title
win.title("My GUI")

# Adding a Label
ttk.Label(win, text = "A label").grid(column=0, row=0)# grid 안 쓰니까 출력 안 됨:) 뭔지는 아직 모르겠음.


# ===============================================================
# Start GUI
# ===============================================================
win.mainloop()