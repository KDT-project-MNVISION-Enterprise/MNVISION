#======================
# imports
#======================
import tkinter as tk
from tkinter import ttk

# Create instance
win = tk.Tk()   

# Add a title       
win.title("Python GUI")

# Adding a Label that will get modified
a_label = ttk.Label(win, text="A Label")
a_label.grid(column=0, row=0)

# Button Click Event Function
def click_me():
    action.configure(text="** I have been Clicked! **") #
    a_label.configure(foreground='red') # 글자색 지정?
    a_label.configure(text='A Red Label') # 눌렀을때 도출되는 문구!

# Adding a Button
action = ttk.Button(win, text="Click Me!", command=click_me) # 'CLick Me!' 버튼을 생성. 실제로 클릭하면 click_me 함수가 실행됨.  
action.grid(column=1, row=0) # 버튼 위치 조정

#======================
# Start GUI
#======================
win.mainloop()