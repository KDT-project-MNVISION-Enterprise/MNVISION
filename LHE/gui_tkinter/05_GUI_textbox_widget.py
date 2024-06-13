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
    action.configure(text="Hello " + name.get()) # name에 입력된 값을 가져와서 버튼을 누르면 Hello + name 도출.

# Changing out label
ttk.Label(win, text="Enter a name: ").grid(column=0, row=0)

# Adding a Text box Entry widget
name = tk.StringVar() # 입력받을 문자열 변수 생성
name_entered = ttk.Entry(win, width=12, textvariable=name) # 사이즈 하드 코딩 = 12 (고정)
name_entered.grid(column=0, row = 1)  # 텍스트 박스 위치 지정

# # Adding a Button
action = ttk.Button(win, text="Click Me!", command=click_me) # 'CLick Me!' 버튼을 생성. 실제로 클릭하면 click_me 함수가 실행됨.  
action.grid(column=1, row=1) # 버튼 위치 지정

#======================
# Start GUI
#======================
win.mainloop()