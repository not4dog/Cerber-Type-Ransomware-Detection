from tkinter import *
from tkinter import filedialog
 
main = Tk()
main.title('ML Ransomware Detection')
main.geometry("800x600")

 
def open():
    main.filename = filedialog.askopenfilename(initialdir='', title='.exe File Select', filetypes=(
    ('exe files', '*.exe'),('all files', '*.*'))) 
 
button1 = Button(main, text='.exe File Upload', command=open).pack()
 
main.mainloop()
