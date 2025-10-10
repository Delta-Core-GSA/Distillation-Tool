

class CustomLogger():
    
    def __init__(self, name, ext, subject=None):
        self.path = f"./logs/{name}.{ext}"
        self.log_file = open(self.path, "w") 
        self.name=name
        self.ext=ext
        if(subject != None):
            self.subject = subject
            self.log_file.write(f"Subejct: {subject}\n")
        self.log_file.close()


#
    def log_value(self,label=None, value=None):
        self.log_file = open(self.path, "a")
        try:
            if(label != None):
                self.log_file.write(f"{label}: {value}\n")
                print(f"{label}: {value}")
            else:
                self.log_file.write(f"{value}\n")
                print(f"{value}")
        except:
            print(f"[ERRORE NEL LOG DI]: {label}")
        self.log_file.close()
    
    def silent_log_value(self,label, value):
        self.log_file = open(self.path, "a")
        try:
            if(label != None):
                self.log_file.write(f"{label}: {value}\n")
            else:
                self.log_file.write(f"{value}\n")
        except:
            pass
        self.log_file.close()
        

    def changeSubject(self, newSubject):
        self.subject = newSubject
        self.log_file = open(self.path, "a")
        self.log_file.write(f"\n\nSubejct: {newSubject}\n")
        self.log_file.close()

