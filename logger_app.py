import logging

def logs():
    
    logging.basicConfig(filename=r"C:\Users\prash\Desktop\data\SMS_Spam_Ham_Classification\logging_file.log",
                        level=logging.INFO, 
                        format="%(asctime)s:%(levelname)s:%(message)s", 
                        datefmt="%d/%m/%Y %I:%M:%S %p")
    

logs()