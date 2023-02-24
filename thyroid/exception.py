import sys, os 

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info() # extract the information about the error 
    file_name = exc_tb.tb_frame.f_code.co_filename # get the file name whrere the error occured 
    # formating the error msg 
    error_message = f"Error occured python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    return error_message


# create a custom Exception class that inherits the from built-in Exception class 
class ThyroidException(Exception):

    def __init__(self, error_message, error_detail:sys):
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
        
