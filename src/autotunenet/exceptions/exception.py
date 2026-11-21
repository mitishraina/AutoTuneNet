import sys


def error_message_detail(error, error_details=sys) -> str:
    """
    Describe `error` with the file and line of the exception being handled.

    Outside an `except` block there is no traceback, so only the message is
    returned (previously this raised AttributeError).
    """
    _, _, exc_tb = error_details.exc_info()
    if exc_tb is None:
        return f"Error occurred: [{error}]"
    while exc_tb.tb_next is not None:
        exc_tb = exc_tb.tb_next
    file_name = exc_tb.tb_frame.f_code.co_filename
    return "Error occurred in script:[{0}] line number:[{1}] error.message:[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )


class CustomException(Exception):
    def __init__(self, error_message, error_details=sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_details)

    def __str__(self):
        return self.error_message
