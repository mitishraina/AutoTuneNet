import sys

from autotunenet.exceptions.exception import CustomException


def test_raise_outside_except_block():
    err = CustomException("boom", sys)
    assert "boom" in str(err)


def test_inside_except_block_reports_file_and_line():
    try:
        1 / 0
    except ZeroDivisionError as e:
        err = CustomException(e, sys)
    assert "test_custom_exception.py" in str(err)
    assert "line number" in str(err)
