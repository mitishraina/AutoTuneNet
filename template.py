import os
from pathlib import Path
import logging 

logging.basicConfig(level=logging.INFO)

# project_name = "AutoTuneNet"

list_of_files=[
    f"src/__init__.py",
    f"src/core/__init__.py",
    f"src/core/optimizer.py",
    f"src/core/parameters.py",
    f"src/core/metrics.py",
    f"src/core/optimizer_test.py",
    f"src/core/sandbox_test.py",
    f"src/adapters/__int__.py",
    f"src/adapters/pytorch.py",
    f"src/logging/__init__.py",
    f"src/logging/logger.py",
    f"src/exceptions/__init__.py",
    f"src/exceptions/exception.py",
    f"src/examples/__init__.py",
    f"src/utils.py",
    "app.py",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir}")
        
    if(not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as fp:
            pass
        logging.info(f"Created file: {filepath}")
    else:
        logging.info(f"{filename} already exists.")