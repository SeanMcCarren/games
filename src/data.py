import cv2
import numpy as np
from pathlib import Path
path = Path(__file__)
location = path.parent.parent.absolute() / 'data'
print(f"Data location: {location}")

def get_sudoku(n=None):
    dir = location / 'sudoku' / 'wichtounet'
    files = [x.stem for x in dir.glob('**/*') if x.is_file()]
    files = list(set(files))
    if n is None:
        n = len(files)
    for i in range(n):
        file = files[i]
        img = cv2.imread(str(dir / (file + '.jpg')))
        data = np.loadtxt(str(dir / (file + '.dat')), skiprows=2, dtype=int)
        assert data.shape == (9,9)
        yield img, data
