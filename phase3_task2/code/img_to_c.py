import cv2
import numpy as np

img = cv2.imread('test.png')

if img is None:
    print("ERROR: Image not found!")
    exit()

img = cv2.resize(img, (128, 128))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

with open('test_image.h', 'w') as fout:
    print('#define STATIC_IMAGE_NAME "test"', file=fout)
    print('static const uint8_t test_image[] = {', file=fout)
    img.tofile(fout, ', ', '0x%02X')
    print('};\n', file=fout)

print("✅ test_image.h created")
