import sys
import os
import struct
from PIL import Image, ImageDraw

input = sys.argv[1]
output = sys.argv[2]

image = Image.new("L", (64, 64), color=0)
pixels = image.load()

with open(input, "rb") as file:
    size = os.path.getsize(input)
    for i in range(0, min(size, image.size[0] * image.size[1])):
        y = int(i / image.size[0])
        x = i % image.size[0]
        byte = file.read(1)
        value = struct.unpack('B', byte)[0]
        pixels[x, y] = value

image.save(output)
