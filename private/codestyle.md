# Pure Python Code Style

## Class

* Upper camel case

* Example: `class PolyCrystal():`

## Function

* Lower camel case

* Example: `def calcOrientationGradient()`

## Variables

* Snake case

* Example: `reconstruction_grid_size = (2048, 2048)`

# GPU (Pycuda) Code Style

Use single letter prefix to indicate the data type when trasnfered to GPU memory.

| Prefix   | Data type in GPU | Note |
| :------: | ----------------: | :---- |
|  a  | array   |                  |
|  f  | float32 | single precision |
|  i  | int32   | single precision |
