# Epicycles

In short, produces animations of arbitrary shapes being drawn from Fourier Epicycles.
The shapes are provided as svg line-art as illustrated in example *drawing-1.svg*. A working `python 3` installation is required to use this script, and installing pip is recommended. Instructions for those who need it are included at the end of this README.

## Usage:
```
python Epicycle_Draw_Script.py infile N outfile
```
## Arguments:
```
Required:
> infile: path to input svg file

Optional:
> N: int maximum number of points to sample along curves (default 200)
> outfile: name of output file (with extension, use mp4)
```
If no output file name is given, the outfile will be named based on the infile. The parameter `N` determines the number of points sampled along the longest curve in the input SVG. See below section on SVG Lineart for more info.

## Dependencies:
Listed in 'requirements.txt'. Can be satisfied automatically using pip via:
```bash
pip install -r requirements.txt
```

# SVG LineArt
The easiest way to understand the required input format is to open 'drawing-1.svg' in Inkscape (or other appropriate vector graphics software). The script parses SVG paths, discretises them, and then computes a discrete Fourier transform (DFT). The SVG file may contain any number of paths of any length.

To create 'drawing-1.svg', and other valid inputs to this script, I simply trace Bezier curves around a reference image in Inkscape. The curves do not have to be closed loops, nor do they have to connect to one another. This can be done easily with a mouse.

# Installation and Troubleshooting
### (Mainly for those unfamiliar with running python scripts or interacting with git repos)
You can download this code by clicking the green `Code` button in the top-right and clicking `Download ZIP`. Extract this `.zip` file somewhere easily accessible.

## Windows:
There is now a python package available in the Microsoft Store. This will be the easiest way to get a working python installation with pip on Windows for those unfamiliar. Once installed, navigate to wherever you unpacked the ZIP file from earlier and open a command prompt here. You should now be able to run the command listed above in the `Dependencies` subsection to complete setup. From here simply follow the `Usage` instructions.
