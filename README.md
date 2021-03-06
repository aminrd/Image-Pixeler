# Image-Pixeler
Image Pixeler is a platform to replace pixels, or group of pixels in an image, with other images. This program is written in Python 3 mainly based on K-Means and OpenCV tools. You can see how to use this app using:

```{r, engine='bash', count_lines}
python main.py -h
```

Possible commands are: 
```{r, engine='bash', count_lines}
usage: main.py [-h] [-in INPUT] [-out OUTPUT] [-g GALLERY] [-s SCALE]
               [--grey GREY] [-r RECURSIVE] [-v VERBOSE]

optional arguments:
  -h, --help            show this help message and exit
  -in INPUT, --input INPUT
  -out OUTPUT, --output OUTPUT
                        Name of output file
  -g GALLERY, --gallery GALLERY
  -s SCALE, --scale SCALE
                        Output image will be scaled up by this factor [H *
                        $(scale), W * $(scale)]
  --grey GREY           Work in Greyscale mode
  -r RECURSIVE, --recursive RECURSIVE
                        Scan all images inside Gallery directory recursively
  -v VERBOSE, --verbose VERBOSE
                        Print steps in system log
```

All dependencies for this program are highlighted in requirements.txt. You can install dependencies using:   
```{r, engine='bash', count_lines} 
pip3 install -r requirements.txt
```

Examples usage: 
```{r, engine='bash', count_lines} 
python main.py -in ./input.jpg -out ./output.jpg -g ./Gallery/ -r True -s 256 -v True 
```
This command, loads input.jpg and save the art image to output.jpg. It scans all files and folders inside ./Gallery/ recursively. Those scanned images are going to replace patches in the main image. It makes all scanned images into square images automatically, so you don't have to take care of that part. It also prints the process steps in system log. Argument `-s 256` lets this program to scale up each patch of 8x8 pixels in the main image to be scaled up to 256x256 pixel. Some sample input and output:

![Output1](./Portfolio/Portfolio1.jpg)

![Output2](./Portfolio/Portfolio2.jpg)

## Video to Gallery
If you don't have multiple images, and you want to capture many frames from the video and at the same time, you want to preserve the diversity of those captured images, you can use `gallery_generator.py`. 

```shell
usage: gallery_generator.py [-h] [-v VIDEO] [-o OUTPUT] [-di DIVERSITY]

optional arguments:
  -h, --help            show this help message and exit
  -v VIDEO, --video VIDEO
                        Path to the file or a publicly available link to video
  -o OUTPUT, --output OUTPUT
                        Output directory to save frames selected diversely
  -di DIVERSITY, --diversity DIVERSITY
                        Diversity index showing how diverse the frames should
                        be [0-1] lowest 0 and highest 1. Suggested value =
                        0.25

``` 
![VideoGallery](./Portfolio/video_merged.jpg)

## Author
* [**Amin Aghaee**](https://github.com/aminrd/) 

 

