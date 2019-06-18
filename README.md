# Faces & Flowers Generator

Artificial faces and flowers generator using DCGANs

## Getting Started

Download source from github


```
git clone https://github.com/kaszperro/faces-generator.git
```

### Prerequisites

We are using pipenv for downloading dependencies, inside project folder just run:

```
pipenv install
```

## Usage

To open main application and start having fun run:

```
pipenv run python3 GeneratorApp.py
```

### Additional modules

#### GifMaker

We have implemented also option for generating GIF from images of network generation, which is under path "./trained/{faces|flowers}/generated/gifs/"

Normally we used PyCharm console to run this, but we implemented small console menu for making it easier to use:

```
pipenv run python3 GifMaker.py
```

#### ImageComparer

We are able to search for similar images using this module and view results comparison - also with implemented small console menu. We are saving in total 15 found images. To run:

```
pipenv run python3 ImageComparer.py
```

**WARNING - Last results will be overwritten after running searching module again**

Everything connected with this module is under folder "./search/":

* "./search/data" - Data about found images (vector used to generate, parameters of MSE and SSIM)
* "./search/img" - Found images, used for showing comparison
