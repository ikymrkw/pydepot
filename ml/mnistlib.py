import numpy as np
from itertools import zip_longest
import matplotlib.pyplot as plt

def imshow(ax, image, vmin=None, vmax=None):
    vmin0, vmax0 = image.min, image.max
    if np.all(image>=0.0) and np.all(image<=1.0):
        vmin0, vmax0 = 0.0, 1.0
    elif np.all(image>=0) and np.all(image<=255):
        vmin0, vmax0 = 0.0, 255.0
    else:
        pass
    if vmin==None: vmin=vmin0
    if vmax==None: vmax=vmax0
    ax.imshow(image.reshape((28, 28)), cmap='gray', vmin=vmin, vmax=vmax)

def imshow_n(width, height, images, titles=None, vmin=None, vmax=None):
    """
    Show MNIST 28x28 grayscale images in a ``width``x``height`` grid array.
    You may have to call plt.show() since this function does not call it.
    
    Only the first ``width``*``height`` elements from ``images`` are plotted;
    excess are ignored. If ``images`` does not contain enough elements,
    extra grid cells are remain unplotted.
    
    ``titles`` is treated as follows:
    - If it is str, int, or float, the value will be set as titles for all grid cells.
    - If it is iterable (other than str), each corresponding value from it is set
        as a title for each grid cell (extras are ignored, and lackings mean no title).
        The return value from iterable should be 2-tuple/2-list or a single value:
          - If it is a tuple or a list, its first element is used as a title, and
            the second element must be a dict to be used as kwargs for set_title.
          - If it is a single value, it is used as a title itself.
    - If it is None or unspecified, there is no title (and hspace between grid cells
      are set narrower in this case).
    
    If vmin/vmax is None or unspecified, it is guessed from values in each image
    (as in mnist_imshow).
    """
    fig = plt.figure(figsize=(width, height))
    titles_set = False
    if type(titles) is np.ndarray:
        titles_set = True
    elif type(titles) in (str, int, float):
        def _(value):
            while True:
                yield str(value)
        titles = _(titles)
        titles_set = True
    else:
        try:
            iter(titles)
            titles_set = True
        except TypeError:
            titles = tuple()
            titles_set = True
    hspace = 0.8 if titles_set else 0.05
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=hspace, wspace=0.05)
    i = 0
    for (image, title) in zip_longest(images, titles):
        if (type(image) is not np.ndarray and image==None) or i>=width*height:
            break
        ax = fig.add_subplot(height, width, i+1, xticks=[], yticks=[])
        i += 1
        imshow(ax, image, vmin=vmin, vmax=vmax)
        if title!=None:
            if type(title) in (tuple, list):
                ax.set_title(str(title[0]), **title[1])
            else:
                ax.set_title(str(title))

if __name__ == '__main__':
    pass
