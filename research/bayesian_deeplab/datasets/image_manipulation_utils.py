"""The file contains utility methods (wrappers over PIL APIs) for image manipulation
like cropping and resizing images.
"""

from PIL import Image


def save_image(image, path):
    """
    Method acting as a simple wrapper over Image.save().

    Args:
        image: An image object
        path: Path for saving the image
    """

    image.save(path)


def crop_image_by_factor_from_file(factor, input_path, dst_path=None):
    """
    Method to crop a given image centrally such that the cropped image has dimensions
    which are (original_dimensions/factor).

    Args:
        factor: Factor by which the dimensions of the image will be reduced
        input_path: Path of the file to be processed
        dst_path: Path of the destination file

    Returns:
        The cropped image.
    """

    # Loading image
    im = Image.open(input_path)
    width, height = im.size

    # Cropping and returning image
    return crop_image_from_file(width/factor, height/factor, input_path, dst_path)


def crop_image_by_factor(factor, image):
    """
    Method to crop a given image centrally such that the cropped image has dimensions
    which are (original_dimensions/factor).

    Args:
        factor: Factor by which the dimensions of the image will be reduced
        image: Image object to be processed

    Returns:
        The cropped image.
    """
    width, height = image.size

    # Cropping and returning image
    return crop_image(width/factor, height/factor, image)


def resize_image_by_factor_from_file(factor, input_path, dst_path=None):
    """
    Method to resize a given image such that the resized image has dimensions
    which are (original_dimensions/factor).

    Args:
        factor: Factor by which the dimensions of the image will be reduced
        input_path: Path of the file to be processed
        dst_path: Path of the destination file

    Returns:
        The resized image.
    """

    # Loading image
    im = Image.open(input_path)
    width, height = im.size

    # Resizing and returning image
    return resize_image_from_file(width/factor, height/factor, input_path, dst_path)


def resize_image_by_factor(factor, image):
    """
    Method to resize a given image such that the resized image has dimensions
    which are (original_dimensions/factor).

    Args:
        factor: Factor by which the dimensions of the image will be reduced
        image: Image object to be processed

    Returns:
        The resized image.
    """
    width, height = image.size

    # Resizing and returning image
    return resize_image(width/factor, height/factor, image)


def crop_image_from_file(new_width, new_height, input_path, dst_path=None):
    """
    Method to crop a given input image to new width and height specifications given.

    Args:
        new_width: Width of the cropped image
        new_height: Height of the cropped image
        input_path: Path of the file to be processed
        dst_path: Path of the destination file

    Returns:
        The cropped image.
    """

    # Loading image
    im = Image.open(input_path)
    width, height = im.size

    # Cropping image
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    res_im = im.crop((left, top, right, bottom))

    # Saving cropped image
    if dst_path is not None:
        res_im.save(dst_path)

    return res_im


def crop_image(new_width, new_height, image):
    """
    Method to crop a given input image to new width and height specifications given.

    Args:
        new_width: Width of the cropped image
        new_height: Height of the cropped image
        image: Image object to be processed

    Returns:
        The cropped image.
    """

    width, height = image.size

    # Cropping image
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    res_im = image.crop((left, top, right, bottom))

    return res_im


def resize_image_from_file(new_width, new_height, input_path, dst_path=None):
    """
    Method to resize a given input image to new width and height specifications given.

    Args:
        new_width: Width of the resized image
        new_height: Height of the resized image
        input_path: Path of the file to be processed
        dst_path: Path of the destination file

    Returns:
        The resized image.
    """

    # Loading image
    im = Image.open(input_path)

    # Resizing image
    res_im = im.resize((int(new_width), int(new_height)), Image.ANTIALIAS)

    # Saving cropped image
    if dst_path is not None:
        res_im.save(dst_path)

    return res_im


def resize_image(new_width, new_height, image):
    """
    Method to resize a given input image to new width and height specifications given.

    Args:
        new_width: Width of the resized image
        new_height: Height of the resized image
        image: Image object to be processed

    Returns:
        The resized image.
    """

    # Resizing image
    return image.resize((int(new_width), int(new_height)), Image.ANTIALIAS)
