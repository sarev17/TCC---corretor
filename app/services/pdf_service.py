from pdf2image import convert_from_path

def pdf_to_images(path):
    images = convert_from_path(path, dpi=200)
    return images