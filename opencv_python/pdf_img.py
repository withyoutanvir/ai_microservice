# pdf_img.py

from pdf2image import convert_from_path

def convert_pdf_to_images(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=300)
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []
