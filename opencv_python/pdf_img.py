from pdf2image import convert_from_path

def convert_pdf_to_images(pdf_path):
    try:
        return convert_from_path(
            pdf_path,
            poppler_path=r"C:\Users\Asus\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"  # <== your poppler path
        )
    except Exception as e:
        print("Error converting PDF to images:", e)
        return []
