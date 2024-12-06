import pytesseract
from PIL import Image

def cm_to_pixels(cm, dpi=300):
    return int((cm / 2.54) * dpi)

def extract_text_from_image(file_path, header_height_cm, footer_height_cm):
    # Load the image
    image = Image.open(file_path)
    
    # Convert header and footer height from cm to pixels
    dpi = image.info.get('dpi', (300, 300))[0]  # Default to 300 DPI if not specified
    header_height_px = cm_to_pixels(header_height_cm, dpi)
    footer_height_px = cm_to_pixels(footer_height_cm, dpi)
    
    # Calculate the cropping box
    width, height = image.size
    crop_box = (0, header_height_px, width, height - footer_height_px)
    
    # Crop the image to exclude header and footer
    cropped_image = image.crop(crop_box)
    
    # Perform OCR on the cropped image
    text = pytesseract.image_to_string(cropped_image)
    
    return text
