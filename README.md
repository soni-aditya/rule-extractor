# Rules-Extractor

## OCR
To create a Python function that uses OCR to parse a document while excluding the header and footer, you can use the pytesseract library for OCR and Pillow for image processing. Here's a step-by-step plan:

- Install the required libraries: pytesseract and Pillow.
- Load the image from the given file path.
- Calculate the pixel height of the header and footer based on the given centimeter values.
- Crop the image to exclude the header and footer.
- Use pytesseract to perform OCR on the cropped image.
- Return the extracted text.

## Gen AI Implementation
To create a function that uses the Mistral 8B model from Hugging Face Hub and LangChain for prompt engineering to extract "rules for making a machine" from the document content, follow these steps:

- Install the required libraries: transformers, langchain, and torch.
- Load the Mistral 8B model from Hugging Face Hub.
- Use LangChain to create a prompt for extracting rules.
- Process the document content with the model and return the extracted rules.