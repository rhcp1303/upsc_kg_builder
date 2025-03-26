import os
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import pdfplumber
import google.generativeai as genai
import time
import pytesseract

api_key = "AIzaSyCxTCYQO7s23L33kC4Io4G-i1p1ytD-OiI"
model = genai.GenerativeModel("gemini-1.5-flash")
genai.configure(api_key=api_key)
os.environ["TESSDATA_PREFIX"] = "/opt/local/share/tessdata/"


class PDFExtractor:

    def extract_text(self, pdf_file_path):
        raise NotImplementedError("Subclass must implement this method")

    def extract_page_text(self, pdf_file_path, page_number):
        raise NotImplementedError("Subclass must implement this method")


class SingleColumnDigitalPDFExtractor(PDFExtractor):
    def extract_page_text(self, pdf_file_path, page_number):
        try:
            with pdfplumber.open(pdf_file_path) as pdf:
                page = pdf.pages[page_number]
                extracted_text = page.extract_text()
                if extracted_text is None:
                    return ""
                print(extracted_text)
                return extracted_text
        except Exception as e:
            raise

    def extract_text(self, pdf_file_path):
        try:
            number_of_pages = len(pdfplumber.open(pdf_file_path).pages)
            with ProcessPoolExecutor(max_workers=4) as executor:
                results = list(
                    executor.map(self.extract_page_text, [pdf_file_path] * number_of_pages, range(number_of_pages)))
                return "\n".join(results)
        except Exception as e:
            print(f"Error processing digital single-column PDF: {e}")


class SingleColumnScannedPDFExtractorUsingLLM(PDFExtractor):
    def extract_page_text(self, pdf_file_path, page_number):
        try:
            with pdfplumber.open(pdf_file_path) as pdf:
                page = pdf.pages[page_number]
                image_path = f"temp/temp_page_{page_number + 1}.png"
                page.to_image(resolution=300).save(image_path)
                img = Image.open(image_path)
                response = model.generate_content(
                    ["extract text from this image without any additional context or headers", img])
                time.sleep(5)
                extracted_text = response.text
                if extracted_text is None:
                    return ""
                print(extracted_text)
                return extracted_text
        except Exception as e:
            raise

    def extract_text(self, pdf_file_path):
        try:
            with ProcessPoolExecutor(max_workers=1) as executor:
                number_of_pages = len(pdfplumber.open(pdf_file_path).pages)
                results = list(
                    executor.map(self.extract_page_text, [pdf_file_path] * number_of_pages,
                                 range(22, 31)))
                return "\n".join(results)
        except Exception as e:
            print(f"Error processing PDF: {e}")
        finally:
            for filename in os.listdir("temp"):
                if filename.startswith("temp_page_"):
                    os.remove(os.path.join("temp", filename))


class SingleColumnScannedPDFExtractorUsingOCR(PDFExtractor):
    def extract_page_text(self, pdf_file_path, page_number):
        try:
            with pdfplumber.open(pdf_file_path) as pdf:
                page = pdf.pages[page_number]
                image_path = f"temp/temp_page_{page_number + 1}.png"
                page.to_image(resolution=300).save(image_path)
                extracted_text = pytesseract.image_to_string(image_path)
                if extracted_text is None:
                    return ""
                print(extracted_text)

                return extracted_text

        except Exception as e:
            raise

    def extract_text(self, pdf_file_path):
        try:
            with ProcessPoolExecutor(max_workers=4) as executor:
                number_of_pages = len(pdfplumber.open(pdf_file_path).pages)
                results = list(
                    executor.map(self.extract_page_text, [pdf_file_path] * number_of_pages, range(number_of_pages)))
                return "\n".join(results)
        except Exception as e:
            print(f"Error processing PDF: {e}")
        finally:
            for filename in os.listdir("temp"):
                if filename.startswith("temp_page_"):
                    os.remove(os.path.join("temp", filename))


class TwoColumnDigitalPDFExtractor(PDFExtractor):
    def extract_page_text(self, pdf_file_path, page_number):
        try:
            with pdfplumber.open(pdf_file_path) as pdf:
                page = pdf.pages[page_number]
                crop_box = (0, 0, page.width / 2, page.height)
                left_half_page = page.within_bbox(crop_box)
                left_half_extracted_text = left_half_page.extract_text()
                if left_half_extracted_text is None:
                    left_half_extracted_text = ""
                crop_box = (page.width / 2, 0, page.width, page.height)
                right_half_page = page.within_bbox(crop_box)
                right_half_extracted_text = right_half_page.extract_text()
                if right_half_extracted_text is None:
                    right_half_extracted_text = ""
                extracted_text = left_half_extracted_text + "\n" + right_half_extracted_text
                print(extracted_text)
                return extracted_text
        except Exception as e:
            raise

    def extract_text(self, pdf_file_path):
        try:
            number_of_pages = len(pdfplumber.open(pdf_file_path).pages)
            with ProcessPoolExecutor(max_workers=4) as executor:
                results = list(
                    executor.map(self.extract_page_text, [pdf_file_path] * number_of_pages, range(number_of_pages)))
                return "\n".join(results)
        except Exception as e:
            print(f"Error processing digital single-column PDF: {e}")


class TwoColumnScannedPDFExtractorUsingLLM(PDFExtractor):
    def extract_page_text(self, pdf_file_path, page_number):
        try:
            with pdfplumber.open(pdf_file_path) as pdf:
                page = pdf.pages[page_number]
                crop_box = (0, 0, page.width / 2, page.height)
                left_half_page = page.within_bbox(crop_box)
                left_half_image = left_half_page.to_image(resolution=300)
                left_half_image_path = f"temp/temp_left_half_page_{page.page_number + 1}.png"
                left_half_image.save(left_half_image_path)
                left_half_img = Image.open(left_half_image_path)
                crop_box = (page.width / 2, 0, page.width, page.height)
                right_half_page = page.within_bbox(crop_box)
                right_half_image = right_half_page.to_image(resolution=300)
                right_half_image_path = f"temp/temp_right_half_page_{page.page_number + 1}.png"
                right_half_image.save(right_half_image_path)
                right_half_img = Image.open(right_half_image_path)

                left_half_extracted_text = model.generate_content(
                    ["extract text from this image without any additional context or headers", left_half_img]).text
                if left_half_extracted_text is None:
                    left_half_extracted_text = ""
                time.sleep(5)
                right_half_extracted_text = model.generate_content(
                    ["extract text from this image without any additional context or headers",
                     right_half_img]).text
                if right_half_extracted_text is None:
                    right_half_extracted_text = ""
                time.sleep(5)
                extracted_text = left_half_extracted_text + "\n" + right_half_extracted_text
                print(extracted_text)
                return extracted_text
        except Exception as e:
            raise

    def extract_text(self, pdf_file_path):
        try:
            with ProcessPoolExecutor(max_workers=1) as executor:
                number_of_pages = len(pdfplumber.open(pdf_file_path).pages)
                results = list(
                    executor.map(self.extract_page_text, [pdf_file_path] * number_of_pages, range(number_of_pages)))
                return "\n".join(results)
        except Exception as e:
            print(f"Error processing PDF: {e}")
        finally:
            for filename in os.listdir("temp"):
                if filename.startswith("temp_left_half_page_"):
                    os.remove(os.path.join("temp", filename))
                if filename.startswith("temp_right_half_page_"):
                    os.remove(os.path.join("temp", filename))


class TwoColumnScannedPDFExtractorUsingOCR(PDFExtractor):
    def extract_page_text(self, pdf_file_path, page_number, lang='eng'):
        try:
            with pdfplumber.open(pdf_file_path) as pdf:
                page = pdf.pages[page_number]
                crop_box = (0, 0, page.width / 2, page.height)
                left_half_page = page.within_bbox(crop_box)
                left_half_image = left_half_page.to_image(resolution=300)
                left_half_image_path = f"temp/temp_left_half_page_{page.page_number + 1}.png"
                left_half_image.save(left_half_image_path)
                crop_box = (page.width / 2, 0, page.width, page.height)
                right_half_page = page.within_bbox(crop_box)
                right_half_image = right_half_page.to_image(resolution=300)
                right_half_image_path = f"temp/temp_right_half_page_{page.page_number + 1}.png"
                right_half_image.save(right_half_image_path)
                left_half_extracted_text = pytesseract.image_to_string(left_half_image_path, lang=lang)
                if left_half_extracted_text is None:
                    left_half_extracted_text = ""
                right_half_extracted_text = pytesseract.image_to_string(right_half_image_path, lang=lang)
                if right_half_extracted_text is None:
                    right_half_extracted_text = ""
                extracted_text = left_half_extracted_text + "\n" + right_half_extracted_text
                print(extracted_text)
                return extracted_text
        except Exception as e:
            raise

    def extract_text(self, pdf_file_path, lang='eng'):
        try:
            with ProcessPoolExecutor(max_workers=4) as executor:
                number_of_pages = len(pdfplumber.open(pdf_file_path).pages)
                results = list(
                    executor.map(self.extract_page_text, [pdf_file_path] * number_of_pages, range(number_of_pages)))
                return "\n".join(results)
        except Exception as e:
            print(f"Error processing PDF: {e}")
        finally:
            for filename in os.listdir("temp"):
                if filename.startswith("temp_left_half_page_"):
                    os.remove(os.path.join("temp", filename))
                if filename.startswith("temp_right_half_page_"):
                    os.remove(os.path.join("temp", filename))
