from ..helpers import pdf_utils as pu


def select_pdf_extractor(pdf_file_type, number_of_columns, use_llm):
    if pdf_file_type == "scanned":
        if use_llm.lower() == 'yes':
            if number_of_columns == 1:
                return pu.SingleColumnScannedPDFExtractorUsingLLM()
            else:
                return pu.TwoColumnScannedPDFExtractorUsingLLM()
        else:
            if number_of_columns == 1:
                return pu.SingleColumnScannedPDFExtractorUsingOCR()
            else:
                return pu.TwoColumnScannedPDFExtractorUsingOCR()
    else:
        if number_of_columns == 1:
            return pu.SingleColumnDigitalPDFExtractor()
        else:
            return pu.TwoColumnDigitalPDFExtractor()
