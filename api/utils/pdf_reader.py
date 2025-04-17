import fitz  

def extract_text_from_pdf(content: bytes) -> str:
    with fitz.open(stream=content, filetype="pdf") as doc:
        return "\n".join([page.get_text() for page in doc])
