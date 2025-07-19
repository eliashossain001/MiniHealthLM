import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    extract_text_from_pdf("data/corpus.pdf", "data/corpus.txt")
