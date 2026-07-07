import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n","\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def extract_text(self, file):
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    
    def process(self, file):
        text = self.extract_text(file)
        return self.text_splitter.split_text(text)