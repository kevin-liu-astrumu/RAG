import os 
from llama_parse import LlamaParse 
from langchain.text_splitter import MarkdownHeaderTextSplitter

class DocumentManager:
    def __init__(self):
        self.parser = LlamaParse(
            api_key = os.getenv("LLAMA_PARSE_API_KEY"), 
            result_type = "markdown"
        )
    
    def load_documents(self, document_path):
        documents = self.parser.load_data(document_path)
        return documents

    def split_documents(self, document_path):
        documents = self.load_documents(document_path)
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")]
        text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        sections = text_splitter.split_text(documents[0].text)
        return sections