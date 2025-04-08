import json

class DocumentParser:
    def __init__(self, extracted_text):
        self.extracted_text = extracted_text

    def parse(self):
        # Split the text into paragraphs
        paragraphs = self.extracted_text.split('\n\n')
        parsed_documents = []

        for paragraph in paragraphs:
            # Further processing can be done here
            cleaned_paragraph = paragraph.strip()
            if cleaned_paragraph:
                parsed_documents.append(cleaned_paragraph)

        return parsed_documents

    def to_json(self):
        parsed_documents = self.parse()
        return json.dumps(parsed_documents)