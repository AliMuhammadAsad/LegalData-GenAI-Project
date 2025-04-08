import unittest
from src.backend.document_processing.textract_handler import extract_text
from src.backend.document_processing.document_parser import parse_text

class TestDocumentProcessing(unittest.TestCase):

    def test_extract_text(self):
        # Assuming we have a sample document in S3 for testing
        sample_document = 's3://your-bucket/sample-document.pdf'
        extracted_text = extract_text(sample_document)
        self.assertIsInstance(extracted_text, str)
        self.assertGreater(len(extracted_text), 0)

    def test_parse_text(self):
        sample_text = "This is a sample text for testing."
        parsed_output = parse_text(sample_text)
        self.assertIsInstance(parsed_output, dict)
        self.assertIn('content', parsed_output)
        self.assertEqual(parsed_output['content'], sample_text)

if __name__ == '__main__':
    unittest.main()