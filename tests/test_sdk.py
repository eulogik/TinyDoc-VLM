import pytest
from PIL import Image
import numpy as np
from pathlib import Path
import sys

# Insert sdk directory into path
sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))

from tinydoc import TinyDocExtractor
from tinydoc.models import ExtractionResult, QAResult, TableResult

@pytest.fixture
def dummy_image():
    return Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))

def test_sdk_extractor_initialisation():
    # Load dummy model/processor
    extractor = TinyDocExtractor(device="cpu")
    assert extractor.model is not None
    assert extractor.processor is not None

def test_sdk_extractor_methods(dummy_image):
    extractor = TinyDocExtractor(device="cpu")
    
    # Test QA
    qa_res = extractor.ask(dummy_image, "What is the total?")
    assert isinstance(qa_res, QAResult)
    assert qa_res.question == "What is the total?"
    assert isinstance(qa_res.answer, str)
    assert qa_res.latency_ms > 0
    assert qa_res.num_tokens_generated >= 0
    
    # Test JSON extraction
    ext_res = extractor.extract(dummy_image, output_format="json")
    assert isinstance(ext_res, ExtractionResult)
    assert isinstance(ext_res.raw_text, str)
    assert isinstance(ext_res.fields, dict)
    
    # Test table extraction
    table_res = extractor.extract_table(dummy_image)
    assert isinstance(table_res, TableResult)
    assert isinstance(table_res.raw_table, str)
    assert isinstance(table_res.markdown, str)

def test_html_table_to_markdown_converter():
    extractor = TinyDocExtractor(device="cpu")
    html = (
        "<table>"
        "<tr><th>Header1</th><th>Header2</th></tr>"
        "<tr><td>Value1</td><td>Value2</td></tr>"
        "</table>"
    )
    md = extractor._html_table_to_markdown(html)
    expected = (
        "| Header1 | Header2 |\n"
        "| --- | --- |\n"
        "| Value1 | Value2 |"
    )
    assert md == expected
