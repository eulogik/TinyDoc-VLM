from .models import ExtractionResult, QAResult, TableResult
from .pipeline import ReceiptPipeline, DocumentResult, FieldResult, build_engine
from .evidence import Evidence, attach_evidence
from .overlay import draw_overlay

__all__ = [
    "ExtractionResult",
    "QAResult",
    "TableResult",
    "ReceiptPipeline",
    "DocumentResult",
    "FieldResult",
    "build_engine",
    "Evidence",
    "attach_evidence",
    "draw_overlay",
    # legacy name kept for import compatibility
    "TinyDocExtractor",
]


def __getattr__(name: str):
    # Lazy legacy extractor (256M path) so pipeline imports do not load torch models
    if name == "TinyDocExtractor":
        from .extractor import TinyDocExtractor

        return TinyDocExtractor
    raise AttributeError(name)
