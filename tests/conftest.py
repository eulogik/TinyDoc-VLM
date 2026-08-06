import pytest
from huggingface_hub import hf_hub_download


@pytest.fixture
def hf_hub_ok():
    """Skip tests that need a live download from the HF hub.

    CI runs from shared/unauthenticated IPs that get HTTP 429 rate-limited,
    so network-dependent tests (tokenizer download etc.) must not hard-fail.
    """
    try:
        hf_hub_download("HuggingFaceTB/SmolLM2-135M-Instruct", "config.json")
    except Exception:
        pytest.skip("HF hub unreachable / rate-limited")
    return True
