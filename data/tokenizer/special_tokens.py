import os
from transformers import AutoTokenizer
from typing import List, Dict

# Document-specific special tokens to add to the vocabulary
DOCUMENT_SPECIAL_TOKENS = [
    "<image>",          # Image placeholder token
    "<image_end>",      # Optional end marker
    "<page_start>",     # Page boundary start
    "<page_end>",       # Page boundary end
    "<table>",          # Table region start
    "</table>",         # Table region end
    "<tr>",             # Table row start
    "</tr>",            # Table row end
    "<td>",             # Table cell start
    "</td>",            # Table cell end
    "<th>",             # Table header cell start
    "</th>",            # Table header cell end
    "<header>",         # Document header region
    "<footer>",         # Document footer region
    "<figure>",         # Figure/chart region
    "<row_sep>",        # Row visual separator (to preserve reading order grids)
    "<checkbox_on>",    # Checked checkbox state
    "<checkbox_off>",   # Unchecked checkbox state
    "<signature>",      # Signature placeholder
    "<stamp>",          # Stamp placeholder
    "<handwritten>",    # Handwritten annotation marker
    "<page_break>",     # Multi-page layout divider
    "<json>",           # JSON block marker
    "</json>",          # End JSON block
    "<markdown>",       # Markdown block marker
    "</markdown>",      # End Markdown block
    "<key>",            # KV pair key start
    "</key>",           # KV pair key end
    "<value>",          # KV pair value start
    "</value>",         # KV pair value end
]

def extend_tokenizer(
    base_model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    output_dir: str = "data/tokenizer/extended_tokenizer"
) -> AutoTokenizer:
    """
    Loads a base tokenizer, extends it with document-specific special tokens,
    and saves the configuration to output_dir.
    """
    print(f"Loading base tokenizer from '{base_model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    print(f"Base tokenizer vocabulary size: {len(tokenizer)}")
    
    # Check which tokens are already present
    new_tokens = [tok for tok in DOCUMENT_SPECIAL_TOKENS if tok not in tokenizer.get_vocab()]
    
    if new_tokens:
        print(f"Adding {len(new_tokens)} new special tokens to the vocabulary...")
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        print(f"New vocabulary size: {len(tokenizer)}")
    else:
        print("All special tokens are already present in vocabulary.")
        
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Extended tokenizer saved to '{output_dir}'.")
    return tokenizer

if __name__ == "__main__":
    extend_tokenizer()
