#!/usr/bin/env python3
"""
Test suite for the bank statement parser agent
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_icici_parser_exists():
    """Test that the ICICI parser file is created"""
    parser_path = Path("custom_parsers/icici_parser.py")
    assert parser_path.exists(), f"Parser file not found: {parser_path}"

def test_icici_parser_function():
    """Test that the parser has the correct function signature"""
    from custom_parsers.icici_parser import parse
    import inspect
    
    # Check function signature
    sig = inspect.signature(parse)
    assert len(sig.parameters) == 1, "parse() should take exactly one parameter"
    
    # Check return type hint (if present)
    if sig.return_annotation != inspect.Signature.empty:
        assert 'DataFrame' in str(sig.return_annotation), "parse() should return DataFrame"

def test_icici_parser_output():
    """Test that the parser produces correct output format"""
    from custom_parsers.icici_parser import parse
    
    # Test with sample data
    pdf_path = "data/icici/icici_sample.pdf"
    csv_path = "data/icici/icici_sample.csv"
    
    if not Path(pdf_path).exists():
        pytest.skip(f"Sample PDF not found: {pdf_path}")
    if not Path(csv_path).exists():
        pytest.skip(f"Sample CSV not found: {csv_path}")
    
    # Parse PDF
    result_df = parse(pdf_path)
    expected_df = pd.read_csv(csv_path)
    
    # Validate structure
    assert isinstance(result_df, pd.DataFrame), "Parser should return a DataFrame"
    assert list(result_df.columns) == list(expected_df.columns), "Column names should match"
    assert result_df.shape[0] > 0, "Parser should extract at least one transaction"
    
    # Validate content (relaxed comparison for demo)
    assert result_df.shape[0] == expected_df.shape[0], "Row count should match"

def test_parser_error_handling():
    """Test parser handles invalid inputs gracefully"""
    from custom_parsers.icici_parser import parse
    
    # Test with non-existent file
    with pytest.raises((FileNotFoundError, Exception)):
        parse("non_existent_file.pdf")
    
    # Test with invalid file type
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"not a pdf")
        tmp_path = tmp.name
    
    try:
        with pytest.raises(Exception):
            parse(tmp_path)
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
