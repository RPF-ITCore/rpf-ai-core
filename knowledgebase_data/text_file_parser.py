import re
from typing import Dict, List
from pathlib import Path


def _extract_field_value(line: str, pattern: str) -> str:
    """
    Extract field value from a bilingual line.
    
    Examples:
        [City] Jobar | المدينة: Jobar -> "Jobar"
        [Aspect] Public Health | المحور: Public Health -> "Public Health"
    
    Args:
        line: The line to parse
        pattern: Regex pattern to match (e.g., r"\[City\]\s*(.+?)\s*\|\s*المدينة:")
    
    Returns:
        Extracted value or "Not Known" if not found
    """
    match = re.search(pattern, line)
    if match:
        value = match.group(1).strip()
        return value if value else "Not Known"
    return "Not Known"


def parse_text_file(text_path: str) -> List[Dict[str, str]]:
    """
    Parse a text file into blocks separated by '-----------------------------'.
    Each block is treated as a full chunk.
    
    Extracts metadata:
    - City from lines matching [City] <value> | المدينة: <value>
    - Aspect from lines matching [Aspect] <value> | المحور: <value>
    - Sub-Aspect from lines matching [Sub-Aspect] <value> | الجانب الفرعي: <value>
    
    Args:
        text_path: Path to the text file to parse
    
    Returns:
        List of dictionaries with structure:
        {
            "id": "block_0", "block_1", etc.
            "city": extracted city name
            "aspect": extracted aspect
            "subaspect": extracted sub-aspect
            "text": full block text (preserving all formatting)
        }
    """
    text_path = Path(text_path)
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")
    
    with open(text_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by the separator line
    blocks = content.split("-----------------------------")
    
    records: List[Dict[str, str]] = []
    
    # Regex patterns for extracting field values
    city_pattern = r"\[City\]\s*(.+?)\s*\|\s*المدينة:"
    aspect_pattern = r"\[Aspect\]\s*(.+?)\s*\|\s*المحور:"
    subaspect_pattern = r"\[Sub-Aspect\]\s*(.+?)\s*\|\s*الجانب الفرعي:"
    
    for index, block in enumerate(blocks):
        # Skip empty blocks
        block = block.strip()
        if not block:
            continue
        
        # Split block into lines for metadata extraction
        lines = block.split("\n")
        
        # Extract metadata from structured lines
        city = "Not Known"
        aspect = "Not Known"
        subaspect = "Not Known"
        
        for line in lines:
            line_stripped = line.strip()
            
            # Extract City
            if "[City]" in line_stripped and "المدينة:" in line_stripped:
                city = _extract_field_value(line_stripped, city_pattern)
            
            # Extract Aspect
            elif "[Aspect]" in line_stripped and "المحور:" in line_stripped:
                aspect = _extract_field_value(line_stripped, aspect_pattern)
            
            # Extract Sub-Aspect
            elif "[Sub-Aspect]" in line_stripped and "الجانب الفرعي:" in line_stripped:
                subaspect = _extract_field_value(line_stripped, subaspect_pattern)
        
        # Store the entire block as text (preserving all formatting)
        # Add back the separator line to maintain the original structure
        full_text = block + "\n-----------------------------"
        
        records.append({
            "id": f"block_{index}",
            "city": city,
            "aspect": aspect,
            "subaspect": subaspect,
            "text": full_text,
        })
    
    return records


if __name__ == "__main__":
    # Simple test when run directly
    import sys
    
    default_path = Path(__file__).parent / "jobar_cleaned_final.txt"
    test_path = sys.argv[1] if len(sys.argv) > 1 else str(default_path)
    
    try:
        records = parse_text_file(test_path)
        print(f"✅ Parsed {len(records)} blocks from {test_path}")
        if records:
            print(f"\nSample block (first):")
            print(f"  ID: {records[0]['id']}")
            print(f"  City: {records[0]['city']}")
            print(f"  Aspect: {records[0]['aspect']}")
            print(f"  Sub-Aspect: {records[0]['subaspect']}")
            print(f"  Text preview: {records[0]['text'][:100]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

