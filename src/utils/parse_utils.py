"""Parsing utilities for extracting structured data from text responses."""

import re

from src.utils.file_utils import parse_numbered_list


def parse_bullet_list(text: str) -> list[str]:
    """Extract items from bullet list format (e.g. '- item' or '* item')."""
    pattern = re.compile(r'^\s*[-*]\s+(.+)$', re.MULTILINE)
    return [m.group(1).strip() for m in pattern.finditer(text)]


def parse_filter_response(response: str) -> bool:
    """Parse filter response to determine if content should be kept."""
    response_upper = response.strip().upper()

    if "GRADE: INCLUDE" in response_upper or "GRADE:INCLUDE" in response_upper:
        return True
    elif "GRADE: EXCLUDE" in response_upper or "GRADE:EXCLUDE" in response_upper:
        return False

    lines = response_upper.split('\n')
    for line in lines:
        line_stripped = line.strip()
        if line_stripped == "INCLUDE":
            return True
        elif line_stripped == "EXCLUDE":
            return False

    if "GRADE: TRUE" in response_upper or "GRADE:TRUE" in response_upper:
        return False
    elif "GRADE: FALSE" in response_upper or "GRADE:FALSE" in response_upper:
        return True

    print(f"Warning: Unclear filter response, keeping by default: {response[:100]}")
    return True


def extract_xml_tag(text: str, tag: str) -> str | None:
    """Extract content from an XML tag."""
    match = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_numbered_items_from_xml(text: str, tag: str = "output") -> list[str]:
    """Extract numbered list items from within an XML tag.

    Falls back to raw parsing if tag not found.
    """
    content = extract_xml_tag(text, tag)
    if not content:
        return parse_numbered_list(text)
    return parse_numbered_list(content)


def parse_v2_filter_response(response: str) -> bool:
    """Parse v2 filter response with <verdict> tag format."""
    verdict = extract_xml_tag(response, "verdict")
    if verdict:
        verdict_upper = verdict.upper().strip()
        if "INCLUDE" in verdict_upper:
            return True
        if "EXCLUDE" in verdict_upper:
            return False

    response_upper = response.upper()
    include_pos = response_upper.rfind("INCLUDE")
    exclude_pos = response_upper.rfind("EXCLUDE")
    if include_pos > exclude_pos:
        return True
    if exclude_pos > include_pos:
        return False

    return parse_filter_response(response)
