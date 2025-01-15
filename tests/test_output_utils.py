"""Tests for output_utils module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table

from par_ai_core.output_utils import (
    DisplayOutputFormat,
    csv_file_to_table,
    csv_to_table,
    display_formatted_output,
    get_output_format_prompt,
    highlight_json,
    highlight_json_file,
)


@pytest.fixture
def sample_csv_data() -> str:
    """Sample CSV data for testing."""
    return "name,age,city\nJohn,30,New York\nJane,25,Los Angeles"


@pytest.fixture
def sample_json_data() -> str:
    """Sample JSON data for testing."""
    return json.dumps({"name": "John", "age": 30, "city": "New York"}, indent=2)


@pytest.fixture
def temp_csv_file(tmp_path: Path, sample_csv_data: str) -> Path:
    """Create a temporary CSV file."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(sample_csv_data, encoding="utf-8")
    return csv_file


@pytest.fixture
def temp_json_file(tmp_path: Path, sample_json_data: str) -> Path:
    """Create a temporary JSON file."""
    json_file = tmp_path / "test.json"
    json_file.write_text(sample_json_data, encoding="utf-8")
    return json_file


def test_display_output_format_enum() -> None:
    """Test DisplayOutputFormat enum values."""
    assert DisplayOutputFormat.NONE == "none"
    assert DisplayOutputFormat.PLAIN == "plain"
    assert DisplayOutputFormat.MD == "md"
    assert DisplayOutputFormat.CSV == "csv"
    assert DisplayOutputFormat.JSON == "json"


def test_csv_to_table_valid_data(sample_csv_data: str) -> None:
    """Test csv_to_table with valid data."""
    table = csv_to_table(sample_csv_data)
    assert isinstance(table, Table)
    assert table.columns[0].header == "name"
    assert table.columns[1].header == "age"
    assert table.columns[2].header == "city"


def test_csv_to_table_empty_data() -> None:
    """Test csv_to_table with empty data."""
    table = csv_to_table("")
    assert isinstance(table, Table)
    assert len(table.columns) == 1
    assert table.columns[0].header == "Empty"


def test_csv_to_table_no_fields() -> None:
    """Test csv_to_table with data containing no fields."""
    table = csv_to_table("\n\n")  # Only newlines, no actual fields
    assert isinstance(table, Table)
    assert len(table.columns) == 1
    assert table.columns[0].header == "Empty"


def test_csv_to_table_invalid_data() -> None:
    """Test csv_to_table with invalid CSV data."""
    table = csv_to_table("invalid,csv\ndata,missing,field")
    assert isinstance(table, Table)
    assert table.columns[0].header == "Error"


def test_csv_to_table_inconsistent_fields() -> None:
    """Test csv_to_table with inconsistent number of fields in rows."""
    data = "name,age,city\nJohn,30\nJane,25,Los Angeles,Extra"  # Second row missing field, third row extra field
    table = csv_to_table(data)
    assert isinstance(table, Table)
    assert table.columns[0].header == "Error"
    assert table.row_count == 1
    # Get the error message from the table
    error_msg = str(table._cells[0][0])  # Access first cell in first row
    assert "Failed to parse CSV data: Inconsistent number of fields" in error_msg


def test_csv_file_to_table(temp_csv_file: Path) -> None:
    """Test csv_file_to_table with a valid file."""
    table = csv_file_to_table(temp_csv_file)
    assert isinstance(table, Table)
    assert table.columns[0].header == "name"
    assert table.title == temp_csv_file.name


def test_highlight_json_valid(sample_json_data: str) -> None:
    """Test highlight_json with valid JSON."""
    syntax = highlight_json(sample_json_data)
    assert isinstance(syntax, Syntax)
    assert syntax.lexer.name == "JSON"


def test_highlight_json_file(temp_json_file: Path) -> None:
    """Test highlight_json_file with a valid file."""
    syntax = highlight_json_file(temp_json_file)
    assert isinstance(syntax, Syntax)
    assert syntax.lexer.name == "JSON"


@pytest.mark.parametrize(
    "format_type,expected_content",
    [
        (DisplayOutputFormat.MD, "table"),
        (DisplayOutputFormat.JSON, "schema"),
        (DisplayOutputFormat.CSV, "header"),
        (DisplayOutputFormat.PLAIN, "plain text"),
        (DisplayOutputFormat.NONE, ""),
    ],
)
def test_get_output_format_prompt(format_type: DisplayOutputFormat, expected_content: str) -> None:
    """Test get_output_format_prompt for different formats."""
    prompt = get_output_format_prompt(format_type)
    assert isinstance(prompt, str)
    assert expected_content in prompt.lower()


class MockConsole:
    """Mock console for testing display output."""

    def __init__(self) -> None:
        """Initialize mock console."""
        self.printed: list[Any] = []

    def print(self, content: Any) -> None:
        """Mock print method."""
        self.printed.append(content)


@pytest.mark.parametrize(
    "format_type,content,expected_type",
    [
        (DisplayOutputFormat.PLAIN, "Hello", str),
        (DisplayOutputFormat.MD, "# Hello", Markdown),
        (DisplayOutputFormat.CSV, "header\ndata", Table),
        (DisplayOutputFormat.JSON, '{"key": "value"}', Syntax),
        (DisplayOutputFormat.NONE, "content", type(None)),
    ],
)
def test_display_formatted_output(format_type: DisplayOutputFormat, content: str, expected_type: type) -> None:
    """Test display_formatted_output for different formats."""
    mock_console = MockConsole()
    display_formatted_output(content, format_type, mock_console)  # type: ignore

    if format_type == DisplayOutputFormat.NONE:
        assert len(mock_console.printed) == 0
    elif format_type == DisplayOutputFormat.PLAIN:
        assert len(mock_console.printed) == 0
    else:
        assert len(mock_console.printed) == 1
        if expected_type != str:
            assert isinstance(mock_console.printed[0], expected_type)
