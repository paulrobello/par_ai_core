"""Output formatting utilities for enhanced display and data presentation.

This module provides a set of utilities for formatting and displaying output in various
formats, including plain text, Markdown, CSV, and JSON. It leverages the Rich library
to create enhanced terminal output with features such as syntax highlighting, tables,
and formatted text.

Key features:
- Support for multiple output formats (plain text, Markdown, CSV, JSON)
- Conversion of CSV data to Rich tables for improved readability
- Syntax highlighting for JSON data
- Flexible display options using Rich console
- Utilities for handling both string data and file inputs

Main components:
- DisplayOutputFormat: An enum class for specifying output format choices
- csv_to_table: Converts CSV string data to a Rich Table
- csv_file_to_table: Converts a CSV file to a Rich Table
- highlight_json: Applies syntax highlighting to JSON string data
- highlight_json_file: Applies syntax highlighting to JSON file content
- get_output_format_prompt: Generates format-specific instructions for AI models
- display_formatted_output: Displays content in the specified format using Rich

Usage:
This module can be used to enhance the presentation of data in command-line
interfaces, improve the readability of structured data like CSV and JSON,
and provide consistent formatting across different output types.

Example:
    from par_ai_core.output_utils import display_formatted_output, DisplayOutputFormat

    data = '{"name": "John", "age": 30}'
    display_formatted_output(data, DisplayOutputFormat.JSON)

Note:
    This module requires the Rich library for enhanced terminal output.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from strenum import StrEnum

from par_ai_core.par_logging import console_err


class DisplayOutputFormat(StrEnum):
    """Enum for display output format choices."""

    NONE = "none"
    """No output."""
    PLAIN = "plain"
    """Plain text output."""
    MD = "md"
    """Rich Markdown output."""
    CSV = "csv"
    """Rich Table output."""
    JSON = "json"
    """Rich JSON output."""


def csv_to_table(data: str, title: str = "Results") -> Table:
    """Convert CSV string data to a Rich Table format.

    Args:
        data: CSV formatted string data to convert
        title: Title to display at the top of the table. Defaults to "Results"

    Returns:
        Table: A Rich Table object containing the formatted CSV data

    Raises:
        csv.Error: If CSV parsing fails
        ValueError: If CSV data is malformed
    """
    data = data.strip()
    table = Table(title=title, show_header=True, header_style="bold cyan")

    if not data:
        table.add_column("Empty", justify="left", style="cyan", no_wrap=True)
        return table

    try:
        reader = csv.reader(io.StringIO(data))
        headers = next(reader, None)

        # Check if headers is None or contains only empty strings
        if not headers or all(not field.strip() for field in headers):
            table.add_column("Error", justify="left", style="red")
            table.add_row("No fields found in CSV data")
            return table

        # Add columns
        for field in headers:
            table.add_column(str(field), justify="left", style="cyan", no_wrap=True)

        # Add rows
        try:
            for row in reader:
                if len(row) != len(headers):
                    table = Table(title=title)
                    table.add_column("Error", justify="left", style="red")
                    table.add_row("Failed to parse CSV data: Inconsistent number of fields")
                    return table
                table.add_row(*[str(val) for val in row])
        except Exception as e:
            table = Table(title=title)
            table.add_column("Error", justify="left", style="red")
            table.add_row(f"Failed to parse CSV data: {str(e)}")
            return table

    except (csv.Error, ValueError) as e:
        table = Table(title=title)
        table.add_column("Error", justify="left", style="red")
        table.add_row(f"Failed to parse CSV data: {str(e)}")
    return table


def csv_file_to_table(csv_file: Path, title: str | None = None) -> Table:
    """Convert a CSV file to a Rich Table format.

    Args:
        csv_file: Path to the CSV file to convert
        title: Optional title for the table. If None, uses the filename

    Returns:
        Table: A Rich Table object containing the formatted CSV data

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        UnicodeDecodeError: If file encoding is not UTF-8
    """
    return csv_to_table(
        csv_file.read_text(encoding="utf-8").strip(),
        csv_file.name if title is None else title,
    )


def highlight_json(data: str) -> Syntax:
    """Create syntax-highlighted JSON output.

    Args:
        data: JSON string to highlight

    Returns:
        Syntax: A Rich Syntax object with JSON highlighting applied
    """
    return Syntax(data, "json", background_color="default")


def highlight_json_file(json_file: Path) -> Syntax:
    """Create syntax-highlighted output from a JSON file.

    Args:
        json_file: Path to the JSON file to highlight

    Returns:
        Syntax: A Rich Syntax object with JSON highlighting applied

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        UnicodeDecodeError: If file encoding is not UTF-8
    """
    return highlight_json(json_file.read_text(encoding="utf-8").strip())


def get_output_format_prompt(display_format: DisplayOutputFormat) -> str:
    """Get the appropriate output format prompt instructions.

    Args:
        display_format: The desired output format enum value

    Returns:
        str: XML-formatted instructions for the specified output format.
            Returns empty string if format is NONE.
    """
    if display_format == DisplayOutputFormat.MD:
        return """<output_instructions>
    <instruction>Output properly formatted Markdown.</instruction>
    <instruction>Use table / list formatting when applicable or requested.</instruction>
    <instruction>Do not include an opening ```markdown or closing ```</instruction>
</output_instructions>
"""
    if display_format == DisplayOutputFormat.JSON:
        return """<output_instructions>
    <instruction>Output proper JSON.</instruction>
    <instruction>Use a schema if provided.</instruction>
    <instruction>Only output JSON. Do not include any other text / markdown or formatting such as opening ```json or closing ```</instruction>
</output_instructions>
"""
    if display_format == DisplayOutputFormat.CSV:
        return """<output_instructions>
    <instruction>Output proper CSV format.</instruction>
    <instruction>Ensure you use double quotes on fields containing line breaks or commas.</instruction>
    <instruction>Include a header with names of the fields.</instruction>
    <instruction>Only output the CSV header and data.</instruction>
    <instruction>Do not include any other text / Markdown such as opening ```csv or closing ```</instruction>
</output_instructions>
"""
    if display_format == DisplayOutputFormat.PLAIN:
        return """<output_instructions>
    <instruction>Output plain text without formatting, do not include any other formatting such as markdown.</instruction>
</output_instructions>
"""
    return ""


def display_formatted_output(content: str, display_format: DisplayOutputFormat, console: Console | None = None) -> None:
    """Display content in the specified format using Rich console output.

    Formats and displays content according to the specified DisplayOutputFormat.
    Supports plain text, Markdown, CSV table, and syntax-highlighted JSON output.

    Args:
        content: The content string to display
        display_format: The format to use for displaying the content
        console: Optional Rich Console instance to use. If None, uses console_err

    Returns:
        None

    Note:
        For CSV format, content is converted to a Rich Table before display.
        For JSON format, syntax highlighting is applied.
        If display_format is NONE, nothing is displayed.
    """
    if display_format == DisplayOutputFormat.NONE:
        return

    if not console:
        console = console_err

    if display_format == DisplayOutputFormat.PLAIN:
        print(content)
    elif display_format == DisplayOutputFormat.MD:
        console.print(Markdown(content))
    elif display_format == DisplayOutputFormat.CSV:
        # Convert CSV to rich Table
        table = Table(title="CSV Data")
        csv_reader = csv.reader(io.StringIO(content))
        headers = next(csv_reader)
        for header in headers:
            table.add_column(header, style="cyan")
        for row in csv_reader:
            table.add_row(*row)
        console.print(table)
    elif display_format == DisplayOutputFormat.JSON:
        console.print(Syntax(content, "json"))
