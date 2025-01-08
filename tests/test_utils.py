"""Tests for utils module."""

from __future__ import annotations

import csv
import io
import os
import sys
from typing import Any


class MockConsole:
    """Mock console for testing display output."""

    def __init__(self):
        self.last_output = ""

    def print(self, content: Any) -> None:
        """Mock print method."""
        self.last_output = str(content)


import tempfile
import time
import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from par_ai_core.utils import (
    add_module_path,
    all_subclasses,
    camel_to_snake,
    catch_to_logger,
    chunks,
    coalesce,
    code_frontend_file_globs,
    code_java_file_globs,
    code_js_file_globs,
    code_python_file_globs,
    code_rust_file_globs,
    detect_syntax,
    dict_keys_to_lower,
    gather_files_for_context,
    get_files,
    has_stdin_content,
    has_value,
    hash_list_by_key,
    id_generator,
    is_date,
    is_float,
    is_int,
    is_valid_uuid_v4,
    is_zero,
    json_serial,
    md,
    md5_hash,
    nested_get,
    non_zero,
    output_to_dicts,
    parse_csv_text,
    read_env_file,
    read_text_file_to_stringio,
    run_cmd,
    run_shell_cmd,
    sha1_hash,
    sha256_hash,
    str_ellipsis,
    suppress_output,
    timer_block,
    to_camel_case,
    to_class_case,
)


def test_has_stdin_content(monkeypatch):
    """Test has_stdin_content function."""
    # Mock sys.stdin.isatty to return True (terminal input)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    assert not has_stdin_content()

    # Test with non-terminal input but no content
    class MockStdin:
        def isatty(self): return False
        def seekable(self): return True
        def tell(self): return 0
        def fileno(self): return 0

    mock_stdin = MockStdin()
    monkeypatch.setattr(sys, "stdin", mock_stdin)
    assert not has_stdin_content()


def test_md():
    """Test md function."""
    html = "<h1>Test</h1><p>Content</p>"
    soup = BeautifulSoup(html, "html.parser")
    result = md(soup)
    assert "Test" in result
    assert "Content" in result


def test_id_generator():
    """Test id_generator function."""
    # Test default length
    result = id_generator()
    assert len(result) == 6
    assert result.isupper() or result.isdigit()

    # Test custom length
    result = id_generator(size=10)
    assert len(result) == 10


def test_json_serial():
    """Test json_serial function."""
    # Test with datetime
    dt = datetime(2024, 1, 1)
    assert json_serial(dt) == "2024-01-01T00:00:00"

    # Test with invalid type
    with pytest.raises(TypeError):
        json_serial(object())


def test_coalesce():
    """Test coalesce function."""
    assert coalesce(None, "", 0, "hello") == ""
    assert coalesce(None, None, 42) == 42
    assert coalesce(None, None, None) is None


def test_to_camel_case():
    """Test to_camel_case function."""
    assert to_camel_case("hello_world") == "helloWorld"
    assert to_camel_case("test_case_conversion") == "testCaseConversion"
    assert to_camel_case("already_camel") == "alreadyCamel"


def test_to_class_case():
    """Test to_class_case function."""
    assert to_class_case("hello_world") == "HelloWorld"
    assert to_class_case("hello world") == "HelloWorld"
    assert to_class_case("test_case") == "TestCase"


def test_is_float():
    """Test is_float function."""
    assert is_float("3.14")
    assert is_float("-2.5")
    assert is_float("42")
    assert not is_float("abc")
    assert not is_float(None)


def test_is_int():
    """Test is_int function."""
    assert is_int("42")
    assert is_int("-17")
    assert not is_int("3.14")
    assert not is_int("abc")
    assert not is_int(None)


def test_is_date():
    """Test is_date function."""
    assert is_date("2024/01/20")
    assert is_date("2024-01-20", fmt="%Y-%m-%d")
    assert not is_date("invalid")
    assert not is_date("2024/13/45")


def test_has_value():
    """Test has_value function."""
    test_dict = {"a": 1, "b": {"c": "test"}, "d": [1, 2, "search"]}
    assert has_value(test_dict, "test")
    assert has_value(test_dict, "search")
    assert not has_value(test_dict, "nonexistent")

    # Test depth limit
    deep_dict = {"a": {"b": {"c": {"d": {"e": "test"}}}}}
    assert not has_value(deep_dict, "test")  # Too deep (>4 levels)

    # Test numeric comparisons
    num_dict = {"int": 42, "float": 3.14159}
    assert has_value(num_dict, "42")
    assert has_value(num_dict, "3.14")
    assert has_value(num_dict, "42.00")  # Test .00 stripping for ints


def test_is_zero():
    """Test is_zero function."""
    assert is_zero(0)
    assert is_zero(0.0)
    assert is_zero(Decimal("0"))
    assert is_zero(Decimal("0.00000"))
    assert not is_zero(Decimal("0.00001"))
    assert not is_zero(1)
    assert not is_zero(None)
    assert not is_zero("0")  # Test with string parameter


def test_non_zero():
    """Test non_zero function."""
    assert non_zero(1)
    assert non_zero(0.1)
    assert non_zero(Decimal("1"))
    assert not non_zero(0)
    assert non_zero(None)


def test_dict_keys_to_lower():
    """Test dict_keys_to_lower function."""
    test_dict = {"Upper": 1, "CASE": 2, "MiXeD": 3}
    result = dict_keys_to_lower(test_dict)
    assert result == {"upper": 1, "case": 2, "mixed": 3}


def test_is_valid_uuid_v4():
    """Test is_valid_uuid_v4 function."""
    valid_uuid = str(uuid.uuid4())
    assert is_valid_uuid_v4(valid_uuid)
    assert not is_valid_uuid_v4("invalid-uuid")
    assert not is_valid_uuid_v4("123e4567-e89b-12d3-a456-426614174000")  # Not v4


def test_parse_csv_text():
    """Test parse_csv_text function."""
    # Test with header
    csv_data = io.StringIO("name,age\nJohn,30\nJane,25")
    result = parse_csv_text(csv_data)
    assert len(result) == 2
    assert result[0]["name"] == "John"
    assert result[0]["age"] == "30"

    # Test without header
    csv_data = io.StringIO("John,30\nJane,25")
    result = parse_csv_text(csv_data, has_header=False)
    assert len(result) == 2
    assert result[0]["0"] == "John"
    assert result[0]["1"] == "30"
    assert result[1]["0"] == "Jane"
    assert result[1]["1"] == "25"

    # Test with empty data
    csv_data = io.StringIO("")
    result = parse_csv_text(csv_data)
    assert len(result) == 0

    # Test with empty data and has_header=False
    csv_data = io.StringIO("")
    result = parse_csv_text(csv_data, has_header=False)
    assert len(result) == 0

    # Test with malformed CSV data that should raise csv.Error
    csv_data = io.StringIO('name,age\n"John"invalid"row",30\nJane,25')  # Invalid quote
    with pytest.raises(csv.Error) as exc_info:
        parse_csv_text(csv_data)
    assert "Error parsing CSV data" in str(exc_info.value)

    # Test with malformed CSV data and no header that should raise csv.Error
    csv_data = io.StringIO('"John"invalid"row",30\nJane,25')  # Invalid quote
    with pytest.raises(csv.Error) as exc_info:
        parse_csv_text(csv_data, has_header=False)
    assert "Error parsing CSV data" in str(exc_info.value)


def test_read_text_file_to_stringio():
    """Test read_text_file_to_stringio function."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
        f.write("test content")
        f.flush()
        result = read_text_file_to_stringio(f.name)
        assert result.getvalue() == "test content"
    os.unlink(f.name)


def test_hash_functions():
    """Test hash functions."""
    test_str = "test string"

    # MD5
    md5_result = md5_hash(test_str)
    assert len(md5_result) == 32
    assert md5_result.isalnum()

    # SHA1
    sha1_result = sha1_hash(test_str)
    assert len(sha1_result) == 40
    assert sha1_result.isalnum()

    # SHA256
    sha256_result = sha256_hash(test_str)
    assert len(sha256_result) == 64
    assert sha256_result.isalnum()


def test_nested_get():
    """Test nested_get function."""
    test_dict = {"a": {"b": {"c": "value"}}}
    assert nested_get(test_dict, "a.b.c") == "value"
    assert nested_get(test_dict, ["a", "b", "c"]) == "value"
    assert nested_get(test_dict, "a.b.d") is None


def test_str_ellipsis():
    """Test str_ellipsis function."""
    assert str_ellipsis("test", 6) == "test  "
    assert str_ellipsis("test string", 7) == "test..."
    assert str_ellipsis("test", 4) == "test"
    assert str_ellipsis("test string", 7, pad_char="") == "test..."
    assert str_ellipsis("test", 7, pad_char="") == "test"


def test_camel_to_snake():
    """Test camel_to_snake function."""
    assert camel_to_snake("camelCase") == "camel_case"
    assert camel_to_snake("ThisIsATest") == "this_is_a_test"
    assert camel_to_snake("ABC") == "abc"


def test_chunks():
    """Test chunks function."""
    test_list = [1, 2, 3, 4, 5, 6]
    result = list(chunks(test_list, 2))
    assert result == [[1, 2], [3, 4], [5, 6]]

    # Test with chunk size larger than list
    result = list(chunks(test_list, 10))
    assert result == [[1, 2, 3, 4, 5, 6]]


def test_get_files():
    """Test get_files function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        Path(tmpdir, "test1.txt").write_text("test1")
        Path(tmpdir, "test2.txt").write_text("test2")
        Path(tmpdir, "test.py").write_text("test")

        # Test without extension filter
        files = get_files(tmpdir)
        assert len(files) == 3
        assert "test1.txt" in files
        assert "test2.txt" in files
        assert "test.py" in files

        # Test with extension filter
        files = get_files(tmpdir, ext=".txt")
        assert len(files) == 1
        assert "test.py" in files


def test_detect_syntax():
    """Test detect_syntax function."""
    # Test bash script
    text = "#!/bin/bash\necho 'hello'"
    assert detect_syntax(text) == "bash"

    # Test sh script
    text = "#!/bin/sh\necho 'hello'"
    assert detect_syntax(text) == "bash"

    # Test non-script text
    text = "Hello World"
    assert detect_syntax(text) is None


def test_hash_list_by_key():
    """Test hash_list_by_key function."""
    test_data = [{"message_id": "1", "content": "first"}, {"message_id": "2", "content": "second"}]
    result = hash_list_by_key(test_data)
    assert result == {"1": {"message_id": "1", "content": "first"}, "2": {"message_id": "2", "content": "second"}}

    # Test with custom key
    result = hash_list_by_key(test_data, id_key="content")
    assert result == {
        "first": {"message_id": "1", "content": "first"},
        "second": {"message_id": "2", "content": "second"},
    }


def test_output_to_dicts():
    """Test output_to_dicts function."""
    # Test valid tab-delimited output
    output = "Name\tAge\nJohn\t30\nJane\t25"
    result = output_to_dicts(output)
    assert result == [{"name": "John", "age": "30"}, {"name": "Jane", "age": "25"}]

    # Test empty output
    assert output_to_dicts("") == []


def test_all_subclasses():
    """Test all_subclasses function."""

    class Parent:
        pass

    class Child1(Parent):
        pass

    class Child2(Parent):
        pass

    class GrandChild(Child1):
        pass

    result = all_subclasses(Parent)
    assert len(result) == 3
    assert Child1 in result
    assert Child2 in result
    assert GrandChild in result


@pytest.mark.parametrize(
    "file_patterns,expected",
    [
        ([], "<files>\n</files>\n"),
        (["/nonexistent/*.txt"], "<files>\n</files>\n"),
    ],
)
def test_gather_files_for_context_empty(file_patterns, expected):
    """Test gather_files_for_context with empty or invalid patterns."""
    result = gather_files_for_context(file_patterns)
    assert result == expected


def test_gather_files_for_context():
    """Test gather_files_for_context with actual files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("test content")

        # Create some files that should be ignored
        (Path(tmpdir) / ".git" / "config").parent.mkdir(exist_ok=True)
        (Path(tmpdir) / ".git" / "config").write_text("git config")

        result = gather_files_for_context([str(test_file)])

        assert "<files>" in result
        assert "<source>" in result
        assert "test.txt" in result
        assert "test content" in result
        assert ".git/config" not in result

        # Test with max_context_length
        result_limited = gather_files_for_context([str(test_file)], max_context_length=10)
        assert len(result_limited) < len(result)

        # Test with empty pattern list
        result_empty = gather_files_for_context([])
        assert result_empty == "<files>\n</files>\n"

        # Test with invalid pattern
        result_invalid = gather_files_for_context(["/nonexistent/*.xyz"])
        assert result_invalid == "<files>\n</files>\n"

        # Test with negative max_context_length
        result_negative = gather_files_for_context([str(test_file)], max_context_length=-1)
        assert "<files>" in result_negative
        assert len(result_negative) > len(result_limited)

        # Test with Path object pattern
        result_path = gather_files_for_context([Path(tmpdir) / "*.txt"])
        assert "<files>" in result_path
        assert "test.txt" in result_path


def test_gather_files_for_context_with_special_chars():
    """Test gather_files_for_context handles special characters correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text('<script>alert("xss");</script>')

        result = gather_files_for_context([str(test_file)])

        assert "&lt;script&gt;" in result
        assert "&quot;" in result
        assert "<script>" not in result


def test_run_cmd_success():
    """Test run_cmd with successful command."""
    result = run_cmd(["echo", "test"])
    assert result == "test"


def test_run_cmd_failure():
    """Test run_cmd with failing command."""
    console = MockConsole()
    result = run_cmd(["nonexistentcommand"], console=console, check=False)
    assert result is None
    assert "Error running command" in console.last_output


def test_run_cmd_with_stderr():
    """Test run_cmd handling stderr output."""
    console = MockConsole()
    result = run_cmd(
        ["python", "-c", "import sys; sys.stderr.write('error\\n'); sys.exit(1)"], console=console, check=False
    )
    assert result is None
    assert "error" in console.last_output


def test_run_cmd_no_console():
    """Test run_cmd works with no console parameter."""
    # Should use console_err as default
    result = run_cmd(["python", "-c", "import sys; sys.stderr.write('test error\\n'); sys.exit(1)"], check=False)
    assert result is None


def test_run_cmd_no_console_file_not_found():
    """Test run_cmd handles FileNotFoundError with no console parameter."""
    # Should use console_err as default when FileNotFoundError occurs
    result = run_cmd(["nonexistent_command_123"], check=False)
    assert result is None


def test_run_cmd_called_process_error():
    """Test run_cmd handles CalledProcessError correctly."""
    # Using python to raise a CalledProcessError by exiting with status 1
    result = run_cmd(["python", "-c", "import sys; sys.exit(1)"])
    assert result is None


def test_run_cmd_file_not_found():
    """Test run_cmd with nonexistent command."""
    console = MockConsole()
    result = run_cmd(["nonexistentcommand123"], console=console, check=False)
    assert result is None
    assert "Error running command" in console.last_output


def test_run_cmd_success_with_console():
    """Test run_cmd with successful command and console output."""
    console = MockConsole()
    result = run_cmd(["echo", "test"], console=console)
    assert result == "test"


def test_suppress_output():
    """Test suppress_output context manager."""
    with suppress_output():
        print("This should not be visible")
        sys.stderr.write("This error should not be visible")


def test_timer_block():
    """Test timer_block context manager."""
    # Test with console
    console = MockConsole()
    with timer_block("Test Timer", console=console):
        time.sleep(0.1)
    assert "Test Timer took" in console.last_output

    # Test without console (should use console_err)
    with timer_block("Test Timer"):
        time.sleep(0.1)


def test_catch_to_logger():
    """Test catch_to_logger context manager."""

    class MockLogger:
        def __init__(self):
            self.last_exception = None

        def exception(self, e):
            self.last_exception = e

    logger = MockLogger()

    # Test with re_throw=False
    with catch_to_logger(logger):
        raise ValueError("Test error")
    assert isinstance(logger.last_exception, ValueError)

    # Test with re_throw=True
    with pytest.raises(ValueError):
        with catch_to_logger(logger, re_throw=True):
            raise ValueError("Test error")

    # Test with logger=None
    with pytest.raises(ValueError):
        with catch_to_logger(None):
            raise ValueError("Test error")


def test_add_module_path():
    """Test add_module_path context manager."""
    test_path = "/tmp/test_path"
    with add_module_path(test_path):
        assert test_path in sys.path
    assert test_path not in sys.path


def test_run_shell_cmd():
    """Test run_shell_cmd function."""
    # Test valid command
    result = run_shell_cmd("echo 'test'")
    assert result == "test"

    # Test invalid command
    result = run_shell_cmd("nonexistentcommand")
    assert result is None

    # Test command with error output
    result = run_shell_cmd("python -c 'import sys; sys.stderr.write(\"error\\n\"); sys.exit(1)'")
    assert result is None


def test_run_cmd():
    """Test run_cmd function."""
    # Test valid command
    result = run_cmd(["echo", "test"])
    assert result == "test"

    # Test invalid command
    console = MockConsole()
    result = run_cmd(["nonexistentcommand"], console=console, check=False)
    assert result is None
    assert "Error running command" in console.last_output


def test_code_file_globs():
    """Test code file glob pattern constants."""
    assert isinstance(code_python_file_globs, list)
    assert isinstance(code_js_file_globs, list)
    assert isinstance(code_frontend_file_globs, list)
    assert isinstance(code_rust_file_globs, list)
    assert isinstance(code_java_file_globs, list)

    # Check that all globs are strings or Path objects
    for glob_list in [
        code_python_file_globs,
        code_js_file_globs,
        code_frontend_file_globs,
        code_rust_file_globs,
        code_java_file_globs,
    ]:
        for pattern in glob_list:
            assert isinstance(pattern, (str, Path))

    # Check specific patterns
    assert "./**/*.py" in code_python_file_globs
    assert "./**/*.js" in code_js_file_globs
    assert "./**/*.html" in code_frontend_file_globs
    assert "./**/*.rs" in code_rust_file_globs
    assert "./**/*.java" in code_java_file_globs


def test_read_env_file():
    """Test read_env_file function."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
        f.write("KEY1=value1\nKEY2=value2\n# Comment\nINVALID_LINE\n")
        f.flush()
        result = read_env_file(f.name)
        assert result == {"KEY1": "value1", "KEY2": "value2"}

        # Test with nonexistent file
        assert read_env_file("nonexistent.env") == {}

        # Test with invalid line format (should be silently skipped)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f2:
            f2.write("INVALID\n")
            f2.flush()
            console = MockConsole()
            result = read_env_file(f2.name, console=console)
            assert result == {}  # Invalid line should be skipped
            assert "Invalid line format" in console.last_output
        os.unlink(f2.name)

        # Test with empty file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f3:
            f3.write("")
            f3.flush()
            result = read_env_file(f3.name)
            assert result == {}
        os.unlink(f3.name)
    os.unlink(f.name)


def test_run_cmd_file_not_found():
    """Test run_cmd with nonexistent command."""
    console = MockConsole()
    result = run_cmd(["nonexistentcommand123"], console=console, check=False)
    assert result is None
    assert "Error running command" in console.last_output


def test_run_cmd_with_stderr():
    """Test run_cmd handling stderr output."""
    console = MockConsole()
    result = run_cmd(
        ["python", "-c", "import sys; sys.stderr.write('error\\n'); sys.exit(1)"], console=console, check=False
    )
    assert result is None
    assert "error" in console.last_output


def test_gather_files_for_context_with_special_chars():
    """Test gather_files_for_context handles special characters correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text('<script>alert("xss");</script>')

        result = gather_files_for_context([str(test_file)])

        assert "&lt;script&gt;" in result
        assert "&quot;" in result
        assert "<script>" not in result


def test_gather_files_for_context_with_invalid_pattern():
    """Test gather_files_for_context with invalid pattern."""
    result = gather_files_for_context(["/nonexistent/*.xyz"])
    assert result == "<files>\n</files>\n"


def test_gather_files_for_context_with_negative_length():
    """Test gather_files_for_context with negative max_context_length."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("test content")
        result = gather_files_for_context([str(test_file)], max_context_length=-1)
        assert "<files>" in result
        assert "test content" in result


def test_gather_files_for_context_with_path_object():
    """Test gather_files_for_context with Path object pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("test content")
        result = gather_files_for_context([Path(tmpdir) / "*.txt"])
        assert "<files>" in result
        assert "test content" in result


def test_gather_files_for_context_with_hidden_files():
    """Test gather_files_for_context properly ignores hidden directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a .git directory with a file
        git_dir = Path(tmpdir) / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("git config")

        # Create a regular file
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("test content")

        result = gather_files_for_context([str(Path(tmpdir) / "*")])
        assert "git config" not in result
        assert "test content" in result


def test_gather_files_for_context_skips_node_modules():
    """Test gather_files_for_context skips node_modules directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a nested .git directory structure
        node_modules_dir = Path(tmpdir) / "repo" / "node_modules"
        node_modules_dir.mkdir(parents=True)
        (node_modules_dir / "package.json").write_text("{}")

        # Create a regular file at the same level
        test_file = Path(tmpdir) / "repo" / "test.txt"
        test_file.parent.mkdir(exist_ok=True)
        test_file.write_text("test content")

        result = gather_files_for_context([str(Path(tmpdir) / "repo" / "**/*")])
        assert "{}" not in result
        assert "test content" in result
