"""
Utility functions and decorators for the par_ai_core package.

This package was split from the original monolithic ``utils.py`` module (ARC-013)
into cohesive submodules while preserving the full public API. Every public name
that ``par_ai_core.utils`` previously exports is re-exported here, so all of
the following continue to work unchanged::

    from par_ai_core.utils import md5_hash, is_url, gather_files_for_context
    import par_ai_core.utils

    par_ai_core.utils.is_url("https://example.com")

Submodule layout:

- :mod:`par_ai_core.utils.hashing` — md5/sha1/sha256 hashes, ``hash_list_by_key``
- :mod:`par_ai_core.utils.urls` — ``is_url``, suffix helpers, ``extract_url_auth``
- :mod:`par_ai_core.utils.strings` — case conversions, ``str_ellipsis``, ``id_generator``
- :mod:`par_ai_core.utils.type_checks` — ``is_float``/``is_int``/``is_date``/``is_zero``/...
- :mod:`par_ai_core.utils.data` — ``coalesce``, ``chunks``, ``nested_get``, CSV/dict helpers
- :mod:`par_ai_core.utils.files` — file listing, context gathering, env-file reading
- :mod:`par_ai_core.utils.markdown` — ``md`` (HTML→Markdown via html2text)
- :mod:`par_ai_core.utils.shell` — ``run_shell_cmd``, ``run_cmd``
- :mod:`par_ai_core.utils.misc` — ``all_subclasses``, context managers, ``DECIMAL_PRECISION``
"""

from __future__ import annotations

from par_ai_core.utils.data import (
    chunks,
    coalesce,
    dict_keys_to_lower,
    json_serial,
    nested_get,
    output_to_dicts,
    parse_csv_text,
)
from par_ai_core.utils.files import (
    add_module_path,
    code_frontend_file_globs,
    code_java_file_globs,
    code_js_file_globs,
    code_python_file_globs,
    code_rust_file_globs,
    gather_files_for_context,
    get_file_list_for_context,
    get_files,
    read_env_file,
    read_text_file_to_stringio,
)
from par_ai_core.utils.hashing import (
    hash_list_by_key,
    md5_hash,
    sha1_hash,
    sha256_hash,
)
from par_ai_core.utils.markdown import md
from par_ai_core.utils.misc import (
    DECIMAL_PRECESSION,
    DECIMAL_PRECISION,
    all_subclasses,
    catch_to_logger,
    suppress_output,
    timer_block,
)
from par_ai_core.utils.shell import (
    run_cmd,
    run_shell_cmd,
)
from par_ai_core.utils.strings import (
    camel_to_snake,
    detect_syntax,
    id_generator,
    str_ellipsis,
    to_camel_case,
    to_class_case,
)
from par_ai_core.utils.type_checks import (
    has_stdin_content,
    has_value,
    is_date,
    is_float,
    is_int,
    is_valid_uuid_v4,
    is_zero,
    non_zero,
)
from par_ai_core.utils.urls import (
    extract_url_auth,
    get_file_suffix,
    get_url_file_suffix,
    is_url,
)

__all__ = [
    "DECIMAL_PRECESSION",
    "DECIMAL_PRECISION",
    "add_module_path",
    "all_subclasses",
    "camel_to_snake",
    "catch_to_logger",
    "chunks",
    "code_frontend_file_globs",
    "code_java_file_globs",
    "code_js_file_globs",
    "code_python_file_globs",
    "code_rust_file_globs",
    "coalesce",
    "detect_syntax",
    "dict_keys_to_lower",
    "extract_url_auth",
    "gather_files_for_context",
    "get_file_list_for_context",
    "get_file_suffix",
    "get_files",
    "get_url_file_suffix",
    "has_stdin_content",
    "has_value",
    "hash_list_by_key",
    "id_generator",
    "is_date",
    "is_float",
    "is_int",
    "is_url",
    "is_valid_uuid_v4",
    "is_zero",
    "json_serial",
    "md",
    "md5_hash",
    "nested_get",
    "non_zero",
    "output_to_dicts",
    "parse_csv_text",
    "read_env_file",
    "read_text_file_to_stringio",
    "run_cmd",
    "run_shell_cmd",
    "sha1_hash",
    "sha256_hash",
    "str_ellipsis",
    "suppress_output",
    "timer_block",
    "to_camel_case",
    "to_class_case",
]
