#!/usr/bin/env python3
"""
CSV to PostgreSQL SQL converter.

Do not remove this pseudocode block. Whenever the program code changes, update this
docstring so the pseudocode stays aligned with the implementation.

Pseudo code
===========

MAIN
    args = parse_command_line_arguments()
    validate_arguments(args)
    args.delimiter, args.quote_char = normalize_cli_csv_options(args.delimiter, args.quote_char)

    input_text, detected_encoding = read_file_with_encoding_detection(
        args.input,
        args.encoding
    )

    csv_dialect = detect_csv_dialect(
        input_text,
        preferred_delimiters = [",", ";", "\\t", "|"],
        preferred_quote_chars = ['"']
    )

    rows = parse_csv_to_list_of_lists(input_text, csv_dialect)
    if rows is empty
        raise error "Input file contains no data"

    if args.has_header == yes
        raw_headers = rows[0]
        data_rows = rows[1:]
    else
        raw_headers = generate_headers_from_first_data_row(rows[0])
        data_rows = rows

    column_names = normalize_sql_column_names(raw_headers)
    validate_column_count_consistency(data_rows, expected_count = len(column_names))

    column_profiles = initialize_column_profiles(column_names)
    for each row in data_rows
        for each column_index, cell_value in row
            normalized_value = normalize_cell_for_profiling(cell_value)
            update_special_character_stats(column_profiles[column_index], cell_value)
            update_null_stats(column_profiles[column_index], normalized_value)
            update_length_stats(column_profiles[column_index], normalized_value)
            update_candidate_type_stats(column_profiles[column_index], normalized_value)
            update_numeric_precision_stats(column_profiles[column_index], normalized_value)
            update_uniqueness_stats(column_profiles[column_index], normalized_value)

    inferred_types = []
    for each profile in column_profiles
        inferred_types.append(infer_postgres_type(profile, args.text_mode))

    pk_definition = build_primary_key_definition(
        auto_pk = args.auto_pk,
        pk_column = args.pk_column,
        column_names = column_names,
        column_profiles = column_profiles
    )
    index_definitions = build_index_definitions(
        requested_indexes = args.index,
        column_names = column_names
    )

    sql_parts = []
    if args.drop_before_create == yes and args.create_table == yes
        sql_parts.append(generate_drop_table_sql(args.table))

    if args.create_table == yes
        sql_parts.append(generate_create_table_sql(
            table_name = args.table,
            column_names = column_names,
            inferred_types = inferred_types,
            pk_definition = pk_definition,
            auto_pk = args.auto_pk
        ))
        sql_parts.extend(generate_index_sql(args.table, index_definitions))

    if args.truncate_before_insert == yes and args.insert_table == yes
        sql_parts.append(generate_truncate_table_sql(args.table))

    if args.insert_table == yes
        if args.insert_mode == "insert"
            sql_parts.append(generate_insert_statements(
                table_name = args.table,
                column_names = column_names,
                inferred_types = inferred_types,
                data_rows = data_rows,
                null_tokens = args.null_tokens,
                batch_size = args.batch
            ))
        else if args.insert_mode == "copy"
            sql_parts.append(generate_copy_block(...))

    final_sql = join_sql_parts(sql_parts)

    if args.target == "screen"
        print(final_sql)
    else if args.target == "file"
        write_output_file(args.output, final_sql)
    else if args.target == "execute"
        execute_sql_on_postgres(final_sql, connection_settings)
        if args.insert_mode == "copy"
            stream copy data with PostgreSQL COPY protocol instead of executing "\\." script text
"""

from __future__ import annotations

import argparse
import csv
import io
import keyword
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

try:
    import psycopg
except ImportError:  # pragma: no cover - optional dependency
    psycopg = None


POSTGRES_RESERVED_WORDS = {
    "all", "analyse", "analyze", "and", "any", "array", "as", "asc", "asymmetric",
    "authorization", "between", "binary", "both", "case", "cast", "check", "collate",
    "column", "constraint", "create", "current_catalog", "current_date",
    "current_role", "current_time", "current_timestamp", "current_user", "default",
    "deferrable", "desc", "distinct", "do", "else", "end", "except", "false", "fetch",
    "for", "foreign", "from", "grant", "group", "having", "in", "initially", "intersect",
    "into", "leading", "limit", "localtime", "localtimestamp", "new", "not", "null",
    "off", "offset", "old", "on", "only", "or", "order", "placing", "primary",
    "references", "returning", "select", "session_user", "some", "symmetric", "table",
    "then", "to", "trailing", "true", "union", "unique", "user", "using", "variadic",
    "when", "where", "window", "with",
}
BOOL_TRUE_TOKENS = {"true", "t", "yes", "y", "1"}
BOOL_FALSE_TOKENS = {"false", "f", "no", "n", "0"}
BOOLEAN_NAME_HINTS = ("is_", "has_", "can_", "should_", "active", "enabled", "deleted", "valid")
DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y")
TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%f",
)
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


@dataclass
class CsvDialectInfo:
    delimiter: str
    quotechar: str
    lineterminator: str = "\n"


@dataclass
class ColumnProfile:
    name: str
    null_count: int = 0
    non_null_count: int = 0
    max_length: int = 0
    distinct_values: set[str] = field(default_factory=set)
    special_characters: Counter[str] = field(default_factory=Counter)
    samples: list[str] = field(default_factory=list)
    saw_letters: bool = False
    has_leading_zero_values: bool = False
    min_integer_value: int | None = None
    max_integer_value: int | None = None
    max_digits_left_of_decimal: int = 0
    max_digits_right_of_decimal: int = 0
    max_total_digits: int = 0
    matches_boolean: bool = True
    matches_integer: bool = True
    matches_decimal: bool = True
    matches_date: bool = True
    matches_timestamp: bool = True
    saw_decimal_comma: bool = False
    saw_decimal_dot: bool = False
    saw_textual_boolean: bool = False


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CSV data to PostgreSQL SQL.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--target", choices=("screen", "file", "execute"), default="screen")
    parser.add_argument("--output", help="Output .sql path when --target file.")
    parser.add_argument("--table", required=True, help="Target table name, optionally schema-qualified.")
    parser.add_argument("--auto-pk", choices=("yes", "no"), default="no")
    parser.add_argument("--pk-column", help="Existing column to use as primary key.")
    parser.add_argument("--has-header", choices=("yes", "no"), default="yes")
    parser.add_argument("--create-table", choices=("yes", "no"), default="yes")
    parser.add_argument("--insert-table", choices=("yes", "no"), default="yes")
    parser.add_argument("--drop-before-create", choices=("yes", "no"), default="no")
    parser.add_argument("--truncate-before-insert", choices=("yes", "no"), default="no")
    parser.add_argument("--delimiter", default="auto", help='Delimiter: auto|,|;|\\t||')
    parser.add_argument("--quote-char", default="auto", help='Quote char: auto|"')
    parser.add_argument("--encoding", default="auto", help="Encoding: auto|utf-8|utf-8-sig|cp1252")
    parser.add_argument("--text-mode", choices=("text", "varchar"), default="text")
    parser.add_argument("--insert-mode", choices=("insert", "copy"), default="insert")
    parser.add_argument("--batch", type=int, help="Rows per INSERT statement when --insert-mode insert.")
    parser.add_argument(
        "--index",
        action="append",
        nargs="+",
        metavar="COLUMN",
        help="Create an index on one or more normalized column names. Repeat for multiple indexes.",
    )
    parser.add_argument("--null-tokens", nargs="*", default=["", "NULL", "null"])
    parser.add_argument("--postgres-host")
    parser.add_argument("--postgres-port", type=int, default=5432)
    parser.add_argument("--postgres-db")
    parser.add_argument("--postgres-user")
    parser.add_argument("--postgres-password")
    parser.add_argument("--postgres-schema")
    parser.add_argument(
        "--postgres-sslmode",
        choices=("disable", "prefer", "require", "verify-ca", "verify-full"),
        default="prefer",
    )
    return parser.parse_args(argv)


def validate_arguments(args: argparse.Namespace) -> None:
    if args.target == "file" and not args.output:
        raise ValueError("--output is required when --target file")
    if args.target == "execute":
        required = {
            "--postgres-host": args.postgres_host,
            "--postgres-db": args.postgres_db,
            "--postgres-user": args.postgres_user,
            "--postgres-password": args.postgres_password,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing PostgreSQL connection settings: {', '.join(missing)}")
    if args.auto_pk == "yes" and args.pk_column:
        raise ValueError("--auto-pk yes and --pk-column cannot be used together")
    if args.drop_before_create == "yes" and args.create_table != "yes":
        raise ValueError("--drop-before-create requires --create-table yes")
    if args.truncate_before_insert == "yes" and args.insert_table != "yes":
        raise ValueError("--truncate-before-insert requires --insert-table yes")
    if args.create_table != "yes" and args.insert_table != "yes":
        raise ValueError("Nothing to do: enable --create-table and/or --insert-table")
    if args.batch is not None and args.batch < 1:
        raise ValueError("--batch must be a positive integer")
    if args.batch is not None and args.insert_mode != "insert":
        raise ValueError("--batch is only supported with --insert-mode insert")


def build_index_definitions(
    requested_indexes: list[list[str]] | None,
    column_names: list[str],
) -> list[tuple[str, ...]]:
    if not requested_indexes:
        return []
    column_set = set(column_names)
    normalized_indexes: list[tuple[str, ...]] = []
    for index_columns in requested_indexes:
        if not index_columns:
            continue
        missing = [column for column in index_columns if column not in column_set]
        if missing:
            raise ValueError(
                f"Index column(s) not found: {', '.join(missing)}. "
                f"Use normalized names: {', '.join(column_names)}"
            )
        normalized_indexes.append(tuple(index_columns))
    return normalized_indexes


def normalize_cli_csv_options(delimiter: str, quote_char: str) -> tuple[str, str]:
    delimiter_map = {r"\t": "\t"}
    quote_map = {r"\"": '"'}
    return delimiter_map.get(delimiter, delimiter), quote_map.get(quote_char, quote_char)


def read_file_with_encoding_detection(path: str, encoding: str) -> tuple[str, str]:
    raw = Path(path).read_bytes()
    candidates = [encoding] if encoding != "auto" else ["utf-8-sig", "utf-8", "cp1252"]
    last_error = None
    for candidate in candidates:
        try:
            return raw.decode(candidate), candidate
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not decode {path}: {last_error}")


def detect_csv_dialect(text: str, delimiter_arg: str, quote_char_arg: str) -> CsvDialectInfo:
    delimiters = [delimiter_arg] if delimiter_arg != "auto" else [",", ";", "\t", "|"]
    quote_chars = [quote_char_arg] if quote_char_arg != "auto" else ['"']
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return CsvDialectInfo(delimiter=delimiters[0], quotechar=quote_chars[0])

    best: tuple[int, int, str, str] | None = None
    sample = "\n".join(lines[:50])
    for delimiter in delimiters:
        for quote_char in quote_chars:
            try:
                reader = csv.reader(io.StringIO(sample), delimiter=delimiter, quotechar=quote_char)
                widths = [len(row) for row in reader if row]
            except csv.Error:
                continue
            if not widths:
                continue
            stable_width = Counter(widths).most_common(1)[0][0]
            stability_score = sum(width == stable_width for width in widths)
            candidate = (stability_score, stable_width, delimiter, quote_char)
            if best is None or candidate > best:
                best = candidate

    if best is None:
        raise ValueError("Unable to detect CSV dialect")
    _, _, delimiter, quotechar = best
    return CsvDialectInfo(delimiter=delimiter, quotechar=quotechar)


def parse_csv_to_list_of_lists(text: str, dialect: CsvDialectInfo) -> list[list[str]]:
    reader = csv.reader(
        io.StringIO(text),
        delimiter=dialect.delimiter,
        quotechar=dialect.quotechar,
    )
    return [list(row) for row in reader]


def generate_headers_from_first_data_row(first_row: list[str]) -> list[str]:
    return [f"col_{index + 1}" for index in range(len(first_row))]


def normalize_sql_column_names(headers: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: Counter[str] = Counter()
    for index, header in enumerate(headers, start=1):
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", header.strip().lower())
        name = re.sub(r"_+", "_", name).strip("_")
        if not name:
            name = f"col_{index}"
        if name[0].isdigit():
            name = f"col_{name}"
        if name in POSTGRES_RESERVED_WORDS or keyword.iskeyword(name):
            name = f"{name}_col"
        seen[name] += 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        normalized.append(name)
    return normalized


def validate_column_count_consistency(data_rows: list[list[str]], expected_count: int) -> None:
    for row_number, row in enumerate(data_rows, start=1):
        if len(row) != expected_count:
            raise ValueError(
                f"Row {row_number} has {len(row)} columns; expected {expected_count}"
            )


def initialize_column_profiles(column_names: list[str]) -> list[ColumnProfile]:
    return [ColumnProfile(name=name) for name in column_names]


def normalize_cell_for_profiling(value: str, null_tokens: set[str]) -> str | None:
    normalized = sanitize_text(value).strip()
    if normalized in null_tokens:
        return None
    return normalized


def sanitize_text(value: str) -> str:
    return value.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")


def update_special_character_stats(profile: ColumnProfile, raw_value: str) -> None:
    for character in raw_value:
        if ord(character) < 32 or ord(character) == 127 or re.match(r"[^\w\s]", character):
            profile.special_characters[character] += 1
    if raw_value and len(profile.samples) < 3:
        profile.samples.append(raw_value)


def update_null_stats(profile: ColumnProfile, normalized_value: str | None) -> None:
    if normalized_value is None:
        profile.null_count += 1
    else:
        profile.non_null_count += 1


def update_length_stats(profile: ColumnProfile, normalized_value: str | None) -> None:
    if normalized_value is not None:
        profile.max_length = max(profile.max_length, len(normalized_value))


def update_uniqueness_stats(profile: ColumnProfile, normalized_value: str | None) -> None:
    if normalized_value is not None:
        profile.distinct_values.add(normalized_value)


def update_candidate_type_stats(profile: ColumnProfile, normalized_value: str | None) -> None:
    if normalized_value is None:
        return

    if re.search(r"[A-Za-z]", normalized_value):
        profile.saw_letters = True

    lowered = normalized_value.lower()
    if lowered in (BOOL_TRUE_TOKENS | BOOL_FALSE_TOKENS):
        if lowered not in {"0", "1"}:
            profile.saw_textual_boolean = True
    else:
        profile.matches_boolean = False

    if re.fullmatch(r"[+-]?\d+", normalized_value):
        stripped = normalized_value.lstrip("+-")
        if len(stripped) > 1 and stripped.startswith("0"):
            profile.has_leading_zero_values = True
    else:
        profile.matches_integer = False

    decimal_kind = classify_decimal_pattern(normalized_value)
    if decimal_kind == "comma":
        profile.saw_decimal_comma = True
    elif decimal_kind == "dot":
        profile.saw_decimal_dot = True
    elif normalized_value and not re.fullmatch(r"[+-]?\d+", normalized_value):
        profile.matches_decimal = False

    if parse_supported_date(normalized_value) is None:
        profile.matches_date = False
    if parse_supported_timestamp(normalized_value) is None:
        profile.matches_timestamp = False


def classify_decimal_pattern(value: str) -> str | None:
    if re.fullmatch(r"[+-]?\d+,\d+", value):
        return "comma"
    if re.fullmatch(r"[+-]?\d+\.\d+", value):
        return "dot"
    return None


def update_numeric_precision_stats(profile: ColumnProfile, normalized_value: str | None) -> None:
    if normalized_value is None:
        return
    if re.fullmatch(r"[+-]?\d+", normalized_value):
        integer_value = int(normalized_value)
        if profile.min_integer_value is None or integer_value < profile.min_integer_value:
            profile.min_integer_value = integer_value
        if profile.max_integer_value is None or integer_value > profile.max_integer_value:
            profile.max_integer_value = integer_value
        digits = len(normalized_value.lstrip("+-"))
        profile.max_total_digits = max(profile.max_total_digits, digits)
        profile.max_digits_left_of_decimal = max(profile.max_digits_left_of_decimal, digits)
        return

    kind = classify_decimal_pattern(normalized_value)
    if kind is None:
        return
    separator = "," if kind == "comma" else "."
    left, right = normalized_value.lstrip("+-").split(separator, 1)
    profile.max_digits_left_of_decimal = max(profile.max_digits_left_of_decimal, len(left))
    profile.max_digits_right_of_decimal = max(profile.max_digits_right_of_decimal, len(right))
    profile.max_total_digits = max(profile.max_total_digits, len(left) + len(right))


def infer_postgres_type(profile: ColumnProfile, text_mode: str) -> str:
    if profile.non_null_count == 0:
        return "TEXT" if text_mode == "text" else "VARCHAR(1)"

    if (
        profile.matches_boolean
        and (profile.saw_textual_boolean or looks_like_boolean_name(profile.name))
    ):
        return "BOOLEAN"

    if profile.matches_integer:
        if profile.has_leading_zero_values:
            return text_type_for_profile(profile, text_mode)
        min_value = profile.min_integer_value if profile.min_integer_value is not None else 0
        max_value = profile.max_integer_value if profile.max_integer_value is not None else 0
        if INT32_MIN <= min_value <= INT32_MAX and INT32_MIN <= max_value <= INT32_MAX:
            return "INTEGER"
        if INT64_MIN <= min_value <= INT64_MAX and INT64_MIN <= max_value <= INT64_MAX:
            return "BIGINT"
        return f"NUMERIC({max(profile.max_total_digits, 1)},0)"

    if profile.matches_decimal and not (profile.saw_decimal_comma and profile.saw_decimal_dot):
        precision = max(profile.max_total_digits, 1)
        scale = profile.max_digits_right_of_decimal
        return f"NUMERIC({precision},{scale})"

    if profile.matches_date and not profile.matches_timestamp:
        return "DATE"

    if profile.matches_timestamp:
        return "TIMESTAMP"

    return text_type_for_profile(profile, text_mode)


def text_type_for_profile(profile: ColumnProfile, text_mode: str) -> str:
    if text_mode == "varchar":
        return f"VARCHAR({max(profile.max_length, 1)})"
    return "TEXT"


def looks_like_boolean_name(name: str) -> bool:
    return any(name.startswith(prefix) or name == prefix for prefix in BOOLEAN_NAME_HINTS)


def parse_supported_date(value: str) -> date | None:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def parse_supported_timestamp(value: str) -> datetime | None:
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def build_primary_key_definition(
    auto_pk: str,
    pk_column: str | None,
    column_names: list[str],
    column_profiles: list[ColumnProfile],
) -> str | None:
    if auto_pk == "yes":
        return None
    if not pk_column:
        return None
    if pk_column not in column_names:
        raise ValueError(f"Primary key column not found: {pk_column}")
    index = column_names.index(pk_column)
    profile = column_profiles[index]
    if profile.null_count > 0:
        raise ValueError(f"PK column contains NULL values: {pk_column}")
    if len(profile.distinct_values) != profile.non_null_count:
        raise ValueError(f"PK column is not unique: {pk_column}")
    return f"PRIMARY KEY ({quote_identifier(pk_column)})"


def generate_drop_table_sql(table_name: str) -> str:
    return f"DROP TABLE IF EXISTS {format_table_name(table_name)};"


def generate_truncate_table_sql(table_name: str) -> str:
    return f"TRUNCATE TABLE {format_table_name(table_name)};"


def generate_create_table_sql(
    table_name: str,
    column_names: list[str],
    inferred_types: list[str],
    pk_definition: str | None,
    auto_pk: str,
    column_profiles: list[ColumnProfile],
) -> str:
    lines = [f"CREATE TABLE {format_table_name(table_name)} ("]
    definitions: list[str] = []
    if auto_pk == "yes":
        definitions.append("    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY")
    for name, inferred_type, profile in zip(column_names, inferred_types, column_profiles):
        nullability = "NOT NULL" if profile.null_count == 0 else "NULL"
        definitions.append(f"    {quote_identifier(name)} {inferred_type} {nullability}")
    if pk_definition:
        definitions.append(f"    {pk_definition}")
    lines.append(",\n".join(definitions))
    lines.append(");")
    return "\n".join(lines)


def generate_index_sql(table_name: str, index_definitions: list[tuple[str, ...]]) -> list[str]:
    statements: list[str] = []
    for index_columns in index_definitions:
        index_name = build_index_name(table_name, index_columns)
        columns_sql = ", ".join(quote_identifier(column) for column in index_columns)
        statements.append(
            f"CREATE INDEX IF NOT EXISTS {quote_identifier(index_name)} "
            f"ON {format_table_name(table_name)} ({columns_sql});"
        )
    return statements


def build_index_name(table_name: str, index_columns: tuple[str, ...]) -> str:
    base_table_name = table_name.split(".")[-1].strip()
    safe_table_name = re.sub(r"[^a-zA-Z0-9_]+", "_", base_table_name).strip("_") or "table"
    raw_name = f"idx_{safe_table_name}_{'_'.join(index_columns)}"
    return raw_name[:63]


def generate_insert_statements(
    table_name: str,
    column_names: list[str],
    inferred_types: list[str],
    data_rows: list[list[str]],
    null_tokens: set[str],
    batch_size: int | None,
) -> str:
    quoted_columns = ", ".join(quote_identifier(name) for name in column_names)
    statements: list[str] = []
    batch_size = batch_size or 1
    for start in range(0, len(data_rows), batch_size):
        batch_rows = data_rows[start:start + batch_size]
        values_sql: list[str] = []
        for row in batch_rows:
            values = [
                sql_literal(cell, inferred_type, null_tokens)
                for cell, inferred_type in zip(row, inferred_types)
            ]
            values_sql.append(f"({', '.join(values)})")
        statements.append(
            f"INSERT INTO {format_table_name(table_name)} ({quoted_columns}) VALUES\n"
            f"    {',\n    '.join(values_sql)};"
        )
    return "\n".join(statements)


def generate_copy_block(
    table_name: str,
    column_names: list[str],
    data_rows: list[list[str]],
    null_tokens: set[str],
) -> str:
    output = io.StringIO()
    writer = csv.writer(output, delimiter=",", quotechar='"', lineterminator="\n")
    for row in data_rows:
        writer.writerow(
            [r"\N" if normalize_cell_for_profiling(cell, null_tokens) is None else sanitize_text(cell) for cell in row]
        )
    quoted_columns = ", ".join(quote_identifier(name) for name in column_names)
    return (
        f"COPY {format_table_name(table_name)} ({quoted_columns}) FROM STDIN WITH "
        "(FORMAT csv, HEADER false, DELIMITER ',', QUOTE '\"', NULL '\\N');\n"
        f"{output.getvalue()}\\.\n"
    )


def join_sql_parts(sql_parts: list[str]) -> str:
    return "\n\n".join(part for part in sql_parts if part.strip())


def write_output_file(path: str, sql: str) -> None:
    Path(path).write_text(sql, encoding="utf-8")


def execute_sql_on_postgres(
    sql: str,
    args: argparse.Namespace,
    column_names: list[str],
    data_rows: list[list[str]],
    null_tokens: set[str],
) -> None:
    if psycopg is None:
        raise RuntimeError("psycopg is required for --target execute")
    conninfo = {
        "host": args.postgres_host,
        "port": args.postgres_port,
        "dbname": args.postgres_db,
        "user": args.postgres_user,
        "password": args.postgres_password,
        "sslmode": args.postgres_sslmode,
    }
    with psycopg.connect(**conninfo) as conn:
        with conn.cursor() as cur:
            if args.postgres_schema:
                cur.execute(f"SET search_path TO {quote_identifier(args.postgres_schema)}")
            if args.insert_mode == "copy":
                execute_sql_without_copy(cur, sql)
                if args.insert_table == "yes":
                    copy_data_to_postgres(cur, args.table, column_names, data_rows, null_tokens)
            else:
                cur.execute(sql)
        conn.commit()


def execute_sql_without_copy(cur: "psycopg.Cursor", sql: str) -> None:
    statements = [statement.strip() for statement in sql.split(";\n") if statement.strip()]
    for statement in statements:
        if statement.startswith("COPY "):
            continue
        cur.execute(statement)


def copy_data_to_postgres(
    cur: "psycopg.Cursor",
    table_name: str,
    column_names: list[str],
    data_rows: list[list[str]],
    null_tokens: set[str],
) -> None:
    copy_sql = (
        f"COPY {format_table_name(table_name)} "
        f"({', '.join(quote_identifier(name) for name in column_names)}) "
        "FROM STDIN WITH (FORMAT csv, HEADER false, DELIMITER ',', QUOTE '\"', NULL '\\N')"
    )
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=",", quotechar='"', lineterminator="\n")
    for row in data_rows:
        writer.writerow(
            [r"\N" if normalize_cell_for_profiling(cell, null_tokens) is None else sanitize_text(cell) for cell in row]
        )
    with cur.copy(copy_sql) as copy:
        copy.write(buffer.getvalue())


def sql_literal(value: str, inferred_type: str, null_tokens: set[str]) -> str:
    normalized = normalize_cell_for_profiling(value, null_tokens)
    if normalized is None:
        return "NULL"
    if inferred_type == "BOOLEAN":
        return "TRUE" if normalized.lower() in BOOL_TRUE_TOKENS else "FALSE"
    if inferred_type in {"INTEGER", "BIGINT"} or inferred_type.startswith("NUMERIC("):
        return normalize_numeric_literal(normalized)
    if inferred_type == "DATE":
        return f"'{parse_supported_date(normalized).isoformat()}'"
    if inferred_type == "TIMESTAMP":
        return f"'{parse_supported_timestamp(normalized).isoformat(sep=' ')}'"
    return f"'{escape_sql_text(normalized)}'"


def normalize_numeric_literal(value: str) -> str:
    value = value.replace(",", ".")
    try:
        return format(Decimal(value), "f")
    except InvalidOperation as exc:
        raise ValueError(f"Invalid numeric value: {value}") from exc


def escape_sql_text(value: str) -> str:
    return sanitize_text(value).replace("'", "''")


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def format_table_name(table_name: str) -> str:
    return ".".join(quote_identifier(part.strip()) for part in table_name.split(".") if part.strip())


def profile_columns(
    data_rows: list[list[str]],
    column_names: list[str],
    null_tokens: set[str],
) -> list[ColumnProfile]:
    profiles = initialize_column_profiles(column_names)
    for row in data_rows:
        for column_index, cell_value in enumerate(row):
            normalized_value = normalize_cell_for_profiling(cell_value, null_tokens)
            profile = profiles[column_index]
            update_special_character_stats(profile, cell_value)
            update_null_stats(profile, normalized_value)
            update_length_stats(profile, normalized_value)
            update_candidate_type_stats(profile, normalized_value)
            update_numeric_precision_stats(profile, normalized_value)
            update_uniqueness_stats(profile, normalized_value)
    return profiles


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    validate_arguments(args)
    args.delimiter, args.quote_char = normalize_cli_csv_options(args.delimiter, args.quote_char)

    input_text, _detected_encoding = read_file_with_encoding_detection(args.input, args.encoding)
    csv_dialect = detect_csv_dialect(input_text, args.delimiter, args.quote_char)
    rows = parse_csv_to_list_of_lists(input_text, csv_dialect)
    if not rows:
        raise ValueError("Input file contains no data")

    if args.has_header == "yes":
        raw_headers = rows[0]
        data_rows = rows[1:]
    else:
        raw_headers = generate_headers_from_first_data_row(rows[0])
        data_rows = rows

    if not raw_headers:
        raise ValueError("Input file contains no columns")

    column_names = normalize_sql_column_names(raw_headers)
    validate_column_count_consistency(data_rows, len(column_names))
    null_tokens = {sanitize_text(token).strip() for token in args.null_tokens}
    column_profiles = profile_columns(data_rows, column_names, null_tokens)
    inferred_types = [infer_postgres_type(profile, args.text_mode) for profile in column_profiles]
    pk_definition = build_primary_key_definition(
        args.auto_pk,
        args.pk_column,
        column_names,
        column_profiles,
    )
    index_definitions = build_index_definitions(args.index, column_names)

    sql_parts: list[str] = []
    if args.drop_before_create == "yes" and args.create_table == "yes":
        sql_parts.append(generate_drop_table_sql(args.table))
    if args.create_table == "yes":
        sql_parts.append(
            generate_create_table_sql(
                args.table,
                column_names,
                inferred_types,
                pk_definition,
                args.auto_pk,
                column_profiles,
            )
        )
        sql_parts.extend(generate_index_sql(args.table, index_definitions))
    if args.truncate_before_insert == "yes" and args.insert_table == "yes":
        sql_parts.append(generate_truncate_table_sql(args.table))
    if args.insert_table == "yes":
        if args.insert_mode == "insert":
            sql_parts.append(
                generate_insert_statements(
                    args.table,
                    column_names,
                    inferred_types,
                    data_rows,
                    null_tokens,
                    args.batch,
                )
            )
        else:
            sql_parts.append(generate_copy_block(args.table, column_names, data_rows, null_tokens))

    final_sql = join_sql_parts(sql_parts)
    if args.target == "screen":
        print(final_sql)
    elif args.target == "file":
        write_output_file(args.output, final_sql)
    else:
        execute_sql_on_postgres(final_sql, args, column_names, data_rows, null_tokens)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - command-line error path
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
