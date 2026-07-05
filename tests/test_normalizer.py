from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
import sqlite3
import tempfile

import duckdb
import polars as pl
import pytest
import yaml

from spotfire_expr_normalizer import (
    DerivedExpression,
    ExpressionCompatibilityError,
    build_expression_layers,
    canonicalize_expressions,
    catalog_db_path,
    compile_expression_file,
    compile_polars_expression_file,
    load_duckdb_layered_expression_yaml,
    load_polars_layered_expression_yaml,
    normalize_expression,
    normalize_expression_for_polars,
    prepare_duckdb_layered_expression,
    prepare_polars_layered_expression,
    pretty_format_expression,
    unsupported_functions_markdown_path,
    validate_expression_compatibility,
    write_layered_expression_yaml,
)


def test_like_operator_wraps_plain_double_quoted_value() -> None:
    assert normalize_expression('[country]~="KOR"') == '"country" LIKE \'%KOR%\''
    assert normalize_expression("[country]~='K%'") == '"country" LIKE \'K%\''


def test_ampersand_operator_normalizes_to_duckdb_string_concat() -> None:
    assert normalize_expression("[first_name] & ' ' & [last_name]") == '"first_name" || \' \' || "last_name"'
    assert normalize_expression('"prefix-" & [code] & "-suffix"') == "'prefix-' || \"code\" || '-suffix'"


def test_percentile_normalizes_to_duckdb_quantile_cont() -> None:
    assert normalize_expression("Percentile([amount], 90)") == 'quantile_cont("amount", 0.9) OVER ()'
    assert normalize_expression("Percentile([amount], 90) OVER ([country])") == (
        'quantile_cont("amount", 0.9) OVER (PARTITION BY "country")'
    )


def test_common_unsupported_functions_are_now_normalized() -> None:
    assert normalize_expression("Base64Encode([name])") == 'base64(encode("name"))'
    assert normalize_expression("Base64Decode([encoded])") == 'decode(from_base64("encoded"))'
    assert normalize_expression("Concatenate([a], '-', [b])") == 'concat("a", \'-\', "b")'
    assert normalize_expression('Split([full_code], "-", 1)') == 'list_extract(string_split("full_code", \'-\'), 1)'
    assert normalize_expression('Split([full_code], "-")') == 'string_split("full_code", \'-\')'
    assert normalize_expression("Len([name])") == 'length("name")'
    assert normalize_expression("Right([name], 2)") == 'right("name", 2)'
    assert normalize_expression("Mid([name], 2, 3)") == 'substr("name", 2, 3)'
    assert normalize_expression("Substitute([name], 'A', 'B')") == 'replace("name", \'A\', \'B\')'
    assert normalize_expression("FromEpochSeconds([ts])") == 'to_timestamp("ts")'
    assert normalize_expression("FromEpochMilliseconds([ts])") == 'to_timestamp(("ts")::DOUBLE / 1000.0)'
    assert normalize_expression("ToEpochSeconds([dt])") == 'epoch("dt")'
    assert normalize_expression("ToEpochMilliseconds([dt])") == 'epoch_ms("dt")'
    assert normalize_expression("ISOWeek([dt])") == 'week("dt")'
    assert normalize_expression("ISOYear([dt])") == 'isoyear("dt")'
    assert normalize_expression("YearAndWeek([dt])") == 'strftime("dt", \'%G-%V\')'


def test_required_functions_normalize_to_polars_expressions() -> None:
    assert normalize_expression_for_polars("Sum([amount]) OVER ([grp])") == 'pl.col(\'amount\').sum().over(\'grp\')'
    assert normalize_expression_for_polars("Avg([amount]) OVER ([grp])") == 'pl.col(\'amount\').mean().over(\'grp\')'
    assert normalize_expression_for_polars("Min([amount]) OVER ([grp])") == 'pl.col(\'amount\').min().over(\'grp\')'
    assert normalize_expression_for_polars("Max([amount]) OVER ([grp])") == 'pl.col(\'amount\').max().over(\'grp\')'
    assert normalize_expression_for_polars("Abs([total_amount])") == 'pl.col(\'total_amount\').abs()'
    assert normalize_expression_for_polars("String([name])") == 'pl.col(\'name\').cast(pl.String)'
    assert normalize_expression_for_polars("Len([name])") == 'pl.col(\'name\').str.len_chars()'
    assert normalize_expression_for_polars('Split([code], "-", 1)') == "pl.col('code').str.split('-').list.get(0)"
    assert normalize_expression_for_polars("Right([name], 2)") == "pl.col('name').str.slice(-(2))"
    assert normalize_expression_for_polars("[name] & '-' & [code_part]") == (
        "pl.concat_str([pl.col('name'), pl.lit('-'), pl.col('code_part')])"
    )


def test_additional_spotfire_functions_normalize_to_polars_expressions() -> None:
    assert normalize_expression_for_polars("Cast([amount] as Integer)") == "pl.col('amount').cast(pl.Int64)"
    assert normalize_expression_for_polars("Currency([amount])") == "pl.col('amount').cast(pl.Decimal(38, 10))"
    assert normalize_expression_for_polars("SN([value], 0)") == "pl.col('value').fill_null(0)"
    assert normalize_expression_for_polars("IsNull([value])") == "pl.col('value').is_null()"
    assert normalize_expression_for_polars("Mid([name], 2, 3)") == "pl.col('name').str.slice((2) - 1, 3)"
    assert normalize_expression_for_polars("Substitute([name], 'A', 'B')") == (
        "pl.col('name').str.replace_all('A', 'B', literal=True)"
    )
    assert normalize_expression_for_polars("Find('i', [name])") == (
        "pl.when(pl.col('name').str.find('i').is_null()).then(pl.lit(0)).otherwise(pl.col('name').str.find('i') + 1)"
    )
    assert normalize_expression_for_polars("ParseDate([raw_date], '%Y-%m-%d')") == (
        "pl.col('raw_date').str.strptime(pl.Date, format='%Y-%m-%d', strict=False)"
    )
    assert normalize_expression_for_polars("DatePart('year', ParseDate([raw_date], '%Y-%m-%d'))") == (
        "pl.col('raw_date').str.strptime(pl.Date, format='%Y-%m-%d', strict=False).dt.year()"
    )
    assert normalize_expression_for_polars("Q1([amount]) OVER ([grp])") == (
        "pl.col('amount').quantile(0.25, interpolation='linear').over('grp')"
    )
    assert normalize_expression_for_polars("WeightedAverage([amount], [weight]) OVER ([grp])") == (
        "((pl.col('amount') * pl.col('weight')).sum() / pl.col('weight').sum()).over('grp')"
    )
    assert normalize_expression_for_polars("ValueForMax([name], [amount]) OVER ([grp])") == (
        "pl.col('name').sort_by(pl.col('amount')).last().over('grp')"
    )
    assert normalize_expression_for_polars('DenseRank([amount], "desc", [grp])') == (
        "pl.col('amount').rank(method='dense', descending=True).over('grp')"
    )
    assert normalize_expression_for_polars('RankReal([amount], "desc", "ties.method=average", [grp])') == (
        "pl.col('amount').rank(method='average', descending=True).over('grp')"
    )
    assert normalize_expression_for_polars("UniqueConcatenate([name]) OVER ([grp])") == (
        "pl.col('name').cast(pl.String).unique().implode().list.join(',').over('grp')"
    )


def test_common_aggregate_functions_are_window_normalized() -> None:
    assert normalize_expression("Sum([amount])") == 'sum("amount")'
    assert normalize_expression("Average([amount])") == 'avg("amount")'
    assert normalize_expression("Count([amount])") == 'count("amount")'
    assert normalize_expression("UniqueCount([account])") == 'count(DISTINCT "account")'
    assert normalize_expression("Sum([amount]) OVER ([country])") == 'sum("amount") OVER (PARTITION BY "country")'
    assert normalize_expression("Max([amount]) OVER ([country])") == 'max("amount") OVER (PARTITION BY "country")'
    assert normalize_expression("Min([amount]) OVER ([country])") == 'min("amount") OVER (PARTITION BY "country")'
    assert normalize_expression("Median([amount])") == 'median("amount") OVER ()'
    assert normalize_expression("Q1([amount])") == 'quantile_cont("amount", 0.25) OVER ()'
    assert normalize_expression("Q3([amount]) OVER ([country])") == (
        'quantile_cont("amount", 0.75) OVER (PARTITION BY "country")'
    )
    assert normalize_expression("IQR([amount])") == (
        '(quantile_cont("amount", 0.75) OVER () - quantile_cont("amount", 0.25) OVER ())'
    )
    assert normalize_expression("WeightedAverage([amount], [weight]) OVER ([country])") == (
        '(sum(("amount") * ("weight")) OVER (PARTITION BY "country") / '
        'nullif(sum("weight") OVER (PARTITION BY "country"), 0))'
    )
    assert normalize_expression("ValueForMax([label], [amount])") == 'arg_max("label", "amount") OVER ()'
    assert normalize_expression("UniqueConcatenate([label]) OVER ([country])") == (
        'string_agg(DISTINCT CAST("label" AS VARCHAR), \',\') OVER (PARTITION BY "country")'
    )


def test_additional_safe_spotfire_functions_are_normalized() -> None:
    assert normalize_expression("Cast([amount] as Integer)") == 'CAST("amount" AS INTEGER)'
    assert normalize_expression("Currency([amount])") == 'CAST("amount" AS DECIMAL)'
    assert normalize_expression("Days([n])") == '(("n") * INTERVAL 1 DAY)'
    assert normalize_expression("TotalHours(Days([n]))") == '(epoch((("n") * INTERVAL 1 DAY)) / 3600.0)'
    assert normalize_expression("ParseDate([raw_date], '%Y-%m-%d')") == 'strptime("raw_date", \'%Y-%m-%d\')::DATE'
    assert normalize_expression("SN([value], 0)") == 'coalesce("value", 0)'
    assert normalize_expression("Rand()") == "random()"
    assert normalize_expression("RandBetween(1, 3, 7)") == "floor(random() * ((3) - (1) + 1) + (1))"
    assert normalize_expression("StdErr([amount])") == (
        '(stddev_samp("amount") OVER () / sqrt(count("amount") OVER ()))'
    )
    assert normalize_expression("NthLargest([amount], 1)") == (
        'list_extract(list_sort(list("amount") OVER (), \'DESC\'), 1)'
    )


def test_rank_transform_and_fence_functions_are_normalized() -> None:
    assert normalize_expression('DenseRank([amount], "desc", [country])') == (
        'dense_rank() OVER (PARTITION BY "country" ORDER BY "amount" DESC)'
    )
    assert normalize_expression('RankReal([amount], "desc", "ties.method=average", [country])') == (
        '(rank() OVER (PARTITION BY "country" ORDER BY "amount" DESC) + '
        'rank() OVER (PARTITION BY "country" ORDER BY "amount" DESC) + '
        'count(*) OVER (PARTITION BY "country", "amount") - 1)::DOUBLE / 2.0'
    )
    assert normalize_expression("FirstValidAfter([value])") == (
        'coalesce("value", first_value("value" IGNORE NULLS) '
        "OVER (ORDER BY __row_id ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING))"
    )
    assert normalize_expression("LastValidBefore([value])") == (
        'coalesce("value", last_value("value" IGNORE NULLS) '
        "OVER (ORDER BY __row_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW))"
    )
    assert normalize_expression("FiscalMonth([dt], 1)") == (
        'date_part(\'month\', date_add("dt", -((1) * INTERVAL 1 MONTH)))'
    )
    assert normalize_expression("TimeSpan(1, 2, 3, 4, 5)") == (
        "((1) * INTERVAL 1 DAY + (2) * INTERVAL 1 HOUR + (3) * INTERVAL 1 MINUTE + "
        "(4) * INTERVAL 1 SECOND + (5) * INTERVAL 1 MILLISECOND)"
    )
    assert normalize_expression("MedianAbsoluteDeviation([amount])") == 'mad("amount") OVER ()'
    assert normalize_expression("LIF([amount])") == (
        '(quantile_cont("amount", 0.25) OVER () - 1.5 * '
        '(quantile_cont("amount", 0.75) OVER () - quantile_cont("amount", 0.25) OVER ()))'
    )


def test_second_pass_statistical_functions_are_normalized() -> None:
    assert normalize_expression("MeanDeviation(2, -3, 4)") == (
        "list_aggr(list_transform([2, -3, 4], x -> abs(x - list_aggr([2, -3, 4], 'avg'))), 'avg')"
    )
    assert normalize_expression("MeanDeviation([amount])") == (
        'list_aggr(list_transform(list("amount") OVER (), '
        'x -> abs(x - list_aggr(list("amount") OVER (), \'avg\'))), \'avg\')'
    )
    assert normalize_expression("TrimmedMean([amount], 10)") == (
        'list_aggr(list_filter(list("amount") OVER (), '
        'x -> x BETWEEN quantile_cont("amount", 0.05) OVER () '
        'AND quantile_cont("amount", 1 - 0.05) OVER ()), \'avg\')'
    )
    assert normalize_expression("UAV([amount])") == (
        'list_aggr(list_filter(list("amount") OVER (), x -> x <= '
        '((quantile_cont("amount", 0.75) OVER () + 1.5 * '
        '(quantile_cont("amount", 0.75) OVER () - quantile_cont("amount", 0.25) OVER ())))), \'max\')'
    )
    assert normalize_expression("Outliers([amount])").startswith(
        'array_length(list_filter(list("amount") OVER (),'
    )


def test_comments_are_rendered_as_yaml_comments() -> None:
    expression = "Sum([a]) OVER ([g]) // Spotfire comment"
    normalized = normalize_expression(expression)
    assert 'PARTITION BY "g"' in normalized
    assert "//" not in normalized
    assert "//" not in pretty_format_expression(expression)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output = Path(tmp_dir) / "expressions.layered.yaml"
        write_layered_expression_yaml(
            output,
            [[DerivedExpression("total_a", expression, normalized, ["a", "g"])]],
        )
        text = output.read_text(encoding="utf-8")

    assert "# Spotfire comment" in text
    assert "sql_expression:" in text


def test_canonicalize_and_layers_rewrite_duplicate_alias() -> None:
    expressions = [
        DerivedExpression("amount_band", "[amount] >= 100", normalize_expression("[amount] >= 100"), []),
        DerivedExpression("amount_band_alias", "[amount] >= 100", normalize_expression("[amount] >= 100"), []),
        DerivedExpression(
            "risk_flag",
            "[amount_band_alias] = true",
            normalize_expression("[amount_band_alias] = true"),
            ["amount_band_alias"],
        ),
    ]

    canonicalized, rewrite_count = canonicalize_expressions(expressions)
    layers = build_expression_layers(canonicalized)

    assert [item.name for item in canonicalized] == ["amount_band", "risk_flag"]
    assert rewrite_count == 1
    assert canonicalized[1].expression == "[amount_band] = true"
    assert [[item.name for item in layer] for layer in layers] == [["amount_band"], ["risk_flag"]]


def test_compile_expression_file_writes_human_and_duckdb_layered_yaml() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = Path(tmp_dir) / "expressions.csv"
        source.write_text(
            "Column Name,Expression\n"
            "amount_band,CASE WHEN [amount] >= 100 THEN 'HIGH' ELSE 'LOW' END\n"
            "amount_band_alias,CASE WHEN [amount] >= 100 THEN 'HIGH' ELSE 'LOW' END\n"
            "risk_flag,CASE WHEN [amount_band_alias] = 'HIGH' THEN 'Y' ELSE 'N' END\n",
            encoding="utf-8",
        )

        result = compile_expression_file(
            source,
            source_format="csv",
            result_name_field="Column Name",
            sql_expression_field="Expression",
        )

        assert result.layered_yaml_path.exists()
        assert result.duckdb_layered_yaml_path.exists()
        assert result.expression_count_before == 3
        assert result.expression_count_after == 2
        assert result.expression_rewrite_count == 1
        duckdb_text = result.duckdb_layered_yaml_path.read_text(encoding="utf-8")
        assert "format: duckdb_layered_expression" in duckdb_text
        assert "duckdb_sql:" in duckdb_text
        loaded = load_duckdb_layered_expression_yaml(result.duckdb_layered_yaml_path)
        assert [item.name for item in loaded] == ["amount_band", "risk_flag"]


def test_prepare_duckdb_layered_expression_supports_csv_layered_yaml_and_duckdb_yaml() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = Path(tmp_dir) / "expressions.csv"
        source.write_text(
            "Column Name,Expression\n"
            "amount_band,CASE WHEN [amount] >= 100 THEN 'HIGH' ELSE 'LOW' END\n"
            "risk_flag,CASE WHEN [amount_band] = 'HIGH' THEN 'Y' ELSE 'N' END\n",
            encoding="utf-8",
        )

        csv_result = prepare_duckdb_layered_expression(
            source,
            result_name_field="Column Name",
            sql_expression_field="Expression",
        )
        assert csv_result.compiled is True
        assert csv_result.source_format == "csv"
        assert csv_result.layered_yaml_path == source.with_suffix(".layered.yaml")
        assert csv_result.duckdb_layered_yaml_path == source.with_suffix(".duckdb.layered.yaml")
        assert [item.name for item in csv_result.expressions] == ["amount_band", "risk_flag"]

        layered_result = prepare_duckdb_layered_expression(csv_result.layered_yaml_path)
        assert layered_result.compiled is True
        assert layered_result.source_format == "yaml"
        assert layered_result.layered_yaml_path == csv_result.layered_yaml_path
        assert layered_result.duckdb_layered_yaml_path == csv_result.duckdb_layered_yaml_path
        assert layered_result.duckdb_layered_yaml_path.exists()
        assert [item.name for item in layered_result.expressions] == ["amount_band", "risk_flag"]

        duckdb_result = prepare_duckdb_layered_expression(
            csv_result.duckdb_layered_yaml_path,
            source_format="yaml",
        )
        assert duckdb_result.compiled is False
        assert duckdb_result.source_format == "duckdb_layered_yaml"
        assert duckdb_result.layered_yaml_path is None
        assert duckdb_result.duckdb_layered_yaml_path == csv_result.duckdb_layered_yaml_path
        assert [item.name for item in duckdb_result.expressions] == ["amount_band", "risk_flag"]


def test_prepare_polars_layered_expression_supports_csv_layered_yaml_and_polars_yaml() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = Path(tmp_dir) / "expressions.csv"
        source.write_text(
            "Column Name,Expression\n"
            "amount_band,CASE WHEN [amount] >= 100 THEN 'HIGH' ELSE 'LOW' END\n"
            "risk_flag,CASE WHEN [amount_band] = 'HIGH' THEN 'Y' ELSE 'N' END\n",
            encoding="utf-8",
        )

        csv_result = prepare_polars_layered_expression(
            source,
            result_name_field="Column Name",
            sql_expression_field="Expression",
        )
        assert csv_result.compiled is True
        assert csv_result.source_format == "csv"
        assert csv_result.layered_yaml_path == source.with_suffix(".layered.yaml")
        assert csv_result.polars_layered_yaml_path == source.with_suffix(".polars.layered.yaml")
        assert [item.name for item in csv_result.expressions] == ["amount_band", "risk_flag"]

        layered_result = prepare_polars_layered_expression(csv_result.layered_yaml_path)
        assert layered_result.compiled is True
        assert layered_result.source_format == "yaml"
        assert layered_result.layered_yaml_path == csv_result.layered_yaml_path
        assert layered_result.polars_layered_yaml_path == csv_result.polars_layered_yaml_path
        assert [item.name for item in layered_result.expressions] == ["amount_band", "risk_flag"]

        polars_result = prepare_polars_layered_expression(
            csv_result.polars_layered_yaml_path,
            source_format="yaml",
        )
        assert polars_result.compiled is False
        assert polars_result.source_format == "polars_layered_yaml"
        assert polars_result.layered_yaml_path is None
        assert polars_result.polars_layered_yaml_path == csv_result.polars_layered_yaml_path
        assert [item.name for item in polars_result.expressions] == ["amount_band", "risk_flag"]


def test_required_spotfire_functions_compile_and_execute_like_etl0202_layers() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = Path(tmp_dir) / "required_functions.csv"
        expression_rows = [
            ("total_amount", "Sum([amount]) OVER ([grp])"),
            ("avg_amount", "Avg([amount]) OVER ([grp])"),
            ("min_amount", "Min([amount]) OVER ([grp])"),
            ("max_amount", "Max([amount]) OVER ([grp])"),
            ("abs_total", "Abs([total_amount])"),
            ("name_text", "String([name])"),
            ("name_len", "Len([name])"),
            ("code_part", 'Split([code], "-", 1)'),
            ("name_suffix", "Right([name], 2)"),
            ("if_bucket", "If([max_amount] > 10, 'BIG', 'SMALL')"),
            (
                "case_bucket",
                'CASE WHEN [amount] < 0 THEN \'NEG\' WHEN [amount] > 10 AND [grp] in ("A", "B") THEN \'HIGH\' ELSE NULL END',
            ),
            ("median_amount", "Median([amount]) OVER ([grp])"),
            ("p90_amount", "Percentile([amount], 90) OVER ([grp])"),
            ("concat_key", "[name] & '-' & [code_part]"),
            ("like_flag", '[name] ~= "Al"'),
            ("nested_math", "Abs(Sum([amount]) OVER ([grp])) + If(Max([amount]) OVER ([grp]) > 10, 1, 0)"),
        ]
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Column Name", "Expression"])
            writer.writeheader()
            for column_name, expression in expression_rows:
                writer.writerow({"Column Name": column_name, "Expression": expression})

        result = compile_expression_file(
            source,
            source_format="csv",
            result_name_field="Column Name",
            sql_expression_field="Expression",
        )

        with duckdb.connect(database=":memory:") as connection:
            connection.execute(
                """
                CREATE TABLE layer_000 AS
                SELECT *
                FROM (
                    VALUES
                        ('A', 10.0, 'Alice', 'AA-001'),
                        ('A', 20.0, 'Bob', 'BB-002'),
                        ('B', -5.0, 'Alicia', 'CC-003')
                ) AS t(grp, amount, name, code)
                """
            )
            _execute_duckdb_layered_yaml(connection, result.duckdb_layered_yaml_path)
            rows = connection.execute(
                """
                SELECT
                    name,
                    total_amount,
                    avg_amount,
                    min_amount,
                    max_amount,
                    abs_total,
                    name_text,
                    name_len,
                    code_part,
                    name_suffix,
                    if_bucket,
                    case_bucket,
                    median_amount,
                    p90_amount,
                    concat_key,
                    like_flag,
                    nested_math
                FROM layer_final
                ORDER BY name
                """
            ).fetchall()

    assert rows == [
        ("Alice", 30.0, 15.0, 10.0, 20.0, 30.0, "Alice", 5, "AA", "ce", "BIG", None, 15.0, 19.0, "Alice-AA", True, 31.0),
        ("Alicia", -5.0, -5.0, -5.0, -5.0, 5.0, "Alicia", 6, "CC", "ia", "SMALL", "NEG", -5.0, -5.0, "Alicia-CC", True, 5.0),
        ("Bob", 30.0, 15.0, 10.0, 20.0, 30.0, "Bob", 3, "BB", "ob", "BIG", "HIGH", 15.0, 19.0, "Bob-BB", False, 31.0),
    ]


def test_required_spotfire_functions_compile_and_execute_as_polars_layers() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = Path(tmp_dir) / "required_functions.csv"
        expression_rows = [
            ("total_amount", "Sum([amount]) OVER ([grp])"),
            ("avg_amount", "Avg([amount]) OVER ([grp])"),
            ("min_amount", "Min([amount]) OVER ([grp])"),
            ("max_amount", "Max([amount]) OVER ([grp])"),
            ("abs_total", "Abs([total_amount])"),
            ("name_text", "String([name])"),
            ("name_len", "Len([name])"),
            ("code_part", 'Split([code], "-", 1)'),
            ("name_suffix", "Right([name], 2)"),
            ("if_bucket", "If([max_amount] > 10, 'BIG', 'SMALL')"),
            (
                "case_bucket",
                'CASE WHEN [amount] < 0 THEN \'NEG\' WHEN [amount] > 10 AND [grp] in ("A", "B") THEN \'HIGH\' ELSE NULL END',
            ),
            ("median_amount", "Median([amount]) OVER ([grp])"),
            ("p90_amount", "Percentile([amount], 90) OVER ([grp])"),
            ("concat_key", "[name] & '-' & [code_part]"),
            ("like_flag", '[name] ~= "Al"'),
            ("logic_flag", '([amount] < 0 OR NOT ([grp] in ("A", "B")))'),
            ("arithmetic_score", "([amount] + 2 - 1) * 3 / 2"),
            ("nested_math", "Abs(Sum([amount]) OVER ([grp])) + If(Max([amount]) OVER ([grp]) > 10, 1, 0)"),
        ]
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Column Name", "Expression"])
            writer.writeheader()
            for column_name, expression in expression_rows:
                writer.writerow({"Column Name": column_name, "Expression": expression})

        result = compile_polars_expression_file(
            source,
            source_format="csv",
            result_name_field="Column Name",
            sql_expression_field="Expression",
        )
        loaded = load_polars_layered_expression_yaml(result.polars_layered_yaml_path)
        assert {item.name for item in loaded} == {name for name, _expr in expression_rows}

        frame = pl.DataFrame(
            {
                "grp": ["A", "A", "B"],
                "amount": [10.0, 20.0, -5.0],
                "name": ["Alice", "Bob", "Alicia"],
                "code": ["AA-001", "BB-002", "CC-003"],
            }
        )
        output = _execute_polars_layered_yaml(frame, result.polars_layered_yaml_path).sort("name")

    rows = output.select(
        [
            "name",
            "total_amount",
            "avg_amount",
            "min_amount",
            "max_amount",
            "abs_total",
            "name_text",
            "name_len",
            "code_part",
            "name_suffix",
            "if_bucket",
            "case_bucket",
            "median_amount",
            "p90_amount",
            "concat_key",
            "like_flag",
            "logic_flag",
            "arithmetic_score",
            "nested_math",
        ]
    ).rows()
    assert rows == [
        ("Alice", 30.0, 15.0, 10.0, 20.0, 30.0, "Alice", 5, "AA", "ce", "BIG", None, 15.0, 19.0, "Alice-AA", True, False, 16.5, 31.0),
        ("Alicia", -5.0, -5.0, -5.0, -5.0, 5.0, "Alicia", 6, "CC", "ia", "SMALL", "NEG", -5.0, -5.0, "Alicia-CC", True, True, -6.0, 5.0),
        ("Bob", 30.0, 15.0, 10.0, 20.0, 30.0, "Bob", 3, "BB", "ob", "BIG", "HIGH", 15.0, 19.0, "Bob-BB", False, False, 31.5, 31.0),
    ]


def test_additional_spotfire_functions_execute_as_polars_layers() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = Path(tmp_dir) / "additional_functions.csv"
        expression_rows = [
            ("amount_int", "Cast([amount] as Integer)"),
            ("amount_money", "Currency([amount])"),
            ("value_default", "SN([maybe], 0)"),
            ("missing_flag", "IsNull([maybe])"),
            ("parsed_date", "ParseDate([raw_date], '%Y-%m-%d')"),
            ("date_year", "DatePart('year', [parsed_date])"),
            ("name_mid", "Mid([name], 2, 3)"),
            ("name_sub", "Substitute([name], 'A', 'B')"),
            ("find_i", "Find('i', [name])"),
            ("rx_prefix", "RxExtract([code], '([A-Z]+)-')"),
            ("q1_amount", "Q1([amount]) OVER ([grp])"),
            ("iqr_amount", "IQR([amount]) OVER ([grp])"),
            ("weighted_amount", "WeightedAverage([amount], [weight]) OVER ([grp])"),
            ("max_name", "ValueForMax([name], [amount]) OVER ([grp])"),
            ("cov_amount_weight", "Covariance([amount], [weight]) OVER ([grp])"),
            ("nth_largest", "NthLargest([amount], 1) OVER ([grp])"),
            ("dense_rank", 'DenseRank([amount], "desc", [grp])'),
            ("rank_real", 'RankReal([amount], "desc", "ties.method=average", [grp])'),
            ("names_in_grp", "UniqueConcatenate([name]) OVER ([grp])"),
        ]
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Column Name", "Expression"])
            writer.writeheader()
            for column_name, expression in expression_rows:
                writer.writerow({"Column Name": column_name, "Expression": expression})

        result = compile_polars_expression_file(
            source,
            source_format="csv",
            result_name_field="Column Name",
            sql_expression_field="Expression",
        )
        frame = pl.DataFrame(
            {
                "grp": ["A", "A", "B"],
                "amount": [10.0, 20.0, -5.0],
                "weight": [1.0, 3.0, 2.0],
                "maybe": [None, 2.0, None],
                "raw_date": ["2024-01-02", "2024-02-03", "2025-01-01"],
                "name": ["Alice", "Bob", "Alicia"],
                "code": ["AA-001", "BB-002", "CC-003"],
            }
        )
        output = _execute_polars_layered_yaml(frame, result.polars_layered_yaml_path).sort("name")

    rows = output.select(
        [
            "name",
            "amount_int",
            pl.col("amount_money").cast(pl.String).alias("amount_money_text"),
            "value_default",
            "missing_flag",
            "date_year",
            "name_mid",
            "name_sub",
            "find_i",
            "rx_prefix",
            "q1_amount",
            "iqr_amount",
            "weighted_amount",
            "max_name",
            "cov_amount_weight",
            "nth_largest",
            "dense_rank",
            "rank_real",
            "names_in_grp",
        ]
    ).rows()
    assert rows == [
        ("Alice", 10, "10.0000000000", 0.0, True, 2024, "lic", "Blice", 3, "AA", 12.5, 5.0, 17.5, "Bob", 10.0, 20.0, 2, 2.0, "Alice,Bob"),
        ("Alicia", -5, "-5.0000000000", 0.0, True, 2025, "lic", "Blicia", 3, "CC", -5.0, 0.0, -5.0, "Alicia", 0.0, -5.0, 1, 1.0, "Alicia"),
        ("Bob", 20, "20.0000000000", 2.0, False, 2024, "ob", "Bob", 0, "BB", 12.5, 5.0, 17.5, "Bob", 10.0, 20.0, 1, 1.0, "Alice,Bob"),
    ]


def test_priority_review_items_execute_as_polars_layers() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = Path(tmp_dir) / "priority_review_functions.csv"
        expression_rows = [
            ("add_days", "DateAdd('day', [n], [dt])"),
            ("diff_days", "DateDiff('day', [dt], [dt2])"),
            ("dur_hours", "Hours([n])"),
            ("total_hours", "TotalHours([dur_hours])"),
            ("first_after", "FirstValidAfter([maybe_text])"),
            ("last_before", "LastValidBefore([maybe_text])"),
            ("mean_deviation", "MeanDeviation([amount]) OVER ([grp])"),
            ("mad_amount", "MedianAbsoluteDeviation([amount]) OVER ([grp])"),
            ("trimmed_mean", "TrimmedMean([amount], 10) OVER ([grp])"),
            ("lav_amount", "LAV([amount]) OVER ([grp])"),
            ("uav_amount", "UAV([amount]) OVER ([grp])"),
            ("outlier_count", "Outliers([amount]) OVER ([grp])"),
            ("outlier_pct", "PctOutliers([amount]) OVER ([grp])"),
            ("rand_value", "Rand()"),
            ("rand_bucket", "RandBetween(1, 3)"),
        ]
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Column Name", "Expression"])
            writer.writeheader()
            for column_name, expression in expression_rows:
                writer.writerow({"Column Name": column_name, "Expression": expression})

        result = compile_polars_expression_file(
            source,
            source_format="csv",
            result_name_field="Column Name",
            sql_expression_field="Expression",
        )
        frame = pl.DataFrame(
            {
                "grp": ["A", "A", "A", "B"],
                "amount": [1.0, 2.0, 100.0, 5.0],
                "n": [1, 2, 3, 4],
                "dt": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"],
                "dt2": ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"],
                "maybe_text": [None, "a", None, "b"],
            }
        ).with_columns(
            pl.col("dt").str.strptime(pl.Date, format="%Y-%m-%d"),
            pl.col("dt2").str.strptime(pl.Date, format="%Y-%m-%d"),
        )
        output = _execute_polars_layered_yaml(frame, result.polars_layered_yaml_path)

    rows = output.select(
        [
            "add_days",
            "diff_days",
            "total_hours",
            "first_after",
            "last_before",
            "mean_deviation",
            "mad_amount",
            "trimmed_mean",
            "lav_amount",
            "uav_amount",
            "outlier_count",
            "outlier_pct",
        ]
    ).rows()
    assert rows == [
        (date(2024, 1, 2), 2, 1, "a", None, 43.77777777777777, 1.0, 2.0, 1.0, 100.0, 0, 0.0),
        (date(2024, 1, 3), 3, 2, "a", "a", 43.77777777777777, 1.0, 2.0, 1.0, 100.0, 0, 0.0),
        (date(2024, 1, 4), 4, 3, "b", "a", 43.77777777777777, 1.0, 2.0, 1.0, 100.0, 0, 0.0),
        (date(2024, 1, 5), 5, 4, "b", "b", 0.0, 0.0, 5.0, 5.0, 5.0, 0, 0.0),
    ]
    rand_rows = output.select(["rand_value", "rand_bucket"]).rows()
    assert all(0.0 <= rand_value < 1.0 for rand_value, _bucket in rand_rows)
    assert all(1 <= rand_bucket <= 3 for _rand_value, rand_bucket in rand_rows)


def test_polars_dateadd_calendar_units_are_not_implemented_yet() -> None:
    with pytest.raises(ValueError, match="day, hour, minute, second, and millisecond"):
        normalize_expression_for_polars("DateAdd('month', 1, [dt])")


def _execute_duckdb_layered_yaml(connection: duckdb.DuckDBPyConnection, path: Path) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    previous_table = "layer_000"
    for layer_index, layer in enumerate(payload["layers"], start=1):
        select_items = ["*"]
        for item in layer["expressions"]:
            column_name = str(item["column_name"]).replace('"', '""')
            select_items.append(f"{item['duckdb_sql']} AS \"{column_name}\"")
        current_table = f"layer_{layer_index:03d}"
        connection.execute(
            f"""
            CREATE TABLE {current_table} AS
            SELECT {", ".join(select_items)}
            FROM {previous_table}
            """
        )
        previous_table = current_table
    connection.execute(f"CREATE VIEW layer_final AS SELECT * FROM {previous_table}")


def _execute_polars_layered_yaml(frame: pl.DataFrame, path: Path) -> pl.DataFrame:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    result = frame
    namespace = {"pl": pl}
    for layer in payload["layers"]:
        expressions = [
            eval(str(item["polars_expr"]), namespace).alias(str(item["column_name"]))
            for item in layer["expressions"]
        ]
        result = result.with_columns(expressions)
    return result


def test_validate_expression_compatibility_writes_unsupported_yaml() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        output = Path(tmp_dir) / "unsupported.yaml"
        expressions = [
            DerivedExpression("safe_math", "Abs([amount])", normalize_expression("Abs([amount])"), []),
            DerivedExpression("bad_func", "AutoBinNumeric([amount])", "AutoBinNumeric(\"amount\")", []),
        ]

        try:
            validate_expression_compatibility(expressions, unsupported_yaml_path=output)
        except ExpressionCompatibilityError as exc:
            assert exc.unsupported_yaml_path == output
        else:
            raise AssertionError("expected ExpressionCompatibilityError")

        text = output.read_text(encoding="utf-8")
        assert "bad_func" in text
        assert "AutoBinNumeric" in text
        assert "safe_math" not in text


def test_packaged_catalog_resources_are_available() -> None:
    db_path = catalog_db_path()
    md_path = unsupported_functions_markdown_path()
    assert db_path.exists()
    assert md_path.exists()
    with sqlite3.connect(db_path) as connection:
        count = connection.execute("SELECT count(*) FROM spotfire_functions").fetchone()[0]
    assert count > 100
