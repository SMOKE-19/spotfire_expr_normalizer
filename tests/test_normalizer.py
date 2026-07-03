from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile

from spotfire_expr_normalizer import (
    DerivedExpression,
    ExpressionCompatibilityError,
    build_expression_layers,
    canonicalize_expressions,
    catalog_db_path,
    compile_expression_file,
    load_duckdb_layered_expression_yaml,
    normalize_expression,
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
    assert normalize_expression("Mid([name], 2, 3)") == 'substr("name", 2, 3)'
    assert normalize_expression("Substitute([name], 'A', 'B')") == 'replace("name", \'A\', \'B\')'
    assert normalize_expression("FromEpochSeconds([ts])") == 'to_timestamp("ts")'
    assert normalize_expression("FromEpochMilliseconds([ts])") == 'to_timestamp(("ts")::DOUBLE / 1000.0)'
    assert normalize_expression("ToEpochSeconds([dt])") == 'epoch("dt")'
    assert normalize_expression("ToEpochMilliseconds([dt])") == 'epoch_ms("dt")'
    assert normalize_expression("ISOWeek([dt])") == 'week("dt")'
    assert normalize_expression("ISOYear([dt])") == 'isoyear("dt")'
    assert normalize_expression("YearAndWeek([dt])") == 'strftime("dt", \'%G-%V\')'


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
