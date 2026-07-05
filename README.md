# spotfire_expr_normalizer

Spotfire expression을 DuckDB SQL로 정규화하는 작은 Python 패키지입니다.

주요 기능:

- CSV/layered YAML/DuckDB layered YAML/Polars layered YAML 입력을 실행용 계약으로 준비
- CSV/YAML expression 묶음을 dependency 기반 layered YAML로 컴파일
- 실행용 DuckDB-normalized layered YAML 생성
- 실행용 Polars layered YAML 생성
- Spotfire `[Column]` 참조를 DuckDB quoted identifier로 변환
- Spotfire `OVER ([col])` 구문을 DuckDB `OVER (PARTITION BY "col")`로 변환
- `~=` 부분일치 연산자를 `LIKE '%value%'`로 변환
- `&` 문자열 연결 연산자를 DuckDB `||`로 변환
- `Split([x], sep, index)`를 DuckDB `string_split`/`list_extract`로 변환
- `Percentile([x], 90)`을 `quantile_cont("x", 0.9)`로 변환
- `Sum`, `Avg`, `Count`, `Max`, `Min` 같은 SQL 호환 집계 함수는 과도하게 풀어쓰지 않고 기본 함수 형태 유지
- expression dependency layer 구성과 중복 expression canonicalize
- Spotfire 함수 카탈로그 SQLite DB와 미지원 함수 문서 포함

기본 사용:

```python
from spotfire_expr_normalizer import (
    normalize_expression,
    normalize_expression_for_polars,
    prepare_duckdb_layered_expression,
    prepare_polars_layered_expression,
)

sql = normalize_expression('CASE WHEN [country] ~= "KOR" THEN "Y" ELSE "N" END')
polars_expr = normalize_expression_for_polars('CASE WHEN [country] ~= "KOR" THEN "Y" ELSE "N" END')

duckdb_prepared = prepare_duckdb_layered_expression(
    "expressions.csv",
    result_name_field="Column Name",
    sql_expression_field="Expression",
)
polars_prepared = prepare_polars_layered_expression(
    "expressions.csv",
    result_name_field="Column Name",
    sql_expression_field="Expression",
)
print(duckdb_prepared.duckdb_layered_yaml_path)
print(polars_prepared.polars_layered_yaml_path)
```

산출물 계약:

- `*.layered.yaml`: 사람이 검토하기 좋은 Spotfire 원문 기반 layered YAML
- `*.duckdb.layered.yaml`: ETL0202 같은 실행기가 읽는 DuckDB SQL layered YAML
- `*.polars.layered.yaml`: Polars backend 실행기가 읽는 Polars expression layered YAML
- `*.unsupported.yaml`: 미지원 Spotfire 함수가 있을 때 생성되는 진단 YAML

입력 흐름:

- CSV: `spotfire_expr_normalizer`가 `*.layered.yaml`과 `*.duckdb.layered.yaml`을 생성
- `*.layered.yaml`: `spotfire_expr_normalizer`가 `*.duckdb.layered.yaml`을 생성
- `*.duckdb.layered.yaml`: 재컴파일하지 않고 검증/로드만 수행
- CSV 또는 `*.layered.yaml`: Polars backend를 선택하면 `*.polars.layered.yaml`을 생성
- `*.polars.layered.yaml`: 재컴파일하지 않고 검증/로드만 수행

검증:

- `tests/test_normalizer.py`에는 필수 구현 함수 목록을 모두 포함한 더미 Spotfire expression CSV를 컴파일하고, 생성된 DuckDB layered YAML을 더미 데이터에 layer별로 실행하는 통합 테스트가 포함되어 있습니다.
