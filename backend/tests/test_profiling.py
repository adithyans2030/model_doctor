import numpy as np
import pandas as pd
import pytest

from app.services.profiling import infer_type, looks_like_id_name


@pytest.mark.parametrize(
    "name,expected",
    [
        ("id", True),
        ("customer_id", True),
        ("customerId", True),
        ("Unnamed: 0", True),
        ("order-key", True),
        ("paid", False),
        ("valid", False),
        ("age", False),
    ],
)
def test_id_names(name, expected):
    assert looks_like_id_name(name) is expected


def _s(values, name="col"):
    return pd.Series(values, name=name)


def test_infer_types():
    n = 100
    assert infer_type(_s([None] * n))[0] == "empty"
    assert infer_type(_s(np.arange(n), "row"))[0] == "id"  # monotonic unique ints
    assert infer_type(_s(np.random.default_rng(0).permutation(n), "user_id"))[0] == "id"
    assert infer_type(_s(np.random.default_rng(0).permutation(n), "score"))[0] == "numeric"
    assert infer_type(_s([0, 1] * 50))[0] == "boolean"
    assert infer_type(_s([True, False] * 50))[0] == "boolean"
    assert infer_type(_s(["Y", "N"] * 50))[0] == "boolean"
    assert infer_type(_s(["red", "green", "blue", "red"] * 25))[0] == "categorical"
    assert infer_type(_s(["2024-01-%02d" % (i % 28 + 1) for i in range(n)]))[0] == "datetime"
    assert infer_type(_s(["1.5", "2", "3.25", "x"] + ["4"] * 96)) == ("numeric", True)
    assert infer_type(_s([1.5, 2.25, 3.0] * 30 + [np.nan] * 10))[0] == "numeric"
