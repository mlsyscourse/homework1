from typing import Dict, List

import numpy as np
import pytest

import auto_diff as ad


def check_evaluator_output(
    evaluator: ad.Evaluator,
    input_values: Dict[ad.Node, np.ndarray],
    expected_outputs: List[np.ndarray],
) -> None:
    output_values = evaluator.run(input_values)
    assert len(output_values) == len(expected_outputs)
    for output_val, expected_val in zip(output_values, expected_outputs):
        np.testing.assert_allclose(actual=output_val, desired=expected_val)


def test_identity():
    x = ad.Variable("x1")
    evaluator = ad.Evaluator(eval_nodes=[x])

    x_val = np.random.rand(4, 5).astype("float32")
    check_evaluator_output(evaluator, input_values={x: x_val}, expected_outputs=[x_val])


def test_add():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.add(x1, x2)
    evaluator = ad.Evaluator(eval_nodes=[y])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: np.array([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: np.array([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
        },
        expected_outputs=[np.array([[1.8, 2.7, 0.4, 3.4], [0.9, 6.6, -2.6, 6.2]])],
    )


def test_add_by_const():
    x1 = ad.Variable("x1")
    y = ad.add_by_const(x1, 2.7)
    evaluator = ad.Evaluator(eval_nodes=[y])

    check_evaluator_output(
        evaluator,
        input_values={x1: np.array([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])},
        expected_outputs=[np.array([[1.7, 4.7, 3.2, 6.1], [3.0, 2.7, -3.1, 5.8]])],
    )


def test_mul():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.mul(x1, x2)
    evaluator = ad.Evaluator(eval_nodes=[y])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: np.array([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: np.array([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
        },
        expected_outputs=[
            np.array([[-2.80, 1.40, -0.05, 0.00], [0.18, 0.00, -18.56, 9.61]])
        ],
    )


def test_mul_by_const():
    x1 = ad.Variable("x1")
    y = ad.mul_by_const(x1, 2.7)
    evaluator = ad.Evaluator(eval_nodes=[y])

    check_evaluator_output(
        evaluator,
        input_values={x1: np.array([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])},
        expected_outputs=[
            np.array([[-2.70, 5.40, 1.35, 9.18], [0.81, 0.00, -15.66, 8.37]])
        ],
    )


def test_div():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.div(x1, x2)
    evaluator = ad.Evaluator(eval_nodes=[y])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: np.array([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: np.array([[2.5, 4.0, -0.1, 0.1], [-8.0, 5.0, -2.5, -1.0]]),
        },
        expected_outputs=[
            np.array([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]])
        ],
    )


def test_div_by_const():
    x1 = ad.Variable("x1")
    y = ad.div_by_const(x1, 5.0)
    evaluator = ad.Evaluator(eval_nodes=[y])

    check_evaluator_output(
        evaluator,
        input_values={x1: np.array([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])},
        expected_outputs=[np.array([[-0.2, 0.4, 0.1, 0.68], [0.06, 0.0, -1.16, 0.62]])],
    )


@pytest.mark.parametrize("trans_A", [False, True])
@pytest.mark.parametrize("trans_B", [False, True])
def test_matmul(trans_A, trans_B):
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.matmul(x1, x2, trans_A=trans_A, trans_B=trans_B)
    evaluator = ad.Evaluator(eval_nodes=[y])

    x1_val = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x2_val = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    if trans_A:
        x1_val = x1_val.T
    if trans_B:
        x2_val = x2_val.T

    check_evaluator_output(
        evaluator,
        input_values={
            x1: x1_val,
            x2: x2_val,
        },
        expected_outputs=[
            np.array([[27.0, 30.0, 33.0], [61.0, 68.0, 75.0], [95.0, 106.0, 117.0]])
        ],
    )


def test_graph():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    x3 = ad.Variable("x3")
    y = ad.matmul(x1, x2, trans_B=True) / 10 + x3
    evaluator = ad.Evaluator(eval_nodes=[y])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: np.array([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: np.array([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
            x3: np.array([[2.71, 3.14], [3.87, -4.0]]),
        },
        expected_outputs=[np.array([[2.565, 5.614], [4.012, -4.877]])],
    )


if __name__ == "__main__":
    test_identity()
    test_add()
    test_add_by_const()
    test_mul()
    test_mul_by_const()
    test_div()
    test_div_by_const()

    for trans_A in [False, True]:
        for trans_B in [False, True]:
            test_matmul(trans_A, trans_B)

    test_graph()
