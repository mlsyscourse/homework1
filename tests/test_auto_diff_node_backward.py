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
        print(repr(output_val))
        np.testing.assert_allclose(actual=output_val, desired=expected_val)


def test_mul():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.mul(x1, x2)
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: np.array([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: np.array([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
            y_grad: np.array([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        },
        expected_outputs=[
            np.array([[-1.12, 0.35, 0.5, 0.0], [-0.0225, 0.0, 7.424, -9.61]]),
            np.array([[0.4, 1.0, -2.5, 115.6], [-0.01125, 0.0, -13.456, -9.61]]),
        ],
    )


def test_div():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.div(x1, x2)
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: np.array([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: np.array([[2.5, 4.0, -0.1, 0.1], [-8.0, 5.0, -2.5, -1.0]]),
            y_grad: np.ones((2, 4), "float32"),
        },
        expected_outputs=[
            np.array([[0.4, 0.25, -10.0, 10.0], [-0.125, 0.2, -0.4, -1.0]]),
            np.array([[0.16, -0.125, -50.0, -340.0], [-0.0046875, 0, 0.928, -3.1]]),
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
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

    x1_val = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x2_val = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    y_grad_val = np.ones((3, 3), "float32")
    x1_grad_expected = np.array([[24.0, 33.0], [24.0, 33.0], [24.0, 33.0]])
    x2_grad_expected = np.array([[9.0, 9.0, 9.0], [12.0, 12.0, 12.0]])

    if trans_A:
        x1_val = x1_val.T
        x1_grad_expected = x1_grad_expected.T
    if trans_B:
        x2_val = x2_val.T
        x2_grad_expected = x2_grad_expected.T

    check_evaluator_output(
        evaluator,
        input_values={
            x1: x1_val,
            x2: x2_val,
            y_grad: y_grad_val,
        },
        expected_outputs=[x1_grad_expected, x2_grad_expected],
    )


if __name__ == "__main__":
    test_mul()
    test_div()

    for trans_A in [False, True]:
        for trans_B in [False, True]:
            test_matmul(trans_A, trans_B)
