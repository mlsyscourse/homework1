from typing import Dict, List

import numpy as np

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


def test_graph():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    x3 = ad.Variable("x3")
    y = ad.matmul(x1, x2, trans_B=True) / 10 * x3
    x1_grad, x2_grad, x3_grad = ad.gradients(y, nodes=[x1, x2, x3])
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad, x3_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: np.array([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: np.array([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
            x3: np.array([[2.71, 3.14], [3.87, -4.0]]),
        },
        expected_outputs=[
            np.array(
                [[0.9472, 2.2621, 0.9777, 0.9734], [0.8436, -2.3691, -1.3187, -1.24]]
            ),
            np.array(
                [[-0.1549, 0.542, -2.1091, 2.1211], [-0.434, 0.628, 2.477, -0.1724]]
            ),
            np.array([[-0.145, 2.474], [0.142, -0.877]]),
        ],
    )


def test_gradient_of_gradient():
    x1 = ad.Variable(name="x1")
    x2 = ad.Variable(name="x2")
    y = x1 * x1 + x1 * x2

    grad_x1, grad_x2 = ad.gradients(y, [x1, x2])
    grad_x1_x1, grad_x1_x2 = ad.gradients(grad_x1, [x1, x2])
    grad_x2_x1, grad_x2_x2 = ad.gradients(grad_x2, [x1, x2])

    evaluator = ad.Evaluator(
        [y, grad_x1, grad_x2, grad_x1_x1, grad_x1_x2, grad_x2_x1, grad_x2_x2]
    )
    check_evaluator_output(
        evaluator,
        input_values={
            x1: np.array([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: np.array([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
        },
        expected_outputs=[
            np.array([[-1.8, 5.4, 0.2, 11.56], [0.27, 0.0, 15.08, 19.22]]),
            np.array([[0.8, 4.7, 0.9, 6.8], [1.2, 6.6, -8.4, 9.3]]),
            np.array([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            2 * np.ones((2, 4), "float32"),
            1 * np.ones((2, 4), "float32"),
            1 * np.ones((2, 4), "float32"),
            np.zeros((2, 4), "float32"),
        ],
    )


if __name__ == "__main__":
    test_graph()
    test_gradient_of_gradient()
