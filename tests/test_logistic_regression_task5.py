from logistic_regression import logistic_regression
import numpy as np
import auto_diff as ad



def check_evaluator_output(evaluator, input_values, expected_outputs):
    output_values = evaluator.run(input_values)
    assert len(output_values) == len(expected_outputs)
    for output_val, expected_val in zip(output_values, expected_outputs):
        np.testing.assert_allclose(actual=output_val, desired=expected_val)



def test_logistic_regression_simple_float():
    X = ad.Variable(name="X")
    W = ad.Variable(name="W")
    b = ad.Variable(name="b")
    model_output = logistic_regression(X, W, b)
    evaluator = ad.Evaluator([model_output])
    
    X_val = np.array([[1.5, 2.5], [3.5, 4.5]])
    W_val = np.array([[5.5, 6.5], [7.5, 8.5]])
    b_val = np.array([1.0, 2.0])
    expected_output = np.array([[28.0, 33.0], [54.0, 63.0]])
    
    check_evaluator_output(evaluator, {X: X_val, W: W_val, b: b_val}, [expected_output])



def test_logistic_regression_broadcasting_float():
    X = ad.Variable(name="X")
    W = ad.Variable(name="W")
    b = ad.Variable(name="b")
    model_output = logistic_regression(X, W, b)
    evaluator = ad.Evaluator([model_output])

    X_val = np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])
    W_val = np.array([[1.7, 2.8], [3.9, 4.0]])
    b_val = np.array([1.5, 2.5])
    expected_output = np.array([[11.95, 14.38], [24.27, 29.34], [36.59, 44.3]])
    
    check_evaluator_output(evaluator, {X: X_val, W: W_val, b: b_val}, [expected_output])



def test_logistic_regression_zero_bias_float():
    X = ad.Variable(name="X")
    W = ad.Variable(name="W")
    b = ad.Variable(name="b")
    model_output = logistic_regression(X, W, b)
    evaluator = ad.Evaluator([model_output])
    
    X_val = np.array([[0.5, 1.5], [2.5, 3.5]])
    W_val = np.array([[4.5, 5.5], [6.5, 7.5]])
    b_val = np.array([0.0, 0.0])
    expected_output = np.array([[12.0, 14.0], [34.0, 40.0]])
    
    check_evaluator_output(evaluator, {X: X_val, W: W_val, b: b_val}, [expected_output])



if __name__ == "__main__":
    test_logistic_regression_simple_float()
    test_logistic_regression_broadcasting_float()
    test_logistic_regression_zero_bias_float()