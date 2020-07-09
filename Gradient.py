import numpy as np

def _numerical_gradient_ndim_one(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        temp_x = x[i]
        x[i] = temp_x + h
        func_plus_x = f(x)

        x[i] = temp_x  - h
        func_minus_x = f(x)

        grad[i] = (func_plus_x - func_minus_x) / (2 * h)        # 미분
        x[i] = temp_x

    return grad

def numerical_gradient(f, x):
    if x.ndim == 1:
        return _numerical_gradient_ndim_one(f, x)
    else:
        grade = np.zeros_like(x)

        for i, data in enumerate(x):
            grade[i] = _numerical_gradient_ndim_one(f, data)

        return grade

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    grade = None
    for i in range(step_num):
        x_history.append(x.copy())

        grade = numerical_gradient(f, x)
        x -= lr * grade
        print(x)

    return x, np.array(x_history)