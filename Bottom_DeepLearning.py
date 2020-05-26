import numpy as np

def AND(x1, x2):
    InputData = np.array([x1, x2])
    Weight = np.array([0.5, 0.5])
    Bias = -0.7

    Result = np.sum(InputData * Weight) + Bias
    if Result > 0:
        return 1
    else:
        return 0

def NAND(x1, x2):
    InputData = np.array([x1, x2])
    Weight = np.array([-0.5, -0.5])
    Bias = +0.7

    Result = np.sum(InputData * Weight) + Bias
    if Result > 0:
        return 1
    else:
        return 0

def OR(x1, x2):
    InputData = np.array([x1, x2])
    Weight = np.array([0.5, 0.5])
    Bias = -(0.7 / 2.0)

    Result = np.sum(InputData * Weight) + Bias
    if Result > 0:
        return 1
    else:
        return 0

def XOR(x1, x2):
    Result_OR = OR(x1, x2)
    Result_NAND = NAND(x1, x2)
    Result = AND(Result_OR, Result_NAND)

    return Result

print("AND----------------------")
print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))

print("NAND----------------------")
print(NAND(0, 0))
print(NAND(1, 0))
print(NAND(0, 1))
print(NAND(1, 1))

print("OR----------------------")
print(OR(0, 0))
print(OR(1, 0))
print(OR(0, 1))
print(OR(1, 1))

print("XOR----------------------")
print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))