"""Demo runner for the Matrix calculator.

Focuses on matrix multiplication first, then shows the other operations.
"""

from matrix import Matrix, MatrixError


def demo_multiplication():
    print("=== Matrix Multiplication ===")
    a = Matrix([
        [1, 2, 3],
        [4, 5, 6],
    ])
    b = Matrix([
        [7, 8],
        [9, 10],
        [11, 12],
    ])

    print("A =")
    print(a)
    print("\nB =")
    print(b)

    product = a.multiply(b)
    print("\nA * B =")
    print(product)


def demo_other_operations():
    print("\n=== Other Operations ===")
    a = Matrix([
        [1, 2],
        [3, 4],
    ])
    b = Matrix([
        [5, 6],
        [7, 8],
    ])

    print("A + B =")
    print(a.add(b))

    print("\nA - B =")
    print(a.subtract(b))

    print("\n3 * A =")
    print(a.scalar_multiply(3))

    print("\nA^T (transpose) =")
    print(a.transpose())


def demo_error_handling():
    print("\n=== Error Handling ===")
    a = Matrix([[1, 2, 3]])
    b = Matrix([[1, 2, 3]])
    try:
        a.multiply(b)
    except MatrixError as exc:
        print("Caught expected error:", exc)


def main():
    demo_multiplication()
    demo_other_operations()
    demo_error_handling()


if __name__ == "__main__":
    main()
