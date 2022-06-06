from numpy import dot, sqrt, abs
import sys


def cosine_distance(a, b):
    """Determine cosine of angle between two iterables of the same size

    Argument(s):
        a (iterable): 1st input iterable
        b (iterable): 2nd input iterable

    Return(s):
        (float): magnitude of distance between two iterables in [0, 1]
    """
    if (len(a) != len(b)):
        sys.exit(
            "Cannot determine distance as iterables not the same size"
        )

    return abs(dot(a, b) / (sqrt(dot(a, a)) * sqrt(dot(b, b))))


def print_cosine_distance(a, b):
    """Print cosine distance between two iterables

    Argument(s):
        a (iterable): 1st input iterable
        b (iterable): 2nd input iterable
    """
    print(
        "Cosine of angle between {} and {}: {}".format(
            a, b, cosine_distance(a, b)
        )
    )


def test_distance_function():
    # Test 1: Identical vectors
    x = [1, 2, 3]
    y = [1, 2, 3]

    print_cosine_distance(x, y)

    # Test 2: Identical vectors with different scalings
    x = [1, 2, 3]
    y = [3, 6, 9]

    print_cosine_distance(x, y)

    # Test 3: Tuple separated by 180 degrees
    x = (1, 2, 3)
    y = (-1, -2, -3)

    print_cosine_distance(x, y)

    # Test 4: Tuple separated by 90 degrees
    x = (0, 1, 0)
    y = (0, 0, 1)

    print_cosine_distance(x, y)

    # Test 5: Tuple of different sizes
    x = (0, 1, 0, 1)
    y = (0, 0, 1)

    print_cosine_distance(x, y)
