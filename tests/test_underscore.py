# We don't support this yet and are unsure if we really want it.

# import pytest
# from hulearn.underscore import _
#
# params_arithmetic = [
#     (_ + 1, [2, 3, 4, 5]),
#     (_ * 2, [2, 4, 6, 8]),
#     (_ - 1, [0, 1, 2, 3]),
#     (_ % 2, [1, 0, 1, 0]),
# ]
#
#
# @pytest.mark.parametrize("f,exp", params_arithmetic)
# def test_base_arithmetic(f, exp):
#     numbers = [1, 2, 3, 4]
#     assert list(map(f, numbers)) == exp
#
#
# boolean_arithmetic = [
#     (_ == 1, [True, False, False, False]),
#     (_ != 2, [True, False, True, True]),
#     (_ > 3,  [False, False, False, True]),
#     (_ >= 3, [False, False, True, True]),
#     (_ <= 3, [True, True, True, False]),
#     (_ < 3, [True, True, False, False]),
# ]
#
#
# @pytest.mark.parametrize("f,exp", boolean_arithmetic)
# def test_boolean_ops(f, exp):
#     numbers = [1, 2, 3, 4]
#     assert list(map(f, numbers)) == exp
#
