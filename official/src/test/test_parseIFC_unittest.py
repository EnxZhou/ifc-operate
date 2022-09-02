import unittest

from src.parseIFC import parse_index_list


class TestParseIFc(unittest.TestCase):
    def test_parse_ifc_index_list(self):
        point_list = [-0.0, 0.0, -0.001, -0.0, 0.0, 0.0, 0.06, 0.0, 0.0, 0.06, 0.0, -0.001]
        index_list = [0, 1, 2]
        res = parse_index_list(point_list, index_list)
        self.assertEqual(res, [-0.0, 0.0, -0.001])
        point_list = [-0.0, 0.0, -0.001, -0.0, 0.0, 0.0, 0.06, 0.0, 0.0, 0.06, 0.0, -0.001]
        index_list = [5, 6, 1]
        res = parse_index_list(point_list, index_list)
        self.assertEqual(res, [0.0, 0.06, -0.0])
