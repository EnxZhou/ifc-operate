import unittest

from parseStepToCsv import parse_name_and_guid


class TestSplitNameGuid(unittest.TestCase):
    def test_parse_name_and_guid(self):
        old_name1 = "示例名称_9d5b5e5b-ecf3-4a75-a0eb-1f6dcd2b245f"
        name1, guid1 = parse_name_and_guid(old_name1)
        self.assertEqual(name1, "示例名称")
        self.assertEqual(guid1, "9d5b5e5b-ecf3-4a75-a0eb-1f6dcd2b245f")

        old_name2 = "中文名称_IDb57e7425-c5bc-44d6-ae23-27f9f9e14de8"
        name2, guid2 = parse_name_and_guid(old_name2)
        self.assertEqual(name2, "中文名称")
        self.assertEqual(guid2, "b57e7425-c5bc-44d6-ae23-27f9f9e14de8")

        old_name3 = "无下划线"
        name3, guid3 = parse_name_and_guid(old_name3)
        self.assertEqual(name3, "无下划线")
        self.assertIsNone(guid3)

        old_name4 = "name_with_underscore_but_no_guid"
        name4, guid4 = parse_name_and_guid(old_name4)
        self.assertEqual(name4, "name_with_underscore_but_no_guid")
        self.assertIsNone(guid4)

        old_name5 = "ID_prefix_but_no_name_15c8f1bd-0e2a-49bb-9e6c-24f30b123ee3"
        name5, guid5 = parse_name_and_guid(old_name5)
        self.assertEqual(name5, "ID_prefix_but_no_name")
        self.assertEqual(guid5, "15c8f1bd-0e2a-49bb-9e6c-24f30b123ee3")


if __name__ == '__main__':
    unittest.main()
