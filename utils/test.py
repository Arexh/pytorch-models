import pandas as pd
import unittest
import util


class TestUtil(unittest.TestCase):

    def test_filter_values_with_threadhold(self):
        testdata1 = {
            "a": [1, 2, 1, 1],
            "b": [2, 3, 4, 5],
            "c": [2, 4, 3, 3],
            "d": [4, 5, 5, 4]
        }
        dataframe = pd.DataFrame(testdata1)
        dataframe1 = dataframe.copy()
        util.filter_values_with_threadhold(dataframe1, 2, 0)
        result1 = {
            "a": [1, 0, 1, 1],
            "b": [0, 0, 0, 0],
            "c": [0, 0, 3, 3],
            "d": [4, 5, 5, 4]
        }
        self.assertEqual(dataframe1.to_dict('list'), result1)

        dataframe2 = dataframe.copy()
        util.filter_values_with_threadhold(dataframe2, 2, -1)
        result2 = {
            "a": [1,-1, 1, 1],
            "b": [-1,-1,-1,-1],
            "c": [-1,-1, 3, 3],
            "d": [4, 5, 5, 4]
        }
        self.assertEqual(dataframe2.to_dict('list'), result2)

        dataframe3 = dataframe.copy()
        util.filter_values_with_threadhold(dataframe3, -1, -1)
        result3 = {
            "a": [1, 2, 1, 1],
            "b": [2, 3, 4, 5],
            "c": [2, 4, 3, 3],
            "d": [4, 5, 5, 4]
        }
        self.assertEqual(dataframe3.to_dict('list'), result3)

        dataframe4 = dataframe.copy()
        util.filter_values_with_threadhold(dataframe4, 10, -1)
        result4 = {
            "a": [-1,-1,-1,-1],
            "b": [-1,-1,-1,-1],
            "c": [-1,-1,-1,-1],
            "d": [-1,-1,-1,-1]
        }
        self.assertEqual(dataframe4.to_dict('list'), result4)

if __name__ == "__main__":
    unittest.main()
