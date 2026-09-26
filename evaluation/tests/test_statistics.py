import unittest
import numpy as np
from full_statistics import holm, metric, compare_arrays

class StatisticsTests(unittest.TestCase):
    def test_holm(self):
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
    def test_mmhal_mean(self):
        self.assertEqual(float(metric('mmhal',np.array([9,3]),1)),3)
    def test_amber_hal_direction(self):
        self.assertAlmostEqual(float(metric('amber',np.array([3,4]),1)),25)
    def test_pope_macro_not_micro(self):
        # one perfect category and one with F1=0 => macro F1 50
        x=np.array([2,0,0,2,2,2,0,0,0,2,0,2,0,0])
        self.assertEqual(float(metric('pope',x,2)),50)
    def test_paired_identical(self):
        a=np.array([[1,1],[2,1],[3,1]],dtype=float)
        weights=np.tile([1,1,1],(20,1));swaps=np.zeros((20,3))
        result=compare_arrays('mmhal',a,a,1,weights,swaps)
        self.assertEqual(result['benefit_delta_A_minus_comparator'],0)
        self.assertEqual(result['paired_cluster_permutation_p'],1)
        self.assertEqual(result['ci95'],[0,0])
    def test_lower_hallucination_is_positive_benefit(self):
        a=np.array([[0,1],[0,1]],dtype=float);b=np.array([[1,1],[1,1]],dtype=float)
        result=compare_arrays('object_halbench',a,b,1,np.tile([1,1],(10,1)),np.zeros((10,2)))
        self.assertEqual(result['benefit_delta_A_minus_comparator'],100)

if __name__=='__main__': unittest.main()
