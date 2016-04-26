"""
File rbm_test.py
This file tests the functions within the RBM class, it is purely to check that the
RBM class runs, this checks for syntax errors and run time errors, not logic errors
Author: Forest Thomas
"""

from RBM import *
import getvectors

def main():
    rbm = RBM(17771, 100, 5, .01)
    print("Creation passed")
    rbm.save_RBM("test_start.rbm")
    print("Saving passed")
    rbm = RBM.load_RBM("test_start.rbm")
    print("Loading passed")
    a = []
    for i in range(1, 10):
        a = []
        for j in range(0, 10):
            a.append(getvectors.getFormattedVector())
        print("batch loaded... learning...")
        rbm.learn_batch(1, a)
        print("batch {0} completed".format(i))
    print("learn_batch passed")
    print(rbm.get_highest_rating(a[0], getRated(a[0])))

if __name__ == "__main__":
    main()
