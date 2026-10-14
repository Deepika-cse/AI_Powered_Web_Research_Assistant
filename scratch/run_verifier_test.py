import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
from src.verifier import ClaimVerifier
from scratch.test_verifier import run_test

if __name__ == '__main__':
    run_test()
