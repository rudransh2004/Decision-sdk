import sys

from futurix_decision.cli import main

if __name__ == "__main__":
    main(["serve", *sys.argv[1:]])
