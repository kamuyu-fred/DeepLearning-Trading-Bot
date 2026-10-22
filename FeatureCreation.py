"""Prepare versioned training, validation, and test datasets."""
import argparse
from pathlib import Path

from pipeline import Config, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=Path(__file__).with_name('eurusd-15m.csv'))
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    args = parser.parse_args()
    print(prepare(args.csv, args.output, Config()))


if __name__ == '__main__':
    main()
