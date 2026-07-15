"""Verify and load the local March Madness data files."""

from src.data_loader import load_mens_data, print_summary


if __name__ == "__main__":
    data = load_mens_data()
    print_summary(data)
