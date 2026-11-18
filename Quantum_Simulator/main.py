"""Command-line interface for the Quantum Atom Simulator."""

from renderer import render
from maths import SPECTROSCOPIC_LETTERS


def get_input(user_input: str, min_val: int, max_val: int) -> int:
    """Prompt user for integer input within [min_val, max_val]."""
    while True:
        try:
            raw = input(user_input).strip()
            value = int(raw)
            if min_val <= value <= max_val:
                return value
            print(f"Please enter a value between {min_val} and {max_val}")
        except ValueError:
            print("Please enter a whole number")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            exit(0)


def main() -> None:
    """Main CLI interactive loop."""
    while True:
        print("================================")
        print("   QUANTUM ATOM SIMULATOR")
        print("================================\n")

        # Get quantum numbers from user
        n = get_input("Enter n (principal quantum number, 1-5): ", 1, 5)
        l = get_input(f"Enter l (angular quantum number, 0-{n-1}): ", 0, n - 1)
        m = get_input(f"Enter m (magnetic quantum number, {-l}-{l}): ", -l, l)
        Z = get_input("Enter Z (atomic number, 1-79): ", 1, 79)

        letter = SPECTROSCOPIC_LETTERS[l] if l < len(SPECTROSCOPIC_LETTERS) else "?"
        print("\nQuantum numbers accepted!")
        print(f"Rendering {n}{letter} orbital for Z={Z}...\n")

        # Render the orbital
        render(n, l, m, Z, num_points=3000)

        # Ask to run again
        try:
            again = input("\nRender another orbital? (y/n): ").strip().lower()
            if again != "y":
                print("Goodbye!")
                break
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
