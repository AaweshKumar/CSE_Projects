from renderer import render

def get_input(user_input, min_val, max_val):
    while True:
        try:
            value = int(input(user_input))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Please enter a value between {min_val} and {max_val}")
        except ValueError:
            print("Please enter a whole number")

def main():
    print("================================")
    print("   QUANTUM ATOM SIMULATOR")
    print("================================\n")

    # Get quantum numbers from user
    n = get_input("Enter n (principal quantum number, 1-5): ", 1, 5)
    l = get_input(f"Enter l (angular quantum number, 0-{n-1}): ", 0, n-1)
    m = get_input(f"Enter m (magnetic quantum number, {-l}-{l}): ", -l, l)
    Z = get_input("Enter Z (atomic number, 1-79): ", 1, 79)

    print("\nQuantum numbers accepted!")
    print(f"Rendering {n}{'spldf'[l]} orbital for Z={Z}...\n")

    # Render the orbital
    render(n, l, m, Z, num_points=3000)

    # Ask to run again
    again = input("\nRender another orbital? (y/n): ")
    if again.lower() == 'y':
        main()
    else:
        print("Goodbye!")

if __name__ == "__main__":
    main()
