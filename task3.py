import numpy as np

def input_matrix(name):
    rows = int(input(f"Enter number of rows for {name}: "))
    cols = int(input(f"Enter number of columns for {name}: "))
    print(f"Enter elements row-wise (space separated) for {name}:")
    
    elements = []
    for i in range(rows):
        row = list(map(float, input(f"Row {i+1}: ").split()))
        elements.append(row)
        
    return np.array(elements)

def print_matrix(mat, title="Result"):
    print(f"\n{title}:")
    print(mat)

while True:
    print("\n=== Matrix Operations Tool ===")
    print("1. Addition")
    print("2. Subtraction")
    print("3. Multiplication")
    print("4. Transpose")
    print("5. Determinant")
    print("6. Exit")

    choice = input("Choose an option (1-6): ")

    if choice in ['1', '2', '3']:
        A = input_matrix("Matrix A")
        B = input_matrix("Matrix B")

        try:
            if choice == '1':
                result = A + B
                print_matrix(result, "A + B")

            elif choice == '2':
                result = A - B
                print_matrix(result, "A - B")

            elif choice == '3':
                result = np.dot(A, B)
                print_matrix(result, "A x B")

        except ValueError:
            print("Error: Matrix dimensions are not compatible for this operation.")

    elif choice == '4':
        A = input_matrix("Matrix")
        result = A.T
        print_matrix(result, "Transpose")

    elif choice == '5':
        A = input_matrix("Matrix")
        if A.shape[0] == A.shape[1]:
            det = np.linalg.det(A)
            print(f"\nDeterminant: {det}")
        else:
            print("Error: Determinant can only be calculated for square matrices.")

    elif choice == '6':
        print("Exiting...")
        break

    else:
        print("Invalid choice. Try again.")
