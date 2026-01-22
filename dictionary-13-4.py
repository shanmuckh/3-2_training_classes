students = {}

while True:
    print("\n1. Add new student")
    print("2. Update existing student")
    print("3. Delete a student")
    print("4. View all students")
    print("5. Exit")
    
    choice = input("Select an option: ")
    
    if choice == '1':
        name = input("Enter student name: ")
        grade = input("Enter student grade: ")
        students[name] = grade
        print("Student added.")
    
    elif choice == '2':
        name = input("Enter student name to update: ")
        if name in students:
            grade = input("Enter new grade: ")
            students[name] = grade
            print("Student updated.")
        else:
            print("Student not found.")
    
    elif choice == '3':
        name = input("Enter student name to delete: ")
        if name in students:
            del students[name]
            print("Student deleted.")
        else:
            print("Student not found.")
    
    elif choice == '4':
        if students:
            for name, grade in students.items():
                print(f"{name}: {grade}")
        else:
            print("No students.")
    
    elif choice == '5':
        break
    
    else:
        print("Invalid option.")