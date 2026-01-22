students = {'Alice': 'A', 'Bob': 'B', 'Charlie': 'C', 'David': 'D'}

sorted_students = sorted(students.items(), key=lambda x: len(x[0]))

print(sorted_students)