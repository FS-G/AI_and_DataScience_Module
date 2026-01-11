# test_student.py
from student import Student

# Creating objects (instances)
john = Student("John", 20)
mary = Student("Mary", 19)

# Using methods
print(john.introduce())  # Hi! I'm John and I'm 20 years old.
print(mary.introduce())  # Hi! I'm Mary and I'm 19 years old.

# Adding grades
john.add_grade(85)
john.add_grade(92)
john.add_grade(78)

print(f"John's average: {john.get_average()}")  # John's average: 85.0