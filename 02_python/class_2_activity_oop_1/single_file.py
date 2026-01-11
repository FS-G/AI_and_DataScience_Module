# student.py
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.grades = []
    
    def add_grade(self, grade):
        self.grades.append(grade)
    
    def get_average(self):
        if len(self.grades) == 0:
            return 0
        return sum(self.grades) / len(self.grades)
    
    def introduce(self):
        return f"Hi! I'm {self.name} and I'm {self.age} years old."
    

# test_student.py
# from student import Student

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