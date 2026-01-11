# book.py
class Book:
    def __init__(self, title, author, isbn):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.is_borrowed = False
        self.borrowed_by = None
    
    def borrow(self, member_name):
        if not self.is_borrowed:
            self.is_borrowed = True
            self.borrowed_by = member_name
            return f"'{self.title}' borrowed by {member_name}"
        return f"'{self.title}' is already borrowed"
    
    def return_book(self):
        if self.is_borrowed:
            borrower = self.borrowed_by
            self.is_borrowed = False
            self.borrowed_by = None
            return f"'{self.title}' returned by {borrower}"
        return f"'{self.title}' was not borrowed"
    
    def get_info(self):
        status = "Available" if not self.is_borrowed else f"Borrowed by {self.borrowed_by}"
        return f"'{self.title}' by {self.author} - {status}"