# library.py
from book import Book
from member import Member

class Library:
    def __init__(self, name):
        self.name = name
        self.books = []
        self.members = []
    
    def add_book(self, title, author, isbn):
        book = Book(title, author, isbn)
        self.books.append(book)
        return f"Added '{title}' to library"
    
    def add_member(self, name, member_id):
        member = Member(name, member_id)
        self.members.append(member)
        return f"Added member: {name}"
    
    def find_book(self, title):
        for book in self.books:
            if book.title.lower() == title.lower():
                return book
        return None
    
    def find_member(self, member_id):
        for member in self.members:
            if member.member_id == member_id:
                return member
        return None
    
    def show_all_books(self):
        if not self.books:
            return "No books in library"
        
        result = f"\n=== {self.name} Library Books ===\n"
        for book in self.books:
            result += book.get_info() + "\n"
        return result
    
    def show_available_books(self):
        available = [book for book in self.books if not book.is_borrowed]
        
        if not available:
            return "No books available"
        
        result = f"\n=== Available Books ===\n"
        for book in available:
            result += book.get_info() + "\n"
        return result