# member.py
class Member:
    def __init__(self, name, member_id):
        self.name = name
        self.member_id = member_id
        self.borrowed_books = []
    
    def borrow_book(self, book):
        if len(self.borrowed_books) < 3:  # Max 3 books
            result = book.borrow(self.name)
            if "borrowed by" in result:
                self.borrowed_books.append(book)
            return result
        return "Cannot borrow more than 3 books"
    
    def return_book(self, book):
        if book in self.borrowed_books:
            result = book.return_book()
            self.borrowed_books.remove(book)
            return result
        return "You haven't borrowed this book"
    
    def get_borrowed_books(self):
        if not self.borrowed_books:
            return f"{self.name} has no borrowed books"
        
        books_list = [book.title for book in self.borrowed_books]
        return f"{self.name} has borrowed: {', '.join(books_list)}"