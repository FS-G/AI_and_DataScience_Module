# main.py
from library import Library

def main():
    # Create library
    lib = Library("City Central")
    
    # Add some books
    lib.add_book("Python Programming", "John Smith", "123456789")
    lib.add_book("Web Development", "Jane Doe", "987654321")
    lib.add_book("Data Science", "Bob Wilson", "456789123")
    
    # Add members
    lib.add_member("Alice Johnson", "M001")
    lib.add_member("Charlie Brown", "M002")
    
    # Show all books
    print(lib.show_all_books())
    
    # Find members and books
    alice = lib.find_member("M001")
    python_book = lib.find_book("Python Programming")
    web_book = lib.find_book("Web Development")
    
    # Borrow books
    if alice and python_book:
        print(alice.borrow_book(python_book))
        print(alice.borrow_book(web_book))
    
    # Show member's books
    if alice:
        print(alice.get_borrowed_books())
    
    # Show available books
    print(lib.show_available_books())
    
    # Return a book
    if alice and python_book:
        print(alice.return_book(python_book))
    
    # Show available books again
    print(lib.show_available_books())

if __name__ == "__main__":
    main()