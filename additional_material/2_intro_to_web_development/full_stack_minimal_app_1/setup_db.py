import sqlite3

# Create database in same folder
conn = sqlite3.connect('guestbook.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    message TEXT NOT NULL
)
''')

# Add sample data
cursor.execute("INSERT INTO messages (name, message) VALUES (?, ?)", 
               ('Alice', 'Hello everyone!'))
cursor.execute("INSERT INTO messages (name, message) VALUES (?, ?)", 
               ('Bob', 'Great website!'))

# Save changes and close
conn.commit()
conn.close()

print("Database created successfully!")