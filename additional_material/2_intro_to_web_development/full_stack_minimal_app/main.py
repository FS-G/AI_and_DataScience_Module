from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import sqlite3

app = FastAPI()

# Enable CORS for frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
def init_db():
    conn = sqlite3.connect('guestbook.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS messages 
                    (id INTEGER PRIMARY KEY, name TEXT, message TEXT)''')
    conn.close()

init_db()

# API endpoint to get all messages
@app.get("/api/messages")
def get_messages():
    conn = sqlite3.connect('guestbook.db')
    cursor = conn.execute("SELECT name, message FROM messages")
    messages = [{"name": row[0], "message": row[1]} for row in cursor.fetchall()]
    conn.close()
    return {"messages": messages}

# API endpoint to add new message
@app.post("/api/messages")
def add_message(name: str = Form(...), message: str = Form(...)):
    conn = sqlite3.connect('guestbook.db')
    conn.execute("INSERT INTO messages (name, message) VALUES (?, ?)", (name, message))
    conn.commit()
    conn.close()
    return {"status": "Message added successfully"}