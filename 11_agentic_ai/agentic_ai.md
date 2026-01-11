# BASICS - ReAct agents, Tools, System Prompt

For detailed overview of langchain - visit their official documentation: https://docs.langchain.com/oss/python/langchain/overview

## Installation

```bash
pip install -U langchain
pip install -U langchain-google-genai
```

> **Note**: The `langchain-google-genai` package is to use gemini from AI studio - since it is free to use

## Setup

Get the API key from the AI studio and put it in here:

```python
import getpass
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
```

## Creating an LLM Instance

Make an instance of the LLM: https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai

> **NOTE**: We are using ChatGoogleGenerativeAI (LLM) as its provider is Google AI studio that gives free quota - In case if you want to change the provider or LLM, do refer to this: https://docs.langchain.com/oss/python/integrations/chat

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
```

## Agents

Agents combine language models with tools to create systems that can reason about tasks, decide which tools to use, and iteratively work towards solutions.

`create_agent` provides a production-ready agent implementation. An LLM Agent runs tools in a loop to achieve a goal. An agent runs until a stop condition is met - i.e., when the model emits a final output or an iteration limit is reached.

### Creating an Agent

```python
from langchain.agents import create_agent
```

## Tools

Tools give agents the ability to take actions. Agents go beyond simple model-only tool binding by facilitating:

- Multiple tool calls in sequence (triggered by a single prompt)
- Parallel tool calls when appropriate
- Dynamic tool selection based on previous results
- Tool retry logic and error handling
- State persistence across tool calls

The simplest way to create a tool is with the `@tool` decorator. By default, the function's docstring becomes the tool's description that helps the model understand when to use it.

**Tool use in the ReAct loop**: Agents follow the ReAct ("Reasoning + Acting") pattern, alternating between brief reasoning steps with targeted tool calls and feeding the resulting observations into subsequent decisions until they can deliver a final answer.

### Defining Tools

```python
from langchain.tools import tool

# define a tool - this is a mock tool just to test
@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"
```

### Creating an Agent with Tools

```python
agent = create_agent(
    model=llm,
    tools=[get_weather, search],
    system_prompt="You are a helpful assistant",
)
```

### Running the Agent

```python
# Run the agent - that can call the available tools
agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in Karachi?"}]}
)

# in case if we are interested in only the final answer
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in Karachi?"}]}
)
response['messages'][-1].content
```

## Tool Error Handling

To customize how tool errors are handled, use the `@wrap_tool_call` decorator to create middleware:

```python
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model=llm,
    tools=[get_weather, search],
    system_prompt="You are a helpful assistant",
    middleware=[handle_tool_errors]
)

# Run the agent - that can call the available tools
agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in Karachi?"}]}
)
```

> **Note**: The LLMs can be used directly too outside of the agent. For more details: https://docs.langchain.com/oss/python/langchain/models. However, our main focus will be on agents.

```python
llm.invoke("What is the weather in karachi?")
```

## Advanced Tool Usage: Getting Context

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime

# this is a mock DB which can be replaced by a real DB having millions of records
USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "alice@example.com"
    },
    "user456": {
        "name": "Bob Smith",
        "account_type": "Standard",
        "balance": 1200,
        "email": "bob@example.com"
    }
}

@dataclass
class UserContext:
    user_id: str

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information."""
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id] # replace this with actual DB query
        return f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}"
    return "User not found"

agent = create_agent(
    llm,
    tools=[get_account_info],
    context_schema=UserContext,
    system_prompt="You are a financial assistant."
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the account holder name and what is the status of the account as of now with balance?"}]},
    context=UserContext(user_id="user123")
)

result
result['messages'][-1].content
```

# Structured Output

`ToolStrategy` uses artificial tool calling to generate structured output. This works with any model that supports tool calling.

```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model=llm,
    tools=[search],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

# Memory

LangGraph implements memory systems for AI agents, distinguishing between two main types based on their scope.

- **Short-term memory** is thread-scoped, tracking conversations within a single session by maintaining message history as part of the agent's state. LangGraph persists this state to a database using checkpointers, allowing threads to resume at any time. The main challenge with short-term memory is managing long conversation histories that may exceed LLM context windows or degrade performance, requiring techniques to filter or remove stale information.

- **Long-term memory** stores information across sessions and threads in custom namespaces using LangGraph's store system. The framework categorizes long-term memory into three types inspired by human memory research: semantic memory (facts about users or concepts), episodic memory (past experiences and actions used for few-shot learning), and procedural memory (instructions and rules, typically stored in system prompts). Semantic memories can be managed either as a single continuously-updated profile or as a collection of documents. There are two approaches for writing memories: "in the hot path" during runtime for immediate availability but with added latency, or "in the background" as separate tasks that avoid impacting application performance but require careful timing considerations.

## Short Term Memory

To add short-term memory (thread-level persistence) to an agent, you need to specify a checkpointer when creating an agent.

For detailed information related to short term memory, refer to this page: https://docs.langchain.com/oss/python/langchain/short-term-memory

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=llm,
    tools=[search],
    checkpointer=InMemorySaver(),
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    {"configurable": {"thread_id": "1"}},
)
```

Check the agent again with same thread:

```python
agent.invoke(
    {"messages": [{"role": "user", "content": "what was my name?"}]},
    {"configurable": {"thread_id": "1"}},
)
```

Now change the thread - the agent will not have the information of the name:

```python
agent.invoke(
    {"messages": [{"role": "user", "content": "what was my name?"}]},
    {"configurable": {"thread_id": "2"}},
)
```

### For Production

We know - for production, we need some persistent memory and we CAN NEVER RELY ON SYSTEM'S STATE - for that, we need to integrate a separately managed DB. Here, we are doing that with Postgres.

```bash
pip install langgraph-checkpoint-postgres
```

```python
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver

# note: do replace this URL with an actual postgres URL
DB_URI = "postgresql://postgres:cNXwBkYtelZXvyFReGdIdjlUxWcwpbwD@caboose.proxy.rlwy.net:51520/railway"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup() # auto create tables in PostgresSql
    agent = create_agent(
        llm,
        [search],
        checkpointer=checkpointer,
    )

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Hello my name is Bob"}]},
        {"configurable": {"thread_id": "1"}},
    )

    print(response)

    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "what was my name?"}]},
        {"configurable": {"thread_id": "1"}},
    )

    print("\n\n\n\n")
    print(response2)
```

Another example with a different thread:

```python
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver

# note: do replace this URL with an actual postgres URL
DB_URI = "postgresql://postgres:cNXwBkYtelZXvyFReGdIdjlUxWcwpbwD@caboose.proxy.rlwy.net:51520/railway"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup() # auto create tables in PostgresSql
    agent = create_agent(
        llm,
        [search],
        checkpointer=checkpointer,
    )

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Hello my name is Alice"}]},
        {"configurable": {"thread_id": "2"}},
    )

    print(response)

    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "what was my name?"}]},
        {"configurable": {"thread_id": "2"}},
    )

    print("\n\n\n\n")
    print(response2)
```

### Customizing Memory

By default, agents use `AgentState` to manage short term memory, specifically the conversation history via a messages key. You can extend `AgentState` to add additional fields. Custom state schemas are passed to `create_agent` using the `state_schema` parameter.

```python
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver

class CustomAgentState(AgentState):
    user_id: str
    preferences: dict

agent = create_agent(
    llm,
    [search],
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),
)

# Custom state can be passed in invoke
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark"}
    },
    {"configurable": {"thread_id": "1"}}
)

result
```

Custom message state saved to production DB:

```python
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver

# note: do replace this URL with an actual postgres URL
DB_URI = "postgresql://postgres:cNXwBkYtelZXvyFReGdIdjlUxWcwpbwD@caboose.proxy.rlwy.net:51520/railway"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup() # auto create tables in PostgresSql
    agent = create_agent(
        llm,
        [search],
        state_schema=CustomAgentState,
        checkpointer=checkpointer,
    )

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Hello my name is Alice"}]},
        {"configurable": {"thread_id": "2"}},
    )

    print(response)

    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "what was my name?"}]},
        {"configurable": {"thread_id": "2"}},
    )

    print("\n\n\n\n")
    print(response2)
```

### Short Term Memory Management

#### Trimming

Most LLMs have a maximum supported context window (denominated in tokens). One way to decide when to truncate messages is to count the tokens in the message history and truncate whenever it approaches that limit. If you're using LangChain, you can use the trim messages utility and specify the number of tokens to keep from the list, as well as the strategy (e.g., keep the last max_tokens) to use for handling the boundary.

To trim message history in an agent, use the `@before_model` middleware decorator:

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any

# trim to 3 messages here
@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent = create_agent(
    llm,
    [search],
    middleware=[trim_messages],
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
```

Output:
```
================================== Ai Message ==================================

Your name is Bob. You told me that earlier.
If you'd like me to call you a nickname or use a different name, just say the word.
```

#### Delete Message

You can delete messages from the graph state to manage the message history. This is useful when you want to remove specific messages or clear the entire message history. To delete messages from the graph state, you can use the `RemoveMessage`.

```python
from langchain.messages import RemoveMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig

@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove old messages to keep conversation manageable."""
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None

agent = create_agent(
    llm,
    [search],
    middleware=[delete_old_messages],
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

# here we are using agent.stream - however, you can still use the same agent.invoke
for event in agent.stream(
    {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
    config,
    stream_mode="values",
):
    print([(message.type, message.content) for message in event["messages"]])

for event in agent.stream(
    {"messages": [{"role": "user", "content": "what's my name?"}]},
    config,
    stream_mode="values",
):
    print([(message.type, message.content) for message in event["messages"]])
```

#### Summarize Message

The problem with trimming or removing messages, as shown above, is that you may lose information from culling of the message queue. Because of this, some applications benefit from a more sophisticated approach of summarizing the message history using a chat model.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

checkpointer = InMemorySaver()

agent = create_agent(
    llm,
    [search],
    middleware=[
        SummarizationMiddleware(
            model=llm, # the same LLM here is used as summarizer
            max_tokens_before_summary=4000,  # Trigger summarization at 4000 tokens
            messages_to_keep=20,  # Keep last 20 messages after summary
        )
    ],
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
```

Output:
```
================================== Ai Message ==================================

Your name is Bob!
```

# Middlewares

For details related to middleware, visit this page: https://docs.langchain.com/oss/python/langchain/middleware/overview

Middleware provides a way to more tightly control what happens inside the agent. Middleware is useful for the following:

- Tracking agent behavior with logging, analytics, and debugging.
- Transforming prompts, tool selection, and output formatting.
- Adding retries, fallbacks, and early termination logic.
- Applying rate limits, guardrails, and PII detection.

One of the common use case:

The core agent loop involves calling a model, letting it choose tools to execute, and then finishing when it calls no more tools. Middleware exposes hooks before and after each of those steps.

We can use the builtin middlewares and can also make custom middleware:

- For builtin middleware - refer to this: https://docs.langchain.com/oss/python/langchain/middleware/built-in
- For custom middleware - refer to this: https://docs.langchain.com/oss/python/langchain/middleware/custom

## Summarization

We already used this middleware above when summarizing the short term memory.

## Human in the Loop

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

@tool
def read_email_tool(email_id: str) -> str:
    """Read an email by its ID."""
    return f"Email {email_id} content: This is a mock email body."

@tool
def send_email_tool(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    print("\n\n\n\nsend email tool executed\n'\n\n\n")
    return f"Mock email sent to {to} with subject '{subject}'."

agent = create_agent(
    model=llm,
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "read_email_tool": False,
            }
        ),
    ],
)

config: RunnableConfig = {"configurable": {"thread_id": "2"}}
agent.invoke({"messages": "send email to farhan.sid to join from tomorrow"}, config)
agent.invoke({"messages": "offer letter related"}, config)
```

Now let's consider a minimal workable agentic example where we put in human in the loop to save a file with custom content:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from pathlib import Path

@tool
def read_email_tool(email_id: str) -> str:
    """Read an email by its ID."""
    return f"Email {email_id} content: This is a mock email body."

@tool
def write_file_tool(filename: str, content: str) -> str:
    """
    Write content to a file in the current directory.

    Args:
        filename: Name of the file to create
        content: Content to write to the file
    """
    try:
        # Write content to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"\n✅ File successfully written: {os.path.abspath(filename)}\n")
        return f"File '{filename}' successfully created at {os.path.abspath(filename)}"

    except Exception as e:
        return f"Error writing file: {str(e)}"

agent = create_agent(
    model=llm,
    tools=[read_email_tool, write_file_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "read_email_tool": False,
            }
        ),
    ],
)

config: RunnableConfig = {"configurable": {"thread_id": "3"}}
agent.invoke({"messages": "write an essay on generative ai and save to a local file"}, config)

# Resume with approval decision
from langgraph.types import Command

agent.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}  # or "edit", "reject"
    ),
    config=config # Same thread ID to resume the paused conversation
)
```

## Guardrails

For detailed info - refer to this: https://docs.langchain.com/oss/python/langchain/guardrails

Guardrails help you build safe, compliant AI applications by validating and filtering content at key points in your agent's execution. They can detect sensitive information, enforce content policies, validate outputs, and prevent unsafe behaviors before they cause problems.

Common use cases include:
- Preventing PII leakage
- Detecting and blocking prompt injection attacks
- Blocking inappropriate or harmful content
- Enforcing business rules and compliance requirements
- Validating output quality and accuracy

You can implement guardrails using middleware to intercept execution at strategic points - before the agent starts, after it completes, or around model and tool calls.

Important thing to consider - the guard rails come in action before agent and then before model and then after agent and after model too.

| Strategy | Description | Example |
|----------|-------------|---------|
| redact | Replace with [REDACTED_TYPE] | [REDACTED_EMAIL] |
| mask | Partially obscure (e.g., last 4 digits) | ****-****-****-1234 |
| hash | Replace with deterministic hash | a8f5f167... |
| block | Raise exception when detected | Error thrown |

The **human in the loop** that we learnt in the previous section is also a guard rail that requires human approval. Also, we can make custom guard rails too.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model=llm,
    # tools=[customer_service_tool, email_tool],
    middleware=[
        # Redact emails in user input before sending to model
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        # Mask credit cards in user input
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        # Block API keys - raise error if detected
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
        ),
    ],
)

# When user provides PII, it will be handled according to the strategy
result = agent.invoke({
    "messages": [{"role": "user", "content": "My email is john.doe@example.com and card is 5105-1051-0510-5100"}]
})

result
```

# RAG and LangGraph

## 🔍 Brief Summary of Retrieval & RAG (with the 3 types included)

Retrieval helps LLMs overcome their limited context and static knowledge by fetching relevant information from external sources at query time. This enables **Retrieval-Augmented Generation (RAG)** — where an LLM uses retrieved context to produce accurate, grounded answers.

A **knowledge base** (documents, databases, CRMs, etc.) supplies this external information. Data is loaded, chunked, embedded, and stored in a vector store so it can be searched efficiently during queries.

Once retrieval is added to generation, three main **RAG architectures** can be used:

---

## Types of RAG

### 1. 2-Step RAG

A fixed pipeline:  
**Retrieve first → then generate.**  
Simple, predictable, and fast. Best for FAQs, documentation bots, and straightforward question-answering.

### 2. Agentic RAG

An agent (LLM) reasons step-by-step and decides **when** and **how** to retrieve using tools.  
More flexible but less predictable latency. Ideal for research assistants or multi-tool systems.

### 3. Hybrid RAG

Mixes both approaches with extra steps like query rewriting, retrieval validation, and answer checking.  
Good for ambiguous queries or domains requiring high accuracy.

---

Overall, RAG enhances LLM capabilities by dynamically grounding answers in relevant, up-to-date context—making applications more reliable and domain-aware.

## 2-Step RAG

```bash
pip install pinecone langchain-huggingface
pip install langchain-pinecone
pip install langchain-community
```

```python
import os
import bs4
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_huggingface import HuggingFaceEmbeddings

# 1. SETUP & CONFIGURATION
# Ensure you have set OPENAI_API_KEY and PINECONE_API_KEY in your environment
# os.environ["OPENAI_API_KEY"] = "sk-..."
# os.environ["PINECONE_API_KEY"] = "pc-..."

INDEX_NAME = "testnew"
model = llm

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_3Wxd5v_HMfV1CfiAhiosQovbt6Tfv6bvEGRi1aHkKaSHjL25uoJj1s6CdKAcWNNSzF8HhF")
index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(embedding=embeddings, index=index)

# 2. INDEXING (Load -> Split -> Store)
# Load data
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Add to Pinecone
print("Indexing documents...")
_ = vector_store.add_documents(documents=all_splits)
print("Indexing complete.")

# 3. DEFINE 2-STEP RAG LOGIC (Middleware)
# This implements the "RAG Chain" approach: Retrieve -> Generate
@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    # Step 1: Retrieve
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query, k=2)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Step 2: Generate System Message with Context
    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message

# 4. CREATE AGENT
# We pass empty tools list because retrieval is handled in the middleware
agent = create_agent(model, tools=[], middleware=[prompt_with_context])

# 5. RUN
query = "What is task decomposition?"

print(f"\nUser Query: {query}\n")

for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

## Agentic RAG

```python
import os
import bs4
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool

# 1. SETUP & CONFIGURATION
INDEX_NAME = "testnew"

# Using your defined llm instance
model = llm

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_3Wxd5v_HMfV1CfiAhiosQovbt6Tfv6bvEGRi1aHkKaSHjL25uoJj1s6CdKAcWNNSzF8HhF")
index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(embedding=embeddings, index=index)

# 2. INDEXING (Load -> Split -> Store)
# Load data
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Add to Pinecone
print("Indexing documents...")
_ = vector_store.add_documents(documents=all_splits)
print("Indexing complete.")

# 3. DEFINE AGENTIC RAG LOGIC (Tool)
# This implements the "RAG Agent" approach: Model decides to call tool
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    # Perform the search on Pinecone
    retrieved_docs = vector_store.similarity_search(query, k=2)

    # Serialize the content for the LLM
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    # Return both the string for the model and the raw docs as artifacts
    return serialized, retrieved_docs

# 4. CREATE AGENT
tools = [retrieve_context]

# Instructions for the agent
system_prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)

# Create agent with tools enabled
agent = create_agent(model, tools, system_prompt=system_prompt)

# 5. RUN
query = "What is task decomposition?"

print(f"\nUser Query: {query}\n")

for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

## Hybrid RAG

```bash
pip install langchain-classic
```

```python
import os
from typing import Literal
from pinecone import Pinecone
from pydantic import BaseModel, Field
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# 1. SETUP & CONFIGURATION
INDEX_NAME = "testnew"

# Assuming 'llm' is already defined in your environment
model = llm

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_3Wxd5v_HMfV1CfiAhiosQovbt6Tfv6bvEGRi1aHkKaSHjL25uoJj1s6CdKAcWNNSzF8HhF")
index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(embedding=embeddings, index=index)

# 2. INDEXING
# Fetch documents
urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# Index to Pinecone
print("Indexing documents...")
_ = vector_store.add_documents(documents=doc_splits)
print("Indexing complete.")

# 3. CREATE RETRIEVER TOOL (FIXED)
# We manually define the tool to avoid the functools.partial TypeError
retriever = vector_store.as_retriever()

@tool
def retrieve_blog_posts(query: str):
    """Search and return information about Lilian Weng blog posts."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# 4. DEFINE NODES & LOGIC

# Node 1: Generate Query or Respond
def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response or call a tool."""
    # Bind the tool manually created above
    response = model.bind_tools([retrieve_blog_posts]).invoke(state["messages"])
    return {"messages": [response]}

# Node 2: Grade Documents
class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    GRADE_PROMPT = (
        "You are a grader assessing relevance of a retrieved document to a user question. \n "
        "Here is the retrieved document: \n\n {context} \n\n"
        "Here is the user question: {question} \n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )

    prompt = GRADE_PROMPT.format(question=question, context=context)
    # Requires a model that supports structured output
    response = model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"

# Node 3: Rewrite Question
def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    question = state["messages"][0].content

    REWRITE_PROMPT = (
        "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
        "Here is the initial question:"
        "\n ------- \n"
        "{question}"
        "\n ------- \n"
        "Formulate an improved question:"
    )

    prompt = REWRITE_PROMPT.format(question=question)
    response = model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}

# Node 4: Generate Answer
def generate_answer(state: MessagesState):
    """Generate the final answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    GENERATE_PROMPT = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n"
        "Question: {question} \n"
        "Context: {context}"
    )

    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

# 5. ASSEMBLE GRAPH
workflow = StateGraph(MessagesState)

# Add Nodes
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retrieve_blog_posts])) # Using fixed tool
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)

# Add Edges
workflow.add_edge(START, "generate_query_or_respond")

# Conditional Edge: Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

# Conditional Edge: Grade documents after retrieval
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)

# Remaining Edges
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile()

# 6. RUN
print("\nRunning Hybrid RAG Agent...\n")

query = "What does Lilian Weng say about types of reward hacking?"
print(f"Query: {query}\n")

for chunk in graph.stream(
    {"messages": [{"role": "user", "content": query}]}
):
    for node, update in chunk.items():
        print(f"--- Update from node: {node} ---")
        if "messages" in update:
            update["messages"][-1].pretty_print()

# 6. PLOT THE GRAPH
from IPython.display import Image, display
try:
    print("Generating Graph Visualization...")
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"Could not plot graph: {e}")
    # Fallback: print text representation
    print(graph.get_graph().print_ascii())
```

# SQL Agent (Database)

In this tutorial, you will learn how to build an agent that can answer questions about a SQL database using LangChain agents.

At a high level, the agent will:

1. Fetch the available tables and schemas from the database
2. Decide which tables are relevant to the question
3. Fetch the schemas for the relevant tables
4. Generate a query based on the question and information from the schemas
5. Double-check the query for common mistakes using an LLM
6. Execute the query and return the results
7. Correct mistakes surfaced by the database engine until the query is successful
8. Formulate a response based on the results

```python
import os
import requests
import pathlib
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# 1. SETUP & CONFIGURATION
# We assume 'llm' is already defined in your environment
model = llm

# 2. CONFIGURE DATABASE
# Download the Chinook database if it doesn't exist
url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = pathlib.Path("Chinook.db")

if local_path.exists():
    print(f"{local_path} already exists, skipping download.")
else:
    response = requests.get(url)
    if response.status_code == 200:
        local_path.write_bytes(response.content)
        print(f"File downloaded and saved as {local_path}")
    else:
        print(f"Failed to download the file.")

# Initialize Database
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(f"Dialect: {db.dialect}")
print(f"Available tables: {db.get_usable_table_names()}")

# 3. ADD TOOLS
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# 4. CREATE AGENT WITH HUMAN-IN-THE-LOOP
system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

# Initialize Memory (Checkpointer) and Middleware
checkpointer = InMemorySaver()

agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"sql_db_query": True}, # Pause before executing actual SQL
            description_prefix="Tool execution pending approval",
        ),
    ],
    checkpointer=checkpointer,
)

# 5. RUN THE AGENT
question = "Which employee has made the most number of invoices?"
config = {"configurable": {"thread_id": "1"}}

print(f"\nQuestion: {question}\n")

# Phase 1: Run until interrupt (Agent generates query, but stops before execution)
print("--- STARTING RUN ---")
last_interrupt = None

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    config,
    stream_mode="values",
):
    if "messages" in step:
        step["messages"][-1].pretty_print()
    elif "__interrupt__" in step:
        print("\n🔴 INTERRUPTED FOR HUMAN REVIEW 🔴")
        last_interrupt = step["__interrupt__"][0]
        for request in last_interrupt.value["action_requests"]:
            print(f"Description: {request['description']}")
            # In a real app, you would inspect the args here: request['args']

# Phase 2: Resume execution (Human approves the query)
if last_interrupt:
    print("\n🟢 RESUMING EXECUTION (APPROVING QUERY) 🟢")

    # We send an "approve" decision back to the graph
    for step in agent.stream(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config,
        stream_mode="values",
    ):
        if "messages" in step:
            step["messages"][-1].pretty_print()
```

# Multi Agent

## Multi-agent Overview

Multi-agent systems break a complex application into multiple specialized agents that work together. Instead of one agent handling everything, multiple focused agents collaborate to improve accuracy, memory handling, and specialization.

### When multi-agent systems help

- When one agent has too many tools.
- When context or memory becomes too large.
- When tasks need specialists (planner, researcher, math expert).

## Multi-agent patterns

### 1. Tool Calling

A main **controller** agent invokes other agents as tools.  
- Centralized decision-making.  
- Subagents don't talk to the user directly.  
- Best for structured workflows and orchestration.

### 2. Handoffs

Agents pass control to each other.  
- Decentralized.  
- The active agent interacts directly with the user.  
- Great for multi-domain conversations.

## Choosing a pattern

| Need | Tool Calling | Handoffs |
|------|--------------|----------|
| Centralized control | ✅ | ❌ |
| Agents talk to user | ❌ | ✅ |
| Specialist conversation | Limited | Strong |

Tip: You can combine both patterns—handoffs for switching, tool calls inside each agent.

## Context engineering

Success depends on giving each agent the right context, including:
- Relevant parts of the conversation.
- Specialized prompts.
- Custom input/output formats.
- What state each agent sees.

## Tool Calling (Flow)

1. Controller receives input.  
2. Chooses a subagent tool.  
3. Subagent processes the task.  
4. Controller uses the result and decides next steps.

You can customize:
- Subagent names and descriptions.
- Input sent to the subagent.
- Output returned to the main agent.

## Handoffs (Flow)

1. Current agent decides another agent should take over.  
2. Passes control and state.  
3. New agent interacts with the user until another handoff or completion.

More implementation details coming soon.

## Implementation Example

```python
import os
from langchain.tools import tool
from langchain.agents import create_agent

# 1. SETUP
# Assuming 'llm' is already defined in your environment
model = llm

# 2. DEFINE SUB-AGENTS
# These agents operate independently but will be called as tools.

# Sub-agent 1: The Math Expert
math_agent = create_agent(
    model,
    tools=[],
    system_prompt="You are a math expert. Solve the problem and give the final answer clearly."
)

# Sub-agent 2: The Creative Writer
writer_agent = create_agent(
    model,
    tools=[],
    system_prompt="You are a creative writer. Write short, engaging poems or stories based on the input."
)

# 3. WRAP SUB-AGENTS AS TOOLS
# The Supervisor communicates with these functions, which in turn invoke the sub-agents.

@tool("ask_math_expert", description="Useful for solving math problems or calculations.")
def call_math_agent(query: str):
    """Delegates a math query to the specialized math agent."""
    print(f"  --> Supervisor delegating to Math Agent: '{query}'")
    result = math_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    # Return the content of the sub-agent's final response
    return result["messages"][-1].content

@tool("ask_creative_writer", description="Useful for writing poems, stories, or creative content.")
def call_writer_agent(query: str):
    """Delegates a creative task to the specialized writer agent."""
    print(f"  --> Supervisor delegating to Writer Agent: '{query}'")
    result = writer_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content

# 4. DEFINE SUPERVISOR AGENT
# This agent sees the sub-agents as tools and decides which one to call.
supervisor_tools = [call_math_agent, call_writer_agent]

supervisor_prompt = (
    "You are a Supervisor Agent. You have access to specialized tools: "
    "a Math Expert and a Creative Writer. "
    "Delegate the user's task to the appropriate tool based on their request. "
    "Do not answer the question yourself if a tool is better suited."
)

supervisor_agent = create_agent(
    model,
    tools=supervisor_tools,
    system_prompt=supervisor_prompt
)

# 5. RUN THE SYSTEM
query = "First calculate 25 * 4, and then ask the writer to write a poem about that number."

print(f"\nUser Query: {query}\n")
print("--- STARTING SUPERVISOR ---")

for step in supervisor_agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    # print the final message of the step
    step["messages"][-1].pretty_print()
```
