from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
import logging
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="LangGraph FastAPI Server", version="1.0")
origins = [
    "http://localhost:3000",  
    "http://127.0.0.1:3000",
    # Add other origins as needed
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request body model
class QueryRequest(BaseModel):
    query: str

class State(TypedDict):
    messages: Annotated[list, add_messages]

chat_model = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=2,
)


def chatbot(state: State):
    return {"messages": [chat_model.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    """API endpoint for processing user queries using LangGraph."""
    try:
       # Convert query into state format
        state = {"messages": [HumanMessage(content=request.query)]}
        result = graph.invoke(state)
        ai_response = next(
            (msg.content for msg in result["messages"] if isinstance(msg, AIMessage)),
            "No AI response generated."
        )
        print("ai_response :", ai_response)
        return {"response": ai_response}
    except Exception as e:
        logger.error(f"Error in LangGraph execution: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing request")

@app.get("/")
def root():
    return {"message": "Welcome to the LangGraph FastAPI Server"}
