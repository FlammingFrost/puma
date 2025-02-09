# TODO: Implement this module
# This module is for future development. Web interface is planned to implement using fastapi.
# interface/web_app.py
from fastapi import FastAPI
from query_engine.query_handler import process_query

app = FastAPI()

@app.get("/query/")
async def query_puma(query: str):
    response = process_query(query)
    return {"query": query, "response": response}