"""
Main FastAPI application entry point for the Agentic Research Assistant.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router as api_router

# Create FastAPI app
app = FastAPI(
    title="Agentic Research Assistant",
    description="An AI-powered autonomous research analyst",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Agentic Research Assistant API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}