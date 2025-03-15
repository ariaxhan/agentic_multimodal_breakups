"""
AI Agent Communication System

This module implements a system where two AI agents can engage in direct conversation with each other.
"""

import os
from dotenv import load_dotenv
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
import weave
# Initialize weave
weave.init("breakup")
import replicate
from agno.models.groq import Groq

from typing import List, Optional
import random
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.anthropic import Claude
from datetime import datetime

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")

def get_random_persona() -> str:
    """
    Randomly select a persona file from the prompts/personas directory.
    
    Returns:
        str: The contents of the randomly selected persona prompt
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    personas_dir = os.path.join(current_dir, "prompts", "personas")
    
    # Get all .txt files from the personas directory
    persona_files = [f for f in os.listdir(personas_dir) if f.endswith('.txt')]
    
    if not persona_files:
        raise FileNotFoundError("No persona files found in the prompts/personas directory")
    
    # Randomly select a persona file
    selected_file = random.choice(persona_files)
    
    # Read and return the contents
    with open(os.path.join(personas_dir, selected_file), 'r') as f:
        return f.read().strip()

def load_prompts():
    """Load the required system prompts for NFT generation."""
    required_prompts = ['prompt', 'summary']
    prompts = {}
    
    for prompt_name in required_prompts:
        try:
            prompts[prompt_name] = load_prompt(prompt_name)
            print(f"Loaded system prompt: {prompt_name}")
        except FileNotFoundError as e:
            print(f"Error: Required system prompt file '{prompt_name}' not found")
            raise
    return prompts

def load_prompt(filename: str) -> str:
    """
    Load a prompt file from the prompts directory.

    Args:
        filename (str): Name of the prompt file without .txt extension

    Returns:
        str: The contents of the prompt file
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, "prompts", f"{filename}.txt")
    
    try:
        with open(prompt_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompt file not found: {prompt_path}")
        raise


prompts = load_prompts()

class Summary(BaseModel):
    title: str
    summary: str


def generate_nft(conversation):
    # Generate a prompt from the conversation
    prompt_generator = Agent(
        name="Prompt Generator",
        model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
        description="You are an AI agent that looks at a conversation and generates a prompt for an NFT.",
        instructions=prompts['prompt'],
        markdown=True
    )

    summary_generator = Agent(
        name="Summary Generator",
        model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
        description="You are an AI agent that looks at a conversation and generates a summary of it and a title.",
        instructions=prompts['summary'],
        markdown=True
    )

    prompt = prompt_generator.run(message=conversation).content
    print(f"Generated Prompt: {prompt}")
    summary = summary_generator.run(message=conversation).content
    print(f"Generated Summary: {summary}")

    api = replicate.Client(api_token=os.environ["REPLICATE_API_KEY"])
    nft =  api.run(
        "ideogram-ai/ideogram-v2a",
        input={
        "aspect_ratio": "1:1",
        "magic_prompt_option": "Auto",
        "prompt":prompt,
        "resolution": "None",
        "style_type": "Render 3D"
        }
    )

    return nft

def create_agents():
    """Create two AI agents with their respective configurations."""
    # Get two random and different personas
    persona_one = get_random_persona()
    persona_two = get_random_persona()
    
    # Make sure we don't get the same persona twice
    while persona_two == persona_one:
        persona_two = get_random_persona()
    
    print("Selected personas for the conversation:")
    print(f"Agent 1's persona: {persona_one[:100]}...")  # Print first 100 chars for preview
    print(f"Agent 2's persona: {persona_two[:100]}...")

    agent_one = Agent(
        name="AI Agent 1",
        model=Claude(id="claude-3-5-sonnet-20241022", api_key=ANTHROPIC_API_KEY),
        instructions=persona_one,
        description="First AI agent in the conversation",
        add_state_in_messages=True,
        add_datetime_to_instructions=True
    )

    agent_two = Agent(
        name="AI Agent 2",
        model=Claude(id="claude-3-5-sonnet-20241022", api_key=ANTHROPIC_API_KEY),
        instructions=persona_two,
        description="Second AI agent in the conversation",
        add_state_in_messages=True,
        add_datetime_to_instructions=True
    )

    return agent_one, agent_two

def facilitate_conversation(agent_one: Agent, agent_two: Agent, num_turns: int = 5):
    """
    Facilitate a conversation between two AI agents.
    
    Args:
        agent_one: First AI agent
        agent_two: Second AI agent
        num_turns: Number of conversation turns (default: 5)
    """
    start_time = datetime.now()
    print(f"\nRelationship started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    current_message = "Hello! Let's start our conversation."
    conversation_history = [current_message]
    
    try:
        for turn in range(num_turns):
            print(f"\nTurn {turn + 1}:")
            
            # Agent One's turn
            print(f"\n{agent_one.name} says:")
            response = agent_one.run(message=current_message)
            current_message = response.content
            conversation_history.append(f"{agent_one.name}: {current_message}")
            print(current_message)
            
            # Agent Two's turn
            print(f"\n{agent_two.name} says:")
            response = agent_two.run(message=current_message)
            current_message = response.content
            conversation_history.append(f"{agent_two.name}: {current_message}")
            print(current_message)

        # Generate an NFT from the full conversation history
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\nRelationship ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Relationship duration: {duration}")
        
        full_conversation = "\n".join(conversation_history)
        nft = generate_nft(full_conversation)
        return nft, start_time, end_time, duration
            
    except Exception as e:
        print(f"Error during conversation: {str(e)}")
        return None, None, None, None

def main():
    # Create the agents
    agent_one, agent_two = create_agents()
    
    # Start the conversation between agents and get the generated NFT
    nft_url, start_time, end_time, duration = facilitate_conversation(agent_one, agent_two)
    
    if nft_url:
        print("\nRelationship Timeline:")
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration}")
        print("\nGenerated NFT URL:", nft_url)  # Just print the URL directly
    else:
        print("\nFailed to generate NFT")

if __name__ == "__main__":
    # Uncomment the line below to run the test instead of the main conversation
    # test_replicate()
    main()
