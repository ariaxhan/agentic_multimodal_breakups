"""
AI Agent Communication System

This module simulates a relationship between two AI agents who engage in conversation and 
generates a commemorative NFT of their interaction. The system uses multiple AI models:
- Claude 3.5 Sonnet for the main conversation agents
- Llama via Groq for generating NFT image prompts and summaries
- Ideogram via Replicate for NFT image generation

PRO TIP: ALWAYS SAVE YOUR INSTRUCTIONS IN SEPARATE FILES AND LOAD THEM IN.
THIS WILL SAVE YOU SO MUCH TIME AND PAIN AND EFFORT TRUST ME.
"""

import os
from dotenv import load_dotenv
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")

import random

import replicate
from agno.models.groq import Groq
from agno.agent import Agent
from agno.models.anthropic import Claude
from datetime import datetime

# NOTE: USE WEAVE. These are the only two line you need to add and it sets up
# A very nice web ui to track your agentic flows! Tip: do debug_mode=True for similar
# logs in your console.
# Initialize weave for potential future tracking/logging
import weave
weave.init("breakup")

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


def generate_nft(conversation):
    """
    Transforms a conversation into an NFT through a multi-step AI pipeline:
    1. Uses Llama to generate an artistic prompt based on the conversation
    2. Uses Llama to create a summary and title
    3. Feeds the generated prompt to Ideogram AI to create the final NFT image
    
    Args:
        conversation (str): The full conversation history between agents
        
    Returns:
        str: URL to the generated NFT image
    """
    try:
        # Create an agent specialized in converting conversations into artistic prompts
        prompt_generator = Agent(
            name="Prompt Generator",
            model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
            description="You are an AI agent that looks at a conversation and generates a prompt for an NFT.",
            instructions=prompts['prompt'],
            markdown=True
        )

        # Create an agent specialized in distilling conversations into concise summaries
        summary_generator = Agent(
            name="Summary Generator",
            model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
            description="You are an AI agent that looks at a conversation and generates a summary of it and a title.",
            instructions=prompts['summary'],
            markdown=True
        )

        # Generate a prompt and summary based on the conversation
        try:
            prompt_response = prompt_generator.run(message=conversation)
            if not prompt_response or not prompt_response.content:
                raise ValueError("Failed to generate prompt from conversation")
            prompt = prompt_response.content
            print(f"Generated Prompt: {prompt}")
            
            summary_response = summary_generator.run(message=conversation)
            if not summary_response or not summary_response.content:
                raise ValueError("Failed to generate summary from conversation")
            summary = summary_response.content
            print(f"Generated Summary: {summary}")
        except Exception as e:
            print(f"Error generating prompt or summary: {str(e)}")
            return None

        # Initialize Replicate client and generate NFT
        try:
            api = replicate.Client(api_token=os.environ["REPLICATE_API_KEY"])
            nft = api.run(
                "ideogram-ai/ideogram-v2a",
                input={
                    "aspect_ratio": "1:1",
                    "magic_prompt_option": "Auto",
                    "prompt": prompt,
                    "resolution": "None",
                    "style_type": "Render 3D"
                }
            )
            
            if not nft:
                raise ValueError("Failed to generate NFT image")
                
            return nft
            
        except Exception as e:
            print(f"Error generating NFT image: {str(e)}")
            return None
            
    except Exception as e:
        print(f"Unexpected error in generate_nft: {str(e)}")
        return None

def create_agents():
    """
    Creates two AI agents with distinct personalities for conversation.
    
    The agents use Claude 3.5 Sonnet and are initialized with:
    - Random, unique personas from predefined templates
    - Ability to maintain conversation state
    - Awareness of current time/context
    
    Returns:
        tuple: Two configured Agent instances ready for conversation
    """
    # Get two random and different personas
    persona_one = get_random_persona()
    persona_two = get_random_persona()
    
    # Make sure we don't get the same persona twice
    while persona_two == persona_one:
        persona_two = get_random_persona()
    
    print("Selected personas for the conversation:")
    print(f"Agent 1's persona: {persona_one[:100]}...")  # Print first 100 chars for preview
    print(f"Agent 2's persona: {persona_two[:100]}...")

    # Create the agents with their respective personas
    # NOTE: Click on the "Agent" keyword below to get a full list of all of Agno's
    # available configurations for their agents.
    # I would recommend taking the time to look through and play with the different options.
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
    Orchestrates a back-and-forth conversation between two AI agents.
    
    The conversation flow:
    1. Agents take turns responding to each other
    2. Each message builds on the context of previous messages
    3. The conversation is tracked for duration and content
    4. Finally generates an NFT capturing the essence of their interaction
    
    Args:
        agent_one (Agent): First conversational AI agent
        agent_two (Agent): Second conversational AI agent
        num_turns (int): Number of back-and-forth exchanges (default: 5)
        
    Returns:
        tuple: (NFT URL, start time, end time, conversation duration)
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
            if not response or not response.content:
                raise ValueError(f"Failed to get response from {agent_one.name}")
            current_message = response.content
            conversation_history.append(f"{agent_one.name}: {current_message}")
            print(current_message)
            
            # Agent Two's turn
            print(f"\n{agent_two.name} says:")
            response = agent_two.run(message=current_message)
            if not response or not response.content:
                raise ValueError(f"Failed to get response from {agent_two.name}")
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
        
        if nft is None:
            print("\nWarning: Failed to generate NFT, but conversation completed successfully")
            
        return nft, start_time, end_time, duration
            
    except Exception as e:
        print(f"Error during conversation: {str(e)}")
        end_time = datetime.now()
        duration = end_time - start_time
        return None, start_time, end_time, duration

def main():
    try:
        # Create the agents
        agent_one, agent_two = create_agents()
        
        # Start the conversation between agents and get the generated NFT
        nft_url, start_time, end_time, duration = facilitate_conversation(agent_one, agent_two)
        
        # Print relationship timeline
        if start_time and end_time:
            print("\nRelationship Timeline:")
            print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Duration: {duration}")
            
            if nft_url:
                print("\nGenerated NFT URL:", nft_url)
            else:
                print("\nNote: Conversation completed but NFT generation failed")
        else:
            print("\nError: Conversation failed to complete properly")
            
    except Exception as e:
        print(f"\nCritical error in main: {str(e)}")

if __name__ == "__main__":
    main()
