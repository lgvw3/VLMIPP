# Function to get VLM recommendations
import base64
import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CONVERSATION = None
MODEL = "gpt-4.1-mini"
SLEEP_REQUIRED = False
SLEEP_TIME = 2


def get_vlm_response(mode, NUM_AGENTS, GRID_SIZE, state_json, image_path):
    if mode == "conversation":
        return get_vlm_moves_with_conversation(NUM_AGENTS, GRID_SIZE, state_json, image_path)
    else:
        return get_vlm_moves(NUM_AGENTS, GRID_SIZE, state_json, image_path)
    


def get_prompt(NUM_AGENTS, GRID_SIZE, state_json):
    return f"""
    You are coordinating {NUM_AGENTS} robots exploring an unknown environment for INFORMATIVE PATH PLANNING.

    IMAGE INTERPRETATION:
    - The image shows the robots' CURRENT BELIEF about rewards (not ground truth)
    - As the bots traverse the environment they will gain more information about the environment and reduce their uncertainty,
    - The whole variance map is not updated by the robots, only that which they have observed.
    - Use the color scale in the ledgend on the right to understand the uncertainty level of the cells.
    - The color scale is the matplotlib cmap='hot' scale with vmin=0 and vmax=1. Meaning the lighter white / yellow colors are highly uncertain
    - The darker reds and blacks are low uncertainty. 
    - RED dots with labels (i.e. A0, A1, A2) = current robot positions
    - Grid size: {GRID_SIZE}x{GRID_SIZE} locations

    CURRENT BELIEF STATE:
    - Time step: {state_json['time']}
    - Total uncertainty (entropy): {state_json['total_entropy']:.4f}
    - High-uncertainty cells: {len(state_json['high_entropy_cells'])} locations
    - Active agents: {sum(1 for agent in state_json['agents'] if agent['active'])} robots

    AGENT STATUS:
    {chr(10).join([f"- Agent {agent['id']}: Position ({agent['pos'][0]}, {agent['pos'][1]}), Budget: {agent['budget']}, Active: {agent['active']}" for agent in state_json['agents']])}

    COORDINATION STRATEGY:
    1. PRIORITIZE HIGH UNCERTAINTY AREAS in the MAP
    2. Avoid sending multiple agents to the same high-uncertainty region
    3. Consider agent budgets and current positions
    4. The goal is to reduce uncertainty, not find high rewards

    STAY NOTES:

    Stay does not reduce uncertainty. It is only used to maintain the current position in order to preserve
    the agents budget constraints.

    IMPORTANT: You are working with BELIEFS, not ground truth. Target areas of high uncertainty to improve the robots' understanding of the environment.

    AVAILABLE MOVES: stay, up, down, left, right

    TASK: Analyze the belief map image and recommend moves for each active agent. Return ONLY a JSON dictionary like:
    {{"0": "right", "1": "up", "2": "stay"}}

    Return ONLY the JSON response with no additional text.
    """
    

def get_vlm_moves(NUM_AGENTS, GRID_SIZE, state_json, image_path):
    if SLEEP_REQUIRED:
        time.sleep(SLEEP_TIME)
    # Load image as base64
    with open(image_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Prompt for VLM
    prompt = get_prompt(NUM_AGENTS, GRID_SIZE, state_json)

    chat = client.responses.create(
        model=MODEL,
        instructions="You are a helpful VLM for agent coordination.",
        input=[
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": prompt },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{img_base64}",
                    },
                ],
            }
        ],
    )

    response = chat.output_text
    print('response', response)
    try:
        return json.loads(response)
    except Exception as e:
        print(f"VLM response error: {e}")
        return {}
    


def get_vlm_moves_with_conversation(NUM_AGENTS, GRID_SIZE, state_json, image_path):
    global CONVERSATION
    if CONVERSATION is None:
        CONVERSATION = client.conversations.create()

    if SLEEP_REQUIRED:
        time.sleep(SLEEP_TIME)

    # Load image as base64
    with open(image_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    
    prompt = get_prompt(NUM_AGENTS, GRID_SIZE, state_json)
    chat = client.responses.create(
        model=MODEL,
        input=[{
            "role": "user", 
            "content": [
                { "type": "input_text", "text": prompt },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{img_base64}",
                },
            ]
        }],
        conversation=CONVERSATION.id
    )
    response = chat.output_text
    print('response', response)
    try:
        return json.loads(response)
    except Exception as e:
        print(f"VLM response error: {e}")
        return {}