import os
from openai import OpenAI

# 1. Setup your free Gemini key and redirect the OpenAI client to Google's server
GEMINI_API_KEY = "AIzaSyDVwcTGaH21jrW8zwTQP1YYFSfqKdrOsdA"

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def chat(messages):
    """
    Send messages to Gemini using OpenAI syntax and get response
    """
    response = client.chat.completions.create(
        model='gemini-2.5-flash',  # Free, fast, and works over the base_url proxy
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content


# 2. Initialize with your custom TechShop Customer Support prompt
support_system_prompt = ''' You are a customer support agent for TechShop, an electronics retailer.
Policies:
- Returns: 30-day return policy
- Shipping: Free over $50, otherwise $5.99
- Warranty: 1 year manufacturer warranty
- Support hours: 9 AM - 6 PM EST, Mon-Fri

Your tone:
- Empathetic and patient
- Solution-focused
- Apologize when appropriate
- Offer to escalate complex issues '''

messages = [
    {'role': 'system', 'content': support_system_prompt}
]

# Conversation loop
print('TechShop Customer Support Bot Ready! Type "quit" to exit.\n')

while True:
    # Get user input
    user_input = input('You: ')

    if user_input.lower() == 'quit':
        print('Goodbye!')
        break

    # Add user message to history
    messages.append({'role': 'user', 'content': user_input})

    # Get response from the model
    try:
        assistant_response = chat(messages)

        # Add assistant reply to history
        messages.append({'role': 'assistant', 'content': assistant_response})

        print(f'Support Agent: {assistant_response}\n')

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please double check your API key and internet connection.\n")
        break