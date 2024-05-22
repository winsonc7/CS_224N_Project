import openai
import transformers
import torch
import os
import time
import random
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set your OpenAI API key
openai.api_key = 'your_openai_api_key'

# Set your Hugging Face API token
hf_token = 'your_huggingface_token'

# Load the pre-trained transformer model with authentication
model_id = "meta-llama/Meta-Llama-3-8B"
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, use_auth_token=hf_token)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)

# Set pad token ID
model.config.pad_token_id = model.config.eos_token_id

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Global variable for the save path
SAVE_PATH = 'model.pth'

# Load the dataset from Hugging Face
df_tr = pd.DataFrame(load_dataset("aladar/craigslist_bargains", split="train"))
df_tr["text"] = df_tr["text"].fillna("")
train_dialogues = df_tr["text"].tolist()

df_te = pd.DataFrame(load_dataset("aladar/craigslist_bargains", split="test"))
df_te["text"] = df_te["text"].fillna("")
test_dialogues = df_te["text"].tolist()

# We will split the training data to create a validation set
validation_split = int(0.1 * len(train_dialogues))
valid_dialogues = train_dialogues[:validation_split]
train_dialogues = train_dialogues[validation_split:]

def generate_advice(conversation):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = openai.Completion.create(
                engine="gpt-4o",
                prompt=(
                    f"Given the following negotiation conversation, provide advice for the seller on how to proceed. "
                    f"Return the advice in the following JSON format: {{\"advice\": \"<your_advice_here>\"}}\n\n"
                    f"Conversation:\n{conversation}\n\nAdvice:"
                ),
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0.7
            )
            advice = response.choices[0].text.strip()
            advice_json = eval(advice)  # Convert string to dictionary
            return advice_json["advice"]
        except (SyntaxError, ValueError, KeyError):
            return "Unable to generate advice. Please retry."
        except openai.error.RateLimitError:
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) + random.uniform(0, 1))
            else:
                raise

def rate_advice(advice):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = openai.Completion.create(
                engine="gpt-4o",
                prompt=(
                    f"Rate the quality of the following negotiation advice on a scale of 1 to 10, considering clarity, effectiveness, and professionalism. "
                    f"Return the rating in the following JSON format: {{\"rating\": <rating_number>}}\n\n"
                    f"Advice: {advice}\n\nRating:"
                ),
                max_tokens=10,
                n=1,
                stop=None,
                temperature=0.7
            )
            rating = response.choices[0].text.strip()
            rating_json = eval(rating)  # Convert string to dictionary
            return rating_json["rating"]
        except (SyntaxError, ValueError, KeyError):
            return 0.0  # Default to lowest rating if parsing fails
        except openai.error.RateLimitError:
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) + random.uniform(0, 1))
            else:
                raise

def generate_llama_advice(conversation, model, tokenizer):
    prompt = (
        f"You are an expert negotiator. Given the following negotiation conversation, provide advice for the seller on how to proceed. "
        f"Conversation:\n{conversation}\n\nAdvice:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    output = model.generate(**inputs, max_length=150, num_return_sequences=1, temperature=0.7, pad_token_id=model.config.eos_token_id)
    advice = tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, '').strip()
    return advice

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(dialogues, model, tokenizer, epochs=3, learning_rate=5e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()  # Set model to training mode
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_conversation = {executor.submit(train_on_conversation, conversation, model, tokenizer, optimizer): conversation for conversation in dialogues}
            for future in tqdm(as_completed(future_to_conversation), total=len(dialogues), desc=f"Training Epoch {epoch+1}/{epochs}"):
                loss = future.result()
                epoch_loss += loss
        
        avg_epoch_loss = epoch_loss / (len(dialogues) * 3)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss}")
    
    # Save the model
    torch.save(model.state_dict(), SAVE_PATH)
    logger.info(f"Model saved to {SAVE_PATH}")

def train_on_conversation(conversation, model, tokenizer, optimizer):
    conversation_loss = 0.0
    for _ in range(3):  # Generate 3 pieces of advice
        advice = generate_llama_advice(conversation, model, tokenizer)
        rating = rate_advice(advice)
        
        # Reward is the rating from GPT-4O
        reward = torch.tensor([rating], dtype=torch.float32).to(device)
        
        # Generate logits for the prompt
        prompt = (
            f"You are an expert negotiator. Given the following negotiation conversation, provide advice for the seller on how to proceed. "
            f"Conversation:\n{conversation}\n\nAdvice:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        labels = tokenizer(advice, return_tensors="pt").input_ids.to(device)
        
        # Get model outputs and calculate loss
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        conversation_loss += loss.item()
    
    return conversation_loss

def validate_model(dialogues, model, tokenizer):
    total_rating = 0
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for conversation in tqdm(dialogues, desc="Validating"):
            advice = generate_llama_advice(conversation, model, tokenizer)
            rating = rate_advice(advice)
            total_rating += rating
    avg_rating = total_rating / len(dialogues)
    logger.info(f"Validation Average Rating: {avg_rating}")
    return avg_rating

def test_model(dialogues, model, tokenizer):
    model.eval()  # Set model to evaluation mode
    results = []
    with torch.no_grad():
        for conversation in tqdm(dialogues, desc="Testing"):
            advice = generate_llama_advice(conversation, model, tokenizer)
            results.append(advice)
            logger.info(f"Test Advice: {advice}")
    # Save test results
    with open('test_results.txt', 'w') as f:
        for advice in results:
            f.write(advice + "\n")
    logger.info("Test results saved to test_results.txt")

def load_model(model, save_path=SAVE_PATH):
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.to(device)
        logger.info(f"Model loaded from {save_path}")
    else:
        logger.warning(f"No model found at {save_path}. Training from scratch.")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train, validate, and test LLaMA model for negotiation advice.")
    parser.add_argument('--train', action='store_true', help="Flag to train the model")
    parser.add_argument('--validate', action='store_true', help="Flag to validate the model")
    parser.add_argument('--test', action='store_true', help="Flag to test the model")
    parser.add_argument('--load', action='store_true', help="Flag to load the saved model")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()

    if args.load:
        load_model(model)
    
    if args.train:
        train_model(train_dialogues, model, tokenizer, epochs=args.epochs, learning_rate=args.lr)
    if args.validate:
        validate_model(valid_dialogues, model, tokenizer)
    if args.test:
        test_model(test_dialogues, model, tokenizer)

if __name__ == "__main__":
    main()
