import os
from openai import OpenAI
from openai import RateLimitError
import transformers
import torch
import time
import random
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import logging
import argparse
from accelerate import Accelerator
from torch.cuda.amp import GradScaler, autocast

# Initialize OpenAI client
client = OpenAI(
    api_key="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",
)

hf_token = "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"
model_id = "meta-llama/Meta-Llama-3-8B"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the accelerator
accelerator = Accelerator()
device = accelerator.device

# Load the pre-trained transformer model
def load_model():
    try:
        logger.info("Downloading and loading the model and tokenizer")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=hf_token)
        
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, 
            token=hf_token, 
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "12GB"}
        )
        
        tokenizer.pad_token = tokenizer.eos_token
        model.gradient_checkpointing_enable()  # Enable gradient checkpointing
        for param in model.parameters():
            param.requires_grad = True  # Ensure gradients are enabled
        
        model = accelerator.prepare(model)
        
        logger.info("Model and tokenizer loaded successfully")
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

model, tokenizer = load_model()

# Check model is on GPU
logger.info(f"Model device: {next(model.parameters()).device}")

# Global variable for the save path
SAVE_PATH = 'model.pth'

# Load the dataset from Hugging Face
def load_datasets():
    try:
        logger.info("Loading datasets")
        df_tr = pd.DataFrame(load_dataset("aladar/craigslist_bargains", split="train", cache_dir='./cache'))
        df_tr["text"] = df_tr["text"].fillna("")
        train_dialogues = df_tr["text"].tolist()

        df_te = pd.DataFrame(load_dataset("aladar/craigslist_bargains", split="test", cache_dir='./cache'))
        df_te["text"] = df_te["text"].fillna("")
        test_dialogues = df_te["text"].tolist()

        validation_split = int(0.1 * len(train_dialogues))
        valid_dialogues = train_dialogues[:validation_split]
        train_dialogues = train_dialogues[validation_split:]

        logger.info("Datasets loaded successfully")
        
        return train_dialogues, valid_dialogues, test_dialogues
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise

train_dialogues, valid_dialogues, test_dialogues = load_datasets()

def generate_advice(conversation):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert negotiator."},
                    {"role": "user", "content": f"Given the following negotiation conversation, provide advice for the seller on how to proceed.\n\nConversation:\n{conversation}\n\nAdvice:"}
                ],
                model="gpt-4o",
                max_tokens=150,
                temperature=0.0
            )
            advice = chat_completion.choices[0].message.content.strip()
            logger.info(f"Generated advice: {advice}")  # Log the advice
            return advice
        except RateLimitError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Rate limit exceeded. Retrying in {(2 ** attempt) + random.uniform(0, 1)} seconds...")
                time.sleep((2 ** attempt) + random.uniform(0, 1))
            else:
                logger.error("Rate limit exceeded. Maximum retry attempts reached.")
                raise e
        except Exception as e:
            logger.error(f"Error generating advice: {e}")
            return "Unable to generate advice. Please retry."

def rate_advice(conversation, advice):
    instructions = (
        """ "You are an expert negotiator meant to provide human feedback for Reinforcement Learning Human Feedback(RLHF) training. 
        Rate the quality of the following negotiation advice on a scale of 0 to 10, considering clarity, effectiveness, and professionalism.
        <Important>
        Format the feedback as 'Rating:X' without the commas or other punctuations. If no advice is provided, output 'Rating:0'.
        </Important>"""
        
    )
    max_retries = 5
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": f"Rate the quality of the following negotiation advice on a scale of 0 to 10, considering clarity, effectiveness, and professionalism.\n\nConversation:\n{conversation}\n\nAdvice: {advice}\n\nRating:"}
                ],
                model="gpt-4o",
                max_tokens=50,
                temperature=0.0
            )
            rating_response = chat_completion.choices[0].message.content.strip()
            logger.info(f"GPT-4o Response: {rating_response}")  # Log the raw response
            
            # Extract the rating value
            rating_value = None
            for line in rating_response.split('\n'):
                if "Rating:" in line:
                    try:
                        rating_value = float(line.split(":")[1].strip())
                        break
                    except ValueError:
                        continue
            
            if rating_value is None:
                return 0.0
            
            return rating_value
        except (SyntaxError, ValueError, KeyError) as e:
            logger.error(f"Error parsing rating: {e}, Response: {rating_response}")
            return 0.0  # Default to lowest rating if parsing fails
        except RateLimitError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Rate limit exceeded. Retrying in {(2 ** attempt) + random.uniform(0, 1)} seconds...")
                time.sleep((2 ** attempt) + random.uniform(0, 1))
            else:
                logger.error("Rate limit exceeded. Maximum retry attempts reached.")
                raise e

def generate_llama_advice(conversation, model, tokenizer):
    prompt = (
        f"You are an expert negotiator. Given the following negotiation conversation, provide concise advice for the seller on how to proceed. "
        f"Conversation:\n{conversation}\n\nAdvice:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    output = model.generate(**inputs, max_new_tokens=150, num_return_sequences=1, temperature=0.7, pad_token_id=tokenizer.pad_token_id)
    advice = tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, '').strip()
    logger.info(f"Generated LLaMA advice: {advice}")  # Log the advice
    return advice

def train_model(dialogues, model, tokenizer, epochs=3, learning_rate=5e-5, accumulation_steps=8):  # Adjusted accumulation steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    model, optimizer = accelerator.prepare(model, optimizer)
    model.train()

    logger.info("Starting training process")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for step, conversation in enumerate(tqdm(dialogues, desc=f"Training Epoch {epoch+1}/{epochs}")):
            try:
                with autocast():
                    loss = train_on_conversation(conversation, model, tokenizer)
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()
                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                epoch_loss += loss.item() * accumulation_steps
            except torch.cuda.OutOfMemoryError:
                logger.error("CUDA out of memory. Restarting training step.")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error in training: {e}")
        
        avg_epoch_loss = epoch_loss / (len(dialogues) * 3)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss}")
    
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), SAVE_PATH)
    logger.info(f"Model saved to {SAVE_PATH}")

def train_on_conversation(conversation, model, tokenizer):
    conversation_loss = 0.0
    for _ in range(3):
        advice = generate_llama_advice(conversation, model, tokenizer)
        rating = rate_advice(conversation, advice)
        
        reward = torch.tensor([rating], dtype=torch.float16).to(device)
        
        prompt = (
            f"You are an expert negotiator. Given the following negotiation conversation, provide succinct advice for the seller on how to proceed. "
            f"Conversation:\n{conversation}\n\nAdvice:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        labels = tokenizer(advice, return_tensors="pt", padding=True, truncation=True, max_length=inputs.input_ids.shape[-1]).input_ids.to(device)
        
        # Ensure the labels tensor has the same shape as the inputs tensor
        if labels.shape[1] < inputs.input_ids.shape[-1]:
            labels = torch.cat([labels, torch.full((labels.shape[0], inputs.input_ids.shape[-1] - labels.shape[1]), tokenizer.pad_token_id, dtype=torch.long, device=device)], dim=1)
        
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        if not torch.is_tensor(loss):
            raise ValueError("Model outputs must be a tensor or an iterable of Tensors")
        
        conversation_loss += loss * reward.item()
    
    return conversation_loss / 3

def validate_model(dialogues, model, tokenizer):
    total_rating = 0
    model.eval()
    with torch.no_grad():
        for conversation in tqdm(dialogues, desc="Validating"):
            advice = generate_llama_advice(conversation, model, tokenizer)
            rating = rate_advice(conversation, advice)
            total_rating += rating
    avg_rating = total_rating / len(dialogues)
    logger.info(f"Validation Average Rating: {avg_rating}")
    return avg_rating

def test_model(dialogues, model, tokenizer):
    model.eval()
    results = []
    with torch.no_grad():
        for conversation in tqdm(dialogues, desc="Testing"):
            advice = generate_llama_advice(conversation, model, tokenizer)
            results.append(advice)
            logger.info(f"Test Advice: {advice}")
    with open('test_results.txt', 'w') as f:
        for advice in results:
            f.write(advice + "\n")
    logger.info("Test results saved to test_results.txt")

def load_model_state(model, save_path=SAVE_PATH):
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
        load_model_state(model)
    
    if args.train:
        train_model(train_dialogues, model, tokenizer, epochs=args.epochs, learning_rate=args.lr)
    if args.validate:
        validate_model(valid_dialogues, model, tokenizer)
    if args.test:
        test_model(test_dialogues, model, tokenizer)

if __name__ == "__main__":
    main()
