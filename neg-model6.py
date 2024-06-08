import os
import json
import torch
import random
import logging
import transformers
from openai import OpenAI, RateLimitError
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from accelerate import Accelerator
import time
import argparse
from torch.cuda.amp import GradScaler, autocast

# Initialize OpenAI client
client = OpenAI(api_key="")

hf_token = ""
model_id = "meta-llama/Llama-2-7b-chat-hf"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the accelerator
accelerator = Accelerator()
device = accelerator.device

SAVE_PATH = 'model.pth'
RESULTS_PATH = 'training_results.json'

# Set environment variable to manage memory fragmentation
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Load the pre-trained transformer model
def load_model():
    logger.info("Downloading and loading the model and tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=hf_token)
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, 
        token=hf_token, 
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: "20GB"}
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    model.gradient_checkpointing_enable()  # Enable gradient checkpointing
    for param in model.parameters():
        param.requires_grad = True  # Ensure gradients are enabled
    
    model = accelerator.prepare(model)
    
    logger.info("Model and tokenizer loaded successfully")
    
    return model, tokenizer

model, tokenizer = load_model()

# Check model is on GPU
logger.info(f"Model device: {next(model.parameters()).device}")

# Global variable for the save path
SAVE_PATH = 'model.pth'

# Load the dataset from Hugging Face
def load_datasets():
    logger.info("Loading datasets")
    df = pd.DataFrame(load_dataset("aladar/craigslist_bargains", split="train[:100]", cache_dir='./cache'))
    df["text"] = df["text"].fillna("")
    dialogues = df["text"].tolist()

    logger.info("Datasets loaded successfully")
    
    return dialogues

dialogues = load_datasets()


def rate_advice(conversation, advice):
    instructions = (
    """ You are an expert negotiator. You will be provided a negotiation conversation, as well as advice for that negotiation. Given the context, first rate the quality of the advice on a scale of 1 to 10 in four categories:
    1. How professional is the advice? (higher score = more professional)
    2. How specific is the advice to the conversation? (higher score = more specific)
    3. How effective does the advice seem? (higher score = more effective)
    4. How concise is the advice? (higher score = more concise)
    Lastly, output your holistic rating of the advice as the average of the four scores.
    <Important>
    Format your final rating as 'Rating:X' without the commas or other punctuations. If no advice is provided, output 'Rating:0'. 
    </Important>"""
    )
    max_retries = 5
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": f"Conversation:\n{conversation}\n\nAdvice: {advice}\n\nRating:"}
                ],
                model="gpt-4o",
                max_tokens=50,
                temperature=0.3
            )
            rating_response = chat_completion.choices[0].message.content.strip()
            logger.info(f"GPT Response: {rating_response}")  # Log the raw response
            
            # Extract the rating value
            rating_value = None
            for line in rating_response.split('\n'):
                if "Rating:" in line:
                    try:
                        rating_value = float(line.split(":")[1].strip())
                        logger.info(f"Rating value: {rating_value}")
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
        f"You are an expert sales negotiator. Given the following negotiation conversation, provide helpful advice to the seller on how to proceed. Explain your reasoning behind your suggestions."
        f"Conversation:\n{conversation}\n\nAdvice:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    output = model.generate(**inputs, max_new_tokens=150, num_return_sequences=1, temperature=0.7, pad_token_id=tokenizer.pad_token_id)
    advice = tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, '').strip()
    logger.info(f"Generated LLaMA advice: {advice}")  # Log the advice
    return advice


def train_model(dialogues, model, tokenizer, epochs=3, learning_rate=5e-5, accumulation_steps=8, checkpoint_interval=10):
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

                # Save checkpoint
                if (step + 1) % checkpoint_interval == 0:
                    save_checkpoint(model, epoch, step, optimizer, scaler)

            except torch.cuda.OutOfMemoryError:
                logger.error("CUDA out of memory. Restarting training step.")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error in training: {e}")
        
        avg_epoch_loss = epoch_loss / len(dialogues)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss}")
    
    accelerator.wait_for_everyone()
    save_checkpoint(model, epoch, step, optimizer, scaler)
    logger.info(f"Model saved to {SAVE_PATH}")

def train_on_conversation(conversation, model, tokenizer):
    conversation_loss = 0.0
    with open(RESULTS_PATH, 'a') as f:
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
            
            # Log results to file
            result = {
                "conversation": conversation,
                "advice": advice,
                "rating": rating
            }
            f.write(json.dumps(result) + "\n")
    
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


def save_checkpoint(model, epoch, step, optimizer, scaler):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),  # Corrected here
        'scaler_state_dict': scaler.state_dict(),
    }
    torch.save(checkpoint, f'checkpoint_epoch_{epoch}_step_{step}.pth')
    print(f"Checkpoint saved for epoch {epoch} step {step}")

def load_checkpoint(model, optimizer, scaler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch']
    start_step = checkpoint['step']
    logger.info(f"Loaded checkpoint from epoch {start_epoch}, step {start_step}")
    return start_epoch, start_step

def main():
    parser = argparse.ArgumentParser(description="Train, validate, and test LLaMA model for negotiation advice.")
    parser.add_argument('--train', action='store_true', help="Flag to train the model")
    parser.add_argument('--validate', action='store_true', help="Flag to validate the model")
    parser.add_argument('--test', action='store_true', help="Flag to test the model")
    parser.add_argument('--load', type=str, help="Path to the saved checkpoint")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    if args.load:
        start_epoch, start_step = load_checkpoint(model, optimizer, scaler, args.load)
    else:
        start_epoch, start_step = 0, 0

    if args.train:
        train_model(dialogues, model, tokenizer, epochs=args.epochs, learning_rate=args.lr)
    if args.validate:
        validate_model(dialogues, model, tokenizer)
    if args.test:
        test_model(dialogues, model, tokenizer)

if __name__ == "__main__":
    main()
