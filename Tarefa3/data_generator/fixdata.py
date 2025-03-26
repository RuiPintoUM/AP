import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the dataset
try:
    df = pd.read_csv(r'C:/Users/ruben/Desktop/Minho/MEI/SI/AP/TP Mod1/Tarefa3/data/small_dataset.csv')
    logger.info(f"Dataset loaded successfully with {len(df)} rows")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    raise

# Load the model and tokenizer
try:
    logger.info("Loading model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token explicitly
    logger.info("Model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model/tokenizer: {e}")
    raise

# Function to complete truncated text
def complete_text(text):
    try:
        encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        logger.debug(f"Generating completion for text: {text[:50]}...")
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + 20,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        completed_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        original_length = len(text)
        completion = completed_text[original_length:]
        first_period = completion.find('.')
        if first_period != -1:
            completion = completion[:first_period + 1]
        return text + completion
    except Exception as e:
        logger.error(f"Error completing text: {e}")
        return text  # Return original text on failure

# Fix truncated AI texts
processed_count = 0
for index, row in df.iterrows():
    try:
        if pd.notna(row['Label']) and row['Label'] == 'AI' and pd.notna(row['Text']) and not row['Text'].strip().endswith(('.', '!', '?')):
            logger.info(f"Fixing truncated text at index {index}: {row['Text'][:50]}...")
            completed_text = complete_text(row['Text'])
            df.at[index, 'Text'] = completed_text
            logger.info(f"Completed text: {completed_text[:50]}...")
            processed_count += 1
    except KeyError as e:
        logger.error(f"Missing column in row {index}: {e}")
    except Exception as e:
        logger.error(f"Error processing row {index}: {e}")

# Save the fixed dataset
try:
    output_path = r'C:/Users/ruben/Desktop/Minho/MEI/SI/AP/TP Mod1/Tarefa3/data/fixed_dataset_final.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset fixed and saved as '{output_path}' with {processed_count} texts completed")
except Exception as e:
    logger.error(f"Failed to save dataset: {e}")
    raise