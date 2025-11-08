from datetime import datetime
from PIL import Image
import numpy as np
import easyocr
import json
import io

from preprocess import preprocess_image

def extract_text_easyocr(image_bytes):
    try:
        reader = easyocr.Reader(['en'])
        
        # Convert bytes → numpy array (RGB)
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        
        # Perform OCR
        results = reader.readtext(image)
        
        # Combine all detected text
        text = '\n'.join([result[1] for result in results])
        return text.strip(), None
    except Exception as e:
        return None, f"OCR Error: {str(e)}"

# Initialize EasyOCR reader (cached to avoid reloading)
_reader = None

def get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader

def extract_text_easyocr(image_bytes, use_preprocessing=True):
    """
    Extract text from image using EasyOCR with optional preprocessing.
    
    Args:
        image_bytes: Image as bytes
        use_preprocessing: Whether to apply CV2 preprocessing
    
    Returns:
        text: Extracted text
        error: Error message if any
    """
    try:
        reader = get_reader()

        # Convert bytes → numpy array (RGB)
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        
        # Perform OCR
        results = reader.readtext(image)
        
        # Combine all detected text
        text = '\n'.join([result[1] for result in results])
        
        return text.strip(), None
    
    except Exception as e:
        return None, f"OCR Error: {str(e)}"

def extract_text_easyocr_with_debug(image_bytes):
    """
    Extract text with preprocessing debug info.
    Returns both text and preprocessing steps.
    """
    try:
        reader = get_reader()
        
        # Apply preprocessing with debug
        preprocessed_image, steps = preprocess_image(image_bytes, debug=True)
        
        # Convert bytes → numpy array (RGB)
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        
        # Perform OCR
        results = reader.readtext(image)
        
        # Combine all detected text
        text = '\n'.join([result[1] for result in results])
        
        return text.strip(), steps, None

    except Exception as e:
        return None, None, f"OCR Error: {str(e)}"

from pydantic import BaseModel, Field
from typing import Optional

# Pydantic model for transaction data
class TransactionData(BaseModel):
    transaction_status: Optional[str] = Field(None, description="Status of transaction (e.g., 'Successful', 'Failed', 'Pending')")
    sender_name: Optional[str] = Field(None, description="Full name of sender")
    sender_account: Optional[str] = Field(None, description="Account number or ID of sender")
    recipient_name: Optional[str] = Field(None, description="Full name of recipient")
    recipient_account: Optional[str] = Field(None, description="Account number or ID of recipient")
    amount: Optional[str] = Field(None, description="Numeric amount transferred")
    currency: Optional[str] = Field(None, description="Currency symbol or code (e.g., 'Rs', 'PKR', 'USD')")
    transaction_date: Optional[str] = Field(None, description="Date of transaction")
    transaction_time: Optional[str] = Field(None, description="Time of transaction")
    transaction_id: Optional[str] = Field(None, description="Transaction ID or reference number")
    payment_method: Optional[str] = Field(None, description="Method/platform used (e.g., 'EasyPaisa', 'JazzCash')")
    bank_name: Optional[str] = Field(None, description="Bank name if applicable")
    notes: Optional[str] = Field(None, description="Any additional notes or messages")
    fee: Optional[str] = Field(None, description="Transaction fee if mentioned")
    any_other_details: Optional[str] = Field(None, description="Any other relevant information")

    
def parse_with_llm(client, ocr_text):
    """
    Parse extracted text using Groq LLM with structured output.
    """
    try:
        # Create the prompt for structured extraction
        prompt = f"""You are a transaction receipt parser. Below is the raw text extracted from a payment receipt screenshot using OCR.

# RAW OCR TEXT:
# {ocr_text}

# Analyze this text and extract transaction information in a structured JSON format.

# Extract the following fields (use null if not found):
# - transaction_status: (e.g., "Successful", "Failed", "Pending")
# - sender_name: Full name of sender
# - sender_account: Account number or ID of sender
# - recipient_name: Full name of recipient
# - recipient_account: Account number or ID of recipient
# - amount: Numeric amount transferred (just the number)
# - currency: Currency symbol or code (e.g., "Rs", "PKR", "USD")
# - transaction_date: Date of transaction
# - transaction_time: Time of transaction
# - transaction_id: Transaction ID or reference number (like TID)
# - payment_method: Method/platform used (e.g., "EasyPaisa", "JazzCash", "Bank Transfer")
# - bank_name: Bank name if applicable
# - notes: Any additional notes or messages
# - fee: Transaction fee if mentioned
# - any_other_details: Any other relevant information

# Return ONLY a valid JSON object with these fields. Do not include any explanation or markdown formatting."""

        # Use Groq with structured output
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a transaction receipt parser. Extract structured information from payment receipts accurately."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        
        result = completion.choices[0].message.content
        transaction_data = json.loads(result)
        
        for key in ["amount", "fee"]:
            if key in transaction_data and transaction_data[key] is not None:
                transaction_data[key] = str(transaction_data[key])
        
        # Validate with Pydantic model
        validated_data = TransactionData(**transaction_data)
        
        # Convert to dict and add metadata
        
        final_data = validated_data.model_dump()
        final_data['extraction_timestamp'] = datetime.now().isoformat()
        final_data['raw_ocr_text'] = ocr_text
        
        return final_data, None
        
    except json.JSONDecodeError as e:
        return None, f"Failed to parse JSON response: {str(e)}"
    except Exception as e:
        return None, f"Error parsing with LLM: {str(e)}"