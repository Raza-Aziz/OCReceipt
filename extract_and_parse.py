from PIL import Image
import easyocr
import json
import io
from datetime import datetime

def extract_text_easyocr(image_bytes):
    try:
        import numpy as np
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

# Parse extracted text using Groq LLM
def parse_with_llm(client, ocr_text):
    prompt = f"""You are a transaction receipt parser. Below is the raw text extracted from a payment receipt screenshot using OCR.

RAW OCR TEXT:
{ocr_text}

Analyze this text and extract transaction information in a structured JSON format.

Extract the following fields (use null if not found):
- transaction_status: (e.g., "Successful", "Failed", "Pending")
- sender_name: Full name of sender
- sender_account: Account number or ID of sender
- recipient_name: Full name of recipient
- recipient_account: Account number or ID of recipient
- amount: Numeric amount transferred (just the number)
- currency: Currency symbol or code (e.g., "Rs", "PKR", "USD")
- transaction_date: Date of transaction
- transaction_time: Time of transaction
- transaction_id: Transaction ID or reference number (like TID)
- payment_method: Method/platform used (e.g., "EasyPaisa", "JazzCash", "Bank Transfer")
- bank_name: Bank name if applicable
- notes: Any additional notes or messages
- fee: Transaction fee if mentioned
- any_other_details: Any other relevant information

Return ONLY a valid JSON object with these fields. Do not include any explanation or markdown formatting."""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        
        # Clean up response
        result = result.strip()
        if result.startswith('```json'):
            result = result[7:]
        if result.startswith('```'):
            result = result[3:]
        if result.endswith('```'):
            result = result[:-3]
        result = result.strip()
        
        transaction_data = json.loads(result)
        transaction_data['extraction_timestamp'] = datetime.now().isoformat()
        transaction_data['raw_ocr_text'] = ocr_text
        
        return transaction_data, None
        
    except json.JSONDecodeError as e:
        return None, f"Failed to parse JSON response: {str(e)}\nRaw response: {result}"
    except Exception as e:
        return None, f"Error parsing with LLM: {str(e)}"