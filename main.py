import streamlit as st
import json
import os
from datetime import datetime
from groq import Groq
from PIL import Image
import io
from dotenv import load_dotenv
import numpy as np

# Alternative: Use EasyOCR (uncomment if preferred)
import easyocr

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize Groq client
def init_groq():
    api_key = GROQ_API_KEY
    if api_key:
        return Groq(api_key=api_key)
    return None

# Extract text using Tesseract OCR
def extract_text_tesseract(image):
    try:
        # Convert to PIL Image if needed
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        
        # Perform OCR
        # text = pytesseract.image_to_string(image)
        # return text.strip(), None
    except Exception as e:
        return None, f"OCR Error: {str(e)}"

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

# Save to JSON file
def save_to_json(data, filename="transactions.json"):
    try:
        # Load existing data if file exists
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
        
        # Append new data
        existing_data.append(data)
        
        # Save back to file
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        return True, filename
    except Exception as e:
        return False, str(e)

# Streamlit UI
def main():
    st.set_page_config(page_title="Payment Receipt OCR", page_icon="💳", layout="wide")
    
    st.title("💳 Payment Receipt OCR + LLM Parser")
    st.markdown("Upload payment screenshots → Extract text with OCR → Parse with AI")
    
    # API Key input in sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=st.session_state.get('groq_api_key', ''),
            help="Enter your Groq API key. Get one at https://console.groq.com"
        )
        
        if api_key:
            st.session_state.groq_api_key = api_key
            st.success("✓ API Key configured")
        else:
            st.warning("⚠️ Please enter your Groq API key")
        
        st.markdown("---")
        
        # OCR Engine selection
        ocr_engine = st.radio(
            "OCR Engine",
            ["Tesseract", "EasyOCR"],
            help="Choose your preferred OCR engine"
        )
        st.session_state.ocr_engine = ocr_engine
        
        st.markdown("---")
        st.markdown("### 📋 Supported Apps")
        st.markdown("""
        - EasyPaisa
        - JazzCash
        - Bank Transfers
        - Mobile Wallets
        - Any payment app!
        """)
        
        st.markdown("---")
        output_file = st.text_input("Output JSON filename", value="transactions.json")
        
        st.markdown("---")
        st.markdown("### 💡 How it works")
        st.markdown("""
        1. **OCR**: Extracts raw text from image
        2. **LLM**: Parses and structures the text
        3. **JSON**: Saves organized data
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Screenshot")
        uploaded_file = st.file_uploader(
            "Choose a payment receipt screenshot",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear screenshot of your payment receipt"
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Screenshot", width='stretch')
    
    with col2:
        st.subheader("📊 Processing Results")
        
        if uploaded_file and api_key:
            if st.button("🔍 Extract & Parse", type="primary", width='stretch'):
                # Step 1: OCR
                with st.spinner("Step 1/2: Extracting text with OCR..."):
                    image_bytes = uploaded_file.read()
                    
                    # Choose OCR engine
                    if st.session_state.get('ocr_engine') == "EasyOCR":
                        ocr_text, ocr_error = extract_text_easyocr(image_bytes)
                    else:
                        ocr_text, ocr_error = extract_text_tesseract(image_bytes)
                    
                    if ocr_error:
                        st.error(f"❌ {ocr_error}")
                        st.stop()
                    
                    st.success("✅ Text extracted with OCR")
                    
                    # Show extracted text in expander
                    with st.expander("📝 View Raw OCR Text"):
                        st.text_area("Extracted Text", ocr_text, height=200)
                
                # Step 2: LLM Parsing
                with st.spinner("Step 2/2: Parsing with LLM..."):
                    client = init_groq()
                    if client:
                        transaction_data, parse_error = parse_with_llm(client, ocr_text)
                        
                        if parse_error:
                            st.error(f"❌ {parse_error}")
                        else:
                            st.success("✅ Details parsed successfully!")
                            
                            # Display extracted data
                            st.json(transaction_data)
                            
                            # Save to JSON
                            success, message = save_to_json(transaction_data, output_file)
                            if success:
                                st.success(f"💾 Saved to {message}")
                                
                                # Download button
                                with open(output_file, 'r') as f:
                                    st.download_button(
                                        label="⬇️ Download JSON File",
                                        data=f.read(),
                                        file_name=output_file,
                                        mime="application/json",
                                        width='stretch'
                                    )
                            else:
                                st.error(f"Failed to save: {message}")
                    else:
                        st.error("Failed to initialize Groq client")
        elif not api_key:
            st.info("👈 Please enter your Groq API key in the sidebar")
        else:
            st.info("👆 Upload a screenshot to begin extraction")
    
    # Show existing transactions
    if os.path.exists(output_file):
        st.markdown("---")
        st.subheader("📜 Transaction History")
        
        with open(output_file, 'r') as f:
            transactions = json.load(f)
        
        if transactions:
            st.write(f"Total transactions: {len(transactions)}")
            
            # Display summary table
            summary_data = []
            for txn in transactions:
                summary_data.append({
                    "Status": txn.get('transaction_status', 'N/A'),
                    "Amount": f"{txn.get('currency', '')} {txn.get('amount', 'N/A')}",
                    "From": txn.get('sender_name', 'N/A'),
                    "To": txn.get('recipient_name', 'N/A'),
                    "Date": txn.get('transaction_date', 'N/A'),
                    "Method": txn.get('payment_method', 'N/A')
                })
            
            st.dataframe(summary_data, width='stretch')
            
            # Display in expandable sections
            st.markdown("### Detailed View")
            for idx, txn in enumerate(reversed(transactions), 1):
                with st.expander(f"Transaction #{len(transactions) - idx + 1} - {txn.get('amount', 'N/A')} {txn.get('currency', '')}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.json({k: v for k, v in txn.items() if k != 'raw_ocr_text'})
                    with col_b:
                        if 'raw_ocr_text' in txn:
                            st.text_area("Raw OCR Text", txn['raw_ocr_text'], height=300)
        else:
            st.info("No transactions saved yet")

if __name__ == "__main__":
    main()
