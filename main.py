import streamlit as st
import json
import os

from utils import init_groq
from save_to_json import save_to_json
from extract_and_parse import extract_text_easyocr, parse_with_llm

# Streamlit UI
def main():
    st.set_page_config(page_title="Payment Receipt OCR", page_icon="💳", layout="wide")
    
    st.title("💳 OCReceipt : Payment Receipt OCR + LLM Parser")
    st.markdown("Upload payment screenshots → Extract text with OCR → Parse with AI")
    
    with st.sidebar:       
        
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
        
        if uploaded_file:
            if st.button("🔍 Extract & Parse", type="primary", width='stretch'):
                # Step 1: OCR
                with st.spinner("Step 1/2: Extracting text with OCR..."):
                    image_bytes = uploaded_file.read()
                    
                    ocr_text, ocr_error = extract_text_easyocr(image_bytes)
                    
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
