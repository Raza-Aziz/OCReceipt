import streamlit as st
import json
import os

from utils import init_groq
from save_to_json import save_to_json
from extract_and_parse import extract_text_easyocr, extract_text_easyocr_with_debug, parse_with_llm
from preprocess import convert_cv_to_pil

# Streamlit UI
def main():
    st.set_page_config(page_title="Payment Receipt OCR", page_icon="💳", layout="wide")
    
    st.title("💳 OCReceipt : Payment Receipt OCR + LLM Parser")
    st.markdown("Upload payment screenshots → Preprocess with CV2 → Extract text with OCR → Parse with AI")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Preprocessing options
        use_preprocessing = st.checkbox(
            "Enable Image Preprocessing", 
            value=True,
            help="Apply CV2 preprocessing for better OCR accuracy"
        )
        
        show_preprocessing_steps = st.checkbox(
            "Show Preprocessing Steps",
            value=False,
            help="Visualize intermediate preprocessing steps"
        )
        
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
        1. **Preprocess**: CV2 enhancement
           - Grayscale conversion
           - Noise reduction
           - Contrast enhancement
           - Adaptive thresholding
           - Deskewing
        2. **OCR**: Extracts raw text
        3. **LLM**: Parses with Pydantic
        4. **JSON**: Saves validated data
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
            st.image(uploaded_file, caption="Original Screenshot", width='stretch')
    
    with col2:
        st.subheader("📊 Processing Results")
        
        if uploaded_file:
            if st.button("🔍 Extract & Parse", type="primary", width='stretch'):
                image_bytes = uploaded_file.read()
                
                # Show preprocessing steps if enabled
                if use_preprocessing and show_preprocessing_steps:
                    with st.spinner("Preprocessing image..."):
                        _, steps, error = extract_text_easyocr_with_debug(image_bytes)
                        
                        if steps and not error:
                            st.success("✅ Preprocessing complete")
                            
                            # Display preprocessing steps
                            with st.expander("🔬 Preprocessing Steps", expanded=True):
                                step_names = list(steps.keys())
                                cols = st.columns(2)
                                
                                for idx, step_name in enumerate(step_names):
                                    col_idx = idx % 2
                                    with cols[col_idx]:
                                        st.caption(step_name.replace('_', ' ').title())
                                        st.image(
                                            convert_cv_to_pil(steps[step_name]), 
                                            width='stretch'
                                        )
                
                # Step 1: OCR
                with st.spinner("Step 1/2: Extracting text with OCR..."):
                    ocr_text, ocr_error = extract_text_easyocr(image_bytes, use_preprocessing=use_preprocessing)
                    
                    if ocr_error:
                        st.error(f"❌ {ocr_error}")
                        st.stop()
                    
                    st.success(f"✅ Text extracted with {'preprocessed' if use_preprocessing else 'raw'} OCR")
                    
                    # Show extracted text in expander
                    with st.expander("📝 View Raw OCR Text"):
                        st.text_area("Extracted Text", ocr_text, height=200, key="ocr_text")
                
                # Step 2: LLM Parsing
                with st.spinner("Step 2/2: Parsing with LLM (Structured Output)..."):
                    client = init_groq()
                    if client:
                        transaction_data, parse_error = parse_with_llm(client, ocr_text)
                        
                        if parse_error:
                            st.error(f"❌ {parse_error}")
                        else:
                            st.success("✅ Details parsed & validated with Pydantic!")
                            
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
                            st.text_area("Raw OCR Text", txn['raw_ocr_text'], height=300, key=f"history_ocr_{idx}")
        else:
            st.info("No transactions saved yet")


if __name__ == "__main__":
    main()