import streamlit as st
import os
import tempfile
import zipfile
import io
from PIL import Image
import pandas as pd
from figure_extractor import PDFFigureExtractor
from figure_classifier import FigureClassifier
from utils import create_download_link, get_file_size

# Initialize session state
if 'extracted_figures' not in st.session_state:
    st.session_state.extracted_figures = []
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

def main():
    st.set_page_config(
        page_title="PDF Figure Extraction & Classification Tool",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 PDF Figure Extraction & Classification Tool")
    st.markdown("Upload a PDF document to automatically extract and classify all figures within it.")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to extract and classify figures"
        )
        
        if uploaded_file is not None:
            file_size = get_file_size(uploaded_file)
            st.info(f"File size: {file_size}")
            
            if st.button("Process PDF", type="primary"):
                process_pdf(uploaded_file)
    
    # Main content area
    if st.session_state.processing_complete and st.session_state.extracted_figures:
        display_results()
    elif uploaded_file is None:
        display_welcome_screen()
    else:
        st.info("Upload a PDF file and click 'Process PDF' to get started.")

def process_pdf(uploaded_file):
    """Process the uploaded PDF file and extract/classify figures."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Initialize components
        extractor = PDFFigureExtractor()
        classifier = FigureClassifier()
        
        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Extract figures
        status_text.text("Extracting figures from PDF...")
        progress_bar.progress(25)
        
        extracted_figures = extractor.extract_figures(tmp_file_path)
        
        if not extracted_figures:
            st.warning("No figures found in the PDF document.")
            os.unlink(tmp_file_path)
            return
        
        progress_bar.progress(50)
        status_text.text(f"Found {len(extracted_figures)} figures. Classifying...")
        
        # Classify figures
        classification_results = []
        for i, figure_data in enumerate(extracted_figures):
            classification = classifier.classify_figure(figure_data['image'])
            classification_results.append({
                'figure_id': i,
                'classification': classification,
                'confidence': classifier.get_confidence(),
                'page': figure_data['page'],
                'bbox': figure_data['bbox']
            })
            
            # Update progress
            progress = 50 + (i + 1) * 40 / len(extracted_figures)
            progress_bar.progress(int(progress))
        
        progress_bar.progress(100)
        status_text.text("Processing complete!")
        
        # Store results in session state
        st.session_state.extracted_figures = extracted_figures
        st.session_state.classification_results = classification_results
        st.session_state.processing_complete = True
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"Successfully extracted and classified {len(extracted_figures)} figures!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        if 'tmp_file_path' in locals():
            os.unlink(tmp_file_path)

def display_welcome_screen():
    """Display welcome screen with instructions."""
    st.markdown("""
    ## Welcome to the PDF Figure Extraction Tool
    
    This tool helps you:
    - 📄 Upload PDF documents
    - 🖼️ Extract all figures and images
    - 🔍 Classify figure types automatically
    - 📊 View comprehensive analysis
    - 💾 Download individual figures or all as ZIP
    
    ### Supported Figure Types:
    - **Charts**: Bar charts, pie charts, line graphs, scatter plots
    - **Diagrams**: Flowcharts, scientific diagrams, technical drawings
    - **Images**: Photographs, illustrations, screenshots
    - **Tables**: Data tables, comparison charts
    - **Maps**: Geographic maps, floor plans
    
    ### How to Use:
    1. Click "Choose a PDF file" in the sidebar
    2. Select your PDF document
    3. Click "Process PDF" to start extraction
    4. View results with classifications and statistics
    5. Download individual figures or all as ZIP
    
    Get started by uploading a PDF file!
    """)

def display_results():
    """Display the extracted figures and classification results."""
    figures = st.session_state.extracted_figures
    classifications = st.session_state.classification_results
    
    # Statistics summary
    st.header("📈 Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Figures", len(figures))
    with col2:
        unique_types = len(set(c['classification'] for c in classifications))
        st.metric("Figure Types", unique_types)
    with col3:
        avg_confidence = sum(c['confidence'] for c in classifications) / len(classifications)
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Figure type distribution
    st.subheader("Figure Type Distribution")
    type_counts = {}
    for classification in classifications:
        fig_type = classification['classification']
        type_counts[fig_type] = type_counts.get(fig_type, 0) + 1
    
    # Create distribution chart
    df_types = pd.DataFrame(list(type_counts.items()), columns=['Type', 'Count'])
    st.bar_chart(df_types.set_index('Type'))
    
    # Download all figures button
    st.subheader("Download Options")
    if st.button("📦 Download All Figures as ZIP"):
        zip_buffer = create_zip_download(figures, classifications)
        st.download_button(
            label="Download ZIP",
            data=zip_buffer,
            file_name="extracted_figures.zip",
            mime="application/zip"
        )
    
    # Display individual figures
    st.header("🖼️ Extracted Figures")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        filter_type = st.selectbox(
            "Filter by type:",
            ['All'] + sorted(list(type_counts.keys()))
        )
    with col2:
        sort_by = st.selectbox(
            "Sort by:",
            ['Page Number', 'Confidence', 'Figure Type']
        )
    
    # Filter and sort figures
    filtered_results = classifications
    if filter_type != 'All':
        filtered_results = [c for c in classifications if c['classification'] == filter_type]
    
    if sort_by == 'Page Number':
        filtered_results.sort(key=lambda x: x['page'])
    elif sort_by == 'Confidence':
        filtered_results.sort(key=lambda x: x['confidence'], reverse=True)
    elif sort_by == 'Figure Type':
        filtered_results.sort(key=lambda x: x['classification'])
    
    # Display figures in grid
    cols_per_row = 2
    for i in range(0, len(filtered_results), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j < len(filtered_results):
                result = filtered_results[i + j]
                figure_data = figures[result['figure_id']]
                
                with cols[j]:
                    display_figure_card(figure_data, result)

def display_figure_card(figure_data, classification_result):
    """Display a single figure card with classification info."""
    with st.container():
        st.image(
            figure_data['image'],
            caption=f"Page {classification_result['page']} - {classification_result['classification']}",
            use_column_width=True
        )
        
        # Figure details
        st.markdown(f"""
        **Type:** {classification_result['classification']}  
        **Confidence:** {classification_result['confidence']:.1%}  
        **Page:** {classification_result['page']}
        """)
        
        # Download button for individual figure
        img_buffer = io.BytesIO()
        figure_data['image'].save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        st.download_button(
            label="Download Figure",
            data=img_buffer,
            file_name=f"figure_page_{classification_result['page']}_{classification_result['figure_id']}.png",
            mime="image/png",
            key=f"download_{classification_result['figure_id']}"
        )

def create_zip_download(figures, classifications):
    """Create a ZIP file containing all extracted figures."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Create summary CSV
        summary_data = []
        for i, (figure, classification) in enumerate(zip(figures, classifications)):
            summary_data.append({
                'Figure ID': i,
                'Filename': f"figure_page_{classification['page']}_{i}.png",
                'Type': classification['classification'],
                'Confidence': f"{classification['confidence']:.1%}",
                'Page': classification['page']
            })
        
        df_summary = pd.DataFrame(summary_data)
        csv_buffer = io.StringIO()
        df_summary.to_csv(csv_buffer, index=False)
        zip_file.writestr('figure_summary.csv', csv_buffer.getvalue())
        
        # Add individual figures
        for i, (figure, classification) in enumerate(zip(figures, classifications)):
            img_buffer = io.BytesIO()
            figure['image'].save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            filename = f"figure_page_{classification['page']}_{i}.png"
            zip_file.writestr(filename, img_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer

if __name__ == "__main__":
    main()
