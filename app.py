import streamlit as st
import os
import tempfile
import zipfile
import io
from datetime import datetime
from PIL import Image
import pandas as pd
from figure_extractor import PDFFigureExtractor
from ai_classifier import AIFigureClassifier
from pdf_downloader import PDFDownloader
from report_generator import PDFReportGenerator
from utils import create_download_link, get_file_size, format_figure_type, get_figure_type_emoji

# Initialize session state
if 'extracted_figures' not in st.session_state:
    st.session_state.extracted_figures = []
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'source_info' not in st.session_state:
    st.session_state.source_info = "PDF Document"

def main():
    st.set_page_config(
        page_title="PDF Figure Extraction & Classification Tool",
        page_icon="📊",
        layout="wide")

    st.title("📊 FigSense: PDF Figure Extraction & Classification Tool")
    st.markdown(
        "Upload a PDF document to automatically extract and classify all figures within it."
    )

    # Sidebar for file upload
    with st.sidebar:
        st.header("PDF Input Options")
        tab1, tab2 = st.tabs(["📁 Upload File", "🔗 From URL"])

        with tab1:
            uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

            if uploaded_file is not None:
                file_size = get_file_size(uploaded_file)
                st.info(f"File size: {file_size}")

                if st.button("Process Uploaded PDF", type="primary"):
                    process_pdf(uploaded_file)

        with tab2:
            pdf_url = st.text_input("Enter PDF URL", placeholder="https://example.com/document.pdf")

            if pdf_url:
                if st.button("Validate URL", type="secondary"):
                    validate_pdf_url(pdf_url)

                if st.button("Process PDF from URL", type="primary"):
                    process_pdf_from_url(pdf_url)

    # Main content
    if st.session_state.processing_complete and st.session_state.extracted_figures:
        display_results()
    else:
        display_welcome_screen()

def process_pdf(uploaded_file, from_url=False, url=None):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        extractor = PDFFigureExtractor()
        classifier = AIFigureClassifier()

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Extracting figures from PDF...")
        progress_bar.progress(25)

        extracted_figures = extractor.extract_figures(tmp_file_path)

        if not extracted_figures:
            st.warning("No figures found in the PDF document.")
            os.unlink(tmp_file_path)
            return

        progress_bar.progress(50)
        status_text.text(f"Found {len(extracted_figures)} figures. Classifying...")

        classification_results = []
        for i, figure_data in enumerate(extracted_figures):
            status_text.text(f"Classifying figure {i+1}/{len(extracted_figures)}...")
            result = classifier.classify_figure(figure_data['image'])
            classification_results.append({
                'figure_id': i,
                'classification': result['classification'],
                'confidence': result['confidence'],
                'description': result['description'],
                'details': result.get('details', {}),
                'reasoning': result.get('reasoning', ''),
                'page': figure_data['page'],
                'bbox': figure_data['bbox']
            })
            progress = 50 + (i + 1) * 40 / len(extracted_figures)
            progress_bar.progress(int(progress))

        progress_bar.progress(100)
        status_text.text("Processing complete!")

        st.session_state.extracted_figures = extracted_figures
        st.session_state.classification_results = classification_results
        st.session_state.processing_complete = True
        st.session_state.source_info = f"PDF from URL: {url}" if from_url else f"Uploaded PDF: {uploaded_file.name}"

        os.unlink(tmp_file_path)
        progress_bar.empty()
        status_text.empty()

        st.success(f"Successfully extracted and classified {len(extracted_figures)} figures!")
        st.rerun()

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass

def validate_pdf_url(url):
    try:
        downloader = PDFDownloader()
        file_info = downloader.get_file_info_from_url(url)

        if file_info is None:
            st.error("Could not access the URL.")
            return
        if not file_info['is_pdf']:
            st.error("The URL does not point to a PDF file.")
            return

        st.success("✅ Valid PDF URL!")
        st.info(f"""
        **File Information:**
        - Size: {file_info['file_size_mb']} MB
        - Type: {file_info['content_type']}
        - URL: {file_info['url']}
        """)

    except Exception as e:
        st.error(f"Error validating URL: {str(e)}")

def process_pdf_from_url(url):
    try:
        downloader = PDFDownloader()
        tmp_file_path = downloader.download_pdf_from_url(url)

        with open(tmp_file_path, 'rb') as f:
            content = f.read()

        class MockUploadedFile:
            def __init__(self, content, name):
                self.content = content
                self.name = name
            def getvalue(self):
                return self.content

        mock_file = MockUploadedFile(content, url.split('/')[-1])
        process_pdf(mock_file, from_url=True, url=url)
        os.unlink(tmp_file_path)

    except Exception as e:
        st.error(f"Error processing PDF from URL: {str(e)}")

def display_welcome_screen():
    st.markdown("""
    ## Welcome to FigSense
    
    This tool helps you:
    - 📄 Upload or link a PDF
    - 🖼️ Extract all figures
    - 🤖 Classify them using AI
    - 📊 Visualize and download results
    
    Try uploading a PDF to begin!
    """)

def display_results():
    figures = st.session_state.extracted_figures
    classifications = st.session_state.classification_results

    st.header("📈 Analysis Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Figures", len(figures))
    with col2:
        unique_types = len(set(c['classification'] for c in classifications))
        st.metric("Figure Types", unique_types)
    with col3:
        avg_conf = sum(c['confidence'] for c in classifications) / len(classifications)
        st.metric("Avg Confidence", f"{avg_conf:.1%}")

    st.subheader("Figure Type Distribution")
    type_counts = {}
    for c in classifications:
        t = c['classification']
        type_counts[t] = type_counts.get(t, 0) + 1
    df_types = pd.DataFrame(list(type_counts.items()), columns=['Type', 'Count'])
    st.bar_chart(df_types.set_index('Type'))

    st.subheader("Download Options")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📦 Download All Figures as ZIP"):
            zip_buffer = create_zip_download(figures, classifications)
            st.download_button("Download ZIP", zip_buffer, "extracted_figures.zip", "application/zip")
    with col2:
        if st.button("📄 Generate Analysis Report"):
            generate_pdf_report(figures, classifications)

    st.header("🖼️ Extracted Figures")
    col1, col2 = st.columns(2)
    with col1:
        filter_type = st.selectbox("Filter by type:", ['All'] + sorted(type_counts.keys()))
    with col2:
        sort_by = st.selectbox("Sort by:", ['Page Number', 'Confidence', 'Figure Type'])

    filtered = classifications
    if filter_type != 'All':
        filtered = [c for c in classifications if c['classification'] == filter_type]
    if sort_by == 'Page Number':
        filtered.sort(key=lambda x: x['page'])
    elif sort_by == 'Confidence':
        filtered.sort(key=lambda x: x['confidence'], reverse=True)
    elif sort_by == 'Figure Type':
        filtered.sort(key=lambda x: x['classification'])

    for i in range(0, len(filtered), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(filtered):
                result = filtered[i + j]
                figure_data = figures[result['figure_id']]
                with cols[j]:
                    display_figure_card(figure_data, result)

def display_figure_card(figure_data, result):
    emoji = get_figure_type_emoji(result['classification'])
    label = format_figure_type(result['classification'])
    st.image(figure_data['image'], caption=f"Page {result['page']} - {emoji} {label}", use_container_width=True)

    conf = result['confidence']
    color = "🟢" if conf > 0.8 else "🟡" if conf > 0.6 else "🔴"
    st.markdown(f"""
    **Type:** {emoji} {label}  
    **Confidence:** {color} {conf:.1%}  
    **Page:** {result['page']}  
    **Description:** {result.get('description', 'N/A')}
    """)

    if result.get('details') and result['details'].get('visual_elements'):
        st.caption("Visual Elements: " + ", ".join(result['details']['visual_elements']))

    if result.get('reasoning'):
        with st.expander("AI Reasoning"):
            st.write(result['reasoning'])

    buffer = io.BytesIO()
    figure_data['image'].save(buffer, format='PNG')
    buffer.seek(0)
    st.download_button("Download Figure", buffer, f"figure_page_{result['page']}_{result['figure_id']}.png", "image/png", key=f"dl_{result['figure_id']}")

def create_zip_download(figures, classifications):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        df = pd.DataFrame([{
            'Figure ID': i,
            'Filename': f"figure_page_{c['page']}_{i}.png",
            'Type': c['classification'],
            'Confidence': f"{c['confidence']:.1%}",
            'Page': c['page']
        } for i, c in enumerate(classifications)])
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        zip_file.writestr('figure_summary.csv', csv_buffer.getvalue())

        for i, (fig, c) in enumerate(zip(figures, classifications)):
            img_buf = io.BytesIO()
            fig['image'].save(img_buf, format='PNG')
            img_buf.seek(0)
            zip_file.writestr(f"figure_page_{c['page']}_{i}.png", img_buf.getvalue())

    zip_buffer.seek(0)
    return zip_buffer

def generate_pdf_report(figures, classifications):
    try:
        progress = st.progress(0)
        status = st.empty()
        status.text("Generating PDF report...")
        progress.progress(25)

        if not figures or not classifications:
            st.error("No figures or classifications found.")
            return

        report_generator = PDFReportGenerator()
        source_info = st.session_state.get('source_info', 'PDF Document')

        progress.progress(50)
        status.text("Creating report...")

        try:
            pdf_buffer = report_generator.create_summary_buffer(figures, classifications, source_info)
        except Exception as e:
            st.warning("Falling back to simple text summary.")
            from io import BytesIO
            pdf_buffer = BytesIO()
            text = f"FigSense Report\n{datetime.now()}\n{source_info}\n\n"
            for c in classifications:
                text += f"- Page {c['page']}: {c['classification']} ({c['confidence']:.1%})\n"
            pdf_buffer.write(text.encode('utf-8'))
            pdf_buffer.seek(0)

        progress.progress(100)
        status.text("Ready for download!")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("📄 Download Report", pdf_buffer, f"figsense_report_{timestamp}.pdf", "application/pdf")

    except Exception as e:
        st.error(f"Error generating report: {str(e)}")

# ✅ Final entry point
if __name__ == "__main__":
    main()
