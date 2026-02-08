import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from datetime import datetime, timedelta
import re
import io
import time
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import os

def clean_fund_name(name: str) -> str:
    """
    Normalizes fund names for comparison:
    - Takes only the part before the first '-'
    - Removes trailing '(...)' groups like (G), (Direct), etc.
    - Removes suffixes like 'Regular Growth', 'Direct Growth', 'Growth'
    - Strips whitespace and lowercases
    """
    if not isinstance(name, str):
        return ""

    # Take only before first '-'
    name = name.split("-")[0]

    # Remove any '(...)' patterns like (G), (Direct), (Regular)
    name = re.sub(r"\(.*?\)", "", name)

    # Remove common trailing plan/growth suffixes
    suffix_patterns = [
        r"regular growth",
        r"regular plan growth",
        r"regular plan",
        r"regular",
        r"direct growth",
        r"direct plan growth",
        r"direct plan",
        r"direct",
        r"growth option",
        r"growth"
    ]

    for pattern in suffix_patterns:
        name = re.sub(pattern + r"$", "", name.strip(), flags=re.IGNORECASE)

    # Normalize
    return name.strip().lower()


def load_rank_file(path: str):
    """Loads a CSV and adds a cleaned name column."""
    df = pd.read_csv(path)
    df["clean_name"] = df["Fund Name"].apply(clean_fund_name)
    return df


def get_fund_rank_across_categories(fund_name: str, base_path="data/rankings"):
    """
    Searches for a fund across 4 ranking files and returns:
    - category name
    - normalized ranking (index / total)
    - raw index
    """

    files = {
        "flexi_cap": "flexi_cap.csv",
        "large_cap": "large_cap.csv",
        "mid_cap": "mid_cap.csv",
        "small_cap": "small_cap.csv",
    }

    query = clean_fund_name(fund_name)

    for category, filename in files.items():
        full_path = Path(base_path) / filename

        df = load_rank_file(full_path)

        match = df[df["clean_name"] == query]

        if not match.empty:
            idx = match.index[0] + 1              # 1‑based rank
            total = len(df)
            normalized_rank = idx / total

            return {
                "category": category,
                "rank": idx,
                "total": total,
                "normalized_rank": normalized_rank,
            }

    return None  # Not found anywhere

# Page configuration
st.set_page_config(page_title="Portfolio Quarterly Report", layout="wide", page_icon="📊")

# Session state for email queue
if 'email_queue' not in st.session_state:
    st.session_state.email_queue = []
if 'sent_emails' not in st.session_state:
    st.session_state.sent_emails = []
if 'failed_emails' not in st.session_state:
    st.session_state.failed_emails = []

# Create necessary folders
DATA_FOLDER = "data"
REPORTS_FOLDER = "reports"

for folder in [DATA_FOLDER, REPORTS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def generate_pdf_report(client_data, client_name, report_date, save_to_disk=True):
    """Generate a PDF report matching the WinRich format"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                           rightMargin=0.75*inch, leftMargin=0.75*inch,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Header
    story.append(Paragraph(f"<b>Client Name:</b> {client_name}", styles['Normal']))
    story.append(Paragraph(f"<b>Report Date:</b> {report_date}", styles['Normal']))
    story.append(Paragraph("<b>Prepared by:</b> Winrich Professional Services", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Title
    story.append(Paragraph("Portfolio Allocation & Holdings Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Asset Class Allocation
    story.append(Paragraph("Asset Class Allocation", heading_style))
    asset_summary = client_data.groupby('Nature').agg({
        'CurValue': 'sum'
    }).reset_index()
    total_value = asset_summary['CurValue'].sum()
    asset_summary['Allocation %'] = (asset_summary['CurValue'] / total_value * 100).round(2)
    
    # Table 1: Asset Class Distribution
    table_data = [['Asset Class', 'Allocation (%)']]
    for _, row in asset_summary.iterrows():
        table_data.append([row['Nature'], f"{row['Allocation %']:.2f}"])
    table_data.append(['Total', '100.00'])
    
    t = Table(table_data, colWidths=[3*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(t)
    story.append(Paragraph("Table 1: Asset Class Distribution", styles['Italic']))
    story.append(Spacer(1, 0.3*inch))
    
    # Fund Type Allocation for Equity
    equity_data = client_data[client_data['Nature'] == 'Equity']
    if len(equity_data) > 0:
        story.append(Paragraph("Equity — Fund-Type Allocation", heading_style))
        
        # Extract fund types from scheme names (simplified categorization)
        def categorize_fund(name):
            name_lower = name.lower()
            if 'flexi' in name_lower or 'multi' in name_lower:
                return 'Flexi Cap Fund'
            elif 'large' in name_lower and 'mid' in name_lower:
                return 'Large & Mid Cap Fund'
            elif 'large' in name_lower:
                return 'Large Cap Fund'
            elif 'mid' in name_lower:
                return 'Mid Cap Fund'
            elif 'small' in name_lower:
                return 'Small Cap Fund'
            else:
                return 'Other Equity / Unclassified'
        
        equity_data = equity_data.copy()
        equity_data['FundType'] = equity_data['s_name'].apply(categorize_fund)
        
        equity_type_summary = equity_data.groupby('FundType').agg({
            'CurValue': 'sum'
        }).reset_index()
        equity_total = equity_type_summary['CurValue'].sum()
        equity_type_summary['Allocation %'] = (equity_type_summary['CurValue'] / equity_total * 100).round(2)
        
        table_data = [['Fund Type', 'Allocation (%)']]
        for _, row in equity_type_summary.iterrows():
            table_data.append([row['FundType'], f"{row['Allocation %']:.2f}"])
        table_data.append(['Total', '100.00'])
        
        t = Table(table_data, colWidths=[3.5*inch, 1.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(t)
        story.append(Paragraph("Table 2: Equity Fund-Type Distribution (out of 100%)", styles['Italic']))
        story.append(Spacer(1, 0.3*inch))
    
    # Fund Holdings with Rankings - Equity
    if len(equity_data) > 0:
        story.append(Paragraph("Fund Holdings with WinRich Rankings", heading_style))
        story.append(Paragraph("Equity Funds", styles['Heading3']))
        
        # Top equity funds
        top_equity = equity_data.nlargest(10, 'CurValue')[['s_name', 'absReturn', 'FolioXIRR']]
        
        table_data = [['Fund Name', 'Category', 'Category Rank', 'Absolute Return (%)', 'XIRR (%)']]
        for idx, row in top_equity.iterrows():
            # Simplified - would need actual ranking data
            ranking = get_fund_rank_across_categories(row['s_name'])  
            if ranking is not None:
                s_ranking = f"{ranking['rank']} / {ranking['total']}"

            table_data.append([
                row['s_name'][:40] + '...' if len(row['s_name']) > 40 else row['s_name'],
                ranking['category'].replace('_', ' ') if ranking is not None else 'N/A',
                s_ranking if ranking is not None else 'N/A',
                f"{row['absReturn']:.2f}" if pd.notna(row['absReturn']) else 'N/A',
                f"{row['FolioXIRR']:.2f}" if pd.notna(row['FolioXIRR']) else 'N/A'
            ])
        
        t = Table(table_data, colWidths=[2.5*inch, 1.2*inch, 1.2*inch, 1*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(t)
        story.append(Paragraph("Table 4: Equity Fund Rankings", styles['Italic']))
        story.append(Spacer(1, 0.2*inch))
    
    # Hybrid Funds
    hybrid_data = client_data[client_data['Nature'] == 'Balance']
    if len(hybrid_data) > 0:
        story.append(Paragraph("Hybrid Funds", styles['Heading3']))
        
        top_hybrid = hybrid_data.nlargest(10, 'CurValue')[['s_name', 'absReturn', 'FolioXIRR']]
        
        table_data = [['Fund Name', 'Category','Category Rank', 'Absolute Return (%)', 'XIRR (%)']]
        for idx, row in top_hybrid.iterrows():
            ranking = get_fund_rank_across_categories(row['s_name'])  # Simplified - would need actual ranking data   
            if ranking is not None:
                s_ranking = f"{ranking['rank']} / {ranking['total']}"
            table_data.append([
                row['s_name'][:40] + '...' if len(row['s_name']) > 40 else row['s_name'],
                ranking['category'].replace('_', ' ') if ranking is not None else 'N/A',
                s_ranking if ranking is not None else 'N/A',
                f"{row['absReturn']:.2f}" if pd.notna(row['absReturn']) else 'N/A',
                f"{row['FolioXIRR']:.2f}" if pd.notna(row['FolioXIRR']) else 'N/A'
            ])
        
        t = Table(table_data, colWidths=[2.5*inch, 1.2*inch, 1.2*inch, 1*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(t)
        story.append(Paragraph("Table 5: Hybrid Fund Rankings", styles['Italic']))
        story.append(Spacer(1, 0.2*inch))
    
    # Footer note
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(
        f"<i>Note: Rankings are based on WinRich proprietary methodology as of {report_date}.</i>",
        styles['Normal']
    ))
    story.append(Paragraph(
        "<i>This report is for informational purposes only.</i>",
        styles['Normal']
    ))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    # Optionally save to disk
    if save_to_disk:
        # Determine quarter from current date
        now = datetime.now()
        year = now.year
        month = now.month
        quarter = f"Q{(month - 1) // 3 + 1}"
        
        # Clean client name for filename (remove special characters)
        clean_client_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in client_name)
        clean_client_name = clean_client_name.replace(' ', '_')
        
        # Format: year_quarter_customer_name.pdf
        filename = f"{year}_{quarter}_{clean_client_name}.pdf"
        filepath = os.path.join(REPORTS_FOLDER, filename)
        
        with open(filepath, 'wb') as f:
            f.write(buffer.getvalue())
        buffer.seek(0)  # Reset buffer for return
        
    return buffer

def send_batch_emails(sender_email, sender_password, smtp_server, smtp_port, 
                     email_subject, email_body, recipients_data, batch_size, 
                     delay_seconds, progress_bar, status_text):
    """Send emails in batches with throttling"""
    total = len(recipients_data)
    sent_count = 0
    
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        
        for idx, (recipient_email, client_name, client_df) in enumerate(recipients_data):
            try:
                # Create email
                msg = MIMEMultipart()
                msg['From'] = sender_email
                msg['To'] = recipient_email
                msg['Subject'] = email_subject
                
                # Personalize email body
                personalized_body = email_body.replace('[CLIENT_NAME]', client_name)
                msg.attach(MIMEText(personalized_body, 'plain'))
                
                # Generate and attach PDF
                pdf_buffer = generate_pdf_report(
                    client_df, 
                    client_name,
                    datetime.now().strftime('%B %d, %Y')
                )
                
                attachment = MIMEBase('application', 'pdf')
                attachment.set_payload(pdf_buffer.read())
                encoders.encode_base64(attachment)
                
                # Format filename: year_quarter_customer_name.pdf
                now = datetime.now()
                year = now.year
                quarter = f"Q{(now.month - 1) // 3 + 1}"
                clean_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in client_name)
                clean_name = clean_name.replace(' ', '_')
                
                attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename={year}_{quarter}_{clean_name}.pdf'
                )
                msg.attach(attachment)
                
                # Send email
                server.send_message(msg)
                sent_count += 1
                st.session_state.sent_emails.append({
                    'email': recipient_email,
                    'client': client_name,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                # Update progress
                progress_bar.progress((idx + 1) / total)
                status_text.text(f"Sent {sent_count}/{total}: {recipient_email}")
                
                # Batch delay
                if (idx + 1) % batch_size == 0 and idx < total - 1:
                    status_text.text(f"Batch complete. Waiting {delay_seconds}s before next batch...")
                    time.sleep(delay_seconds)
                else:
                    # Small delay between individual emails
                    time.sleep(2)
                    
            except Exception as e:
                st.session_state.failed_emails.append({
                    'email': recipient_email,
                    'client': client_name,
                    'error': str(e),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                status_text.text(f"Failed to send to {recipient_email}: {str(e)}")
        
        server.quit()
        return sent_count
        
    except Exception as e:
        st.error(f"SMTP Error: {str(e)}")
        return sent_count

# Title
st.title("📊 Quarterly Portfolio Report Generator")
st.markdown("---")

# File handling - automatically load the specific CSV
CSV_FILENAME = "Datawarehouse_MutualFunds_2026_01_01_mutualfunds.csv"
csv_path = os.path.join(DATA_FOLDER, CSV_FILENAME)

df = None

if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path)
        
        # Sidebar - minimal info
        st.sidebar.success(f"✅ Data Loaded")
        st.sidebar.metric("Total Records", len(df))
        st.sidebar.metric("Total Clients", df['c_name'].nunique())
        
        # Show generated reports in sidebar
        if os.path.exists(REPORTS_FOLDER):
            pdf_files = [f for f in os.listdir(REPORTS_FOLDER) if f.endswith('.pdf')]
            if pdf_files:
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 📄 Generated Reports")
                st.sidebar.markdown(f"**{len(pdf_files)} PDF reports**")
                
                with st.sidebar.expander("View & Download Reports"):
                    for pdf_file in sorted(pdf_files, reverse=True)[:20]:  # Show last 20
                        file_path = os.path.join(REPORTS_FOLDER, pdf_file)
                        file_size = os.path.getsize(file_path) / 1024  # KB
                        st.sidebar.text(f"📄 {pdf_file}")
                        st.sidebar.caption(f"   {file_size:.1f} KB")
                        
                        # Download button for each report
                        with open(file_path, 'rb') as f:
                            st.sidebar.download_button(
                                label="⬇️ Download",
                                data=f,
                                file_name=pdf_file,
                                mime="application/pdf",
                                key=f"download_{pdf_file}"
                            )
                        st.sidebar.markdown("---")
        
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        st.info(f"Please ensure `{CSV_FILENAME}` is in the `{DATA_FOLDER}/` folder")
        df = None
else:
    st.error(f"❌ File not found: `{csv_path}`")
    st.info(f"Please place `{CSV_FILENAME}` in the `{DATA_FOLDER}/` folder")
    df = None

if df is not None:
    
    # Data processing
    df['CurValue'] = pd.to_numeric(df['CurValue'], errors='coerce')
    df['InvAmt'] = pd.to_numeric(df['InvAmt'], errors='coerce')
    df['NotionalGain'] = pd.to_numeric(df['NotionalGain'], errors='coerce')
    df['absReturn'] = pd.to_numeric(df['absReturn'], errors='coerce')
    
    # Client filter in main panel
    st.markdown("### 🔍 Select Client")
    clients = df['c_name'].unique()
    selected_client = st.selectbox("Choose a client to view details:", ['All'] + list(clients))
    
    if selected_client != 'All':
        df_filtered = df[df['c_name'] == selected_client]
    else:
        df_filtered = df.copy()
    
    st.markdown("---")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_investment = df_filtered['InvAmt'].sum()
    current_value = df_filtered['CurValue'].sum()
    total_gain = df_filtered['NotionalGain'].sum()
    gain_percentage = (total_gain / total_investment * 100) if total_investment > 0 else 0
    unique_clients = df_filtered['c_name'].nunique()
    
    with col1:
        if selected_client != 'All':
            st.metric("Total Investment", f"₹{total_investment:,.0f}")
    with col2:
        if selected_client != 'All':
            st.metric("Current Value", f"₹{current_value:,.0f}")
    with col3:
        if selected_client != 'All':
            st.metric("Total Gain", f"₹{total_gain:,.0f}", delta=f"{gain_percentage:.2f}%")
    with col4:
        if selected_client != 'All':
            st.metric("Clients", unique_clients)
    
    st.markdown("---")
    
    # Tabs - Email Report first
    tab1, tab2, tab3, tab4 = st.tabs(["📧 Email Report", "📈 Overview", "🎯 Asset Allocation", "🏆 Top Performers"])
    
    with tab1:
        st.subheader("📧 Email Quarterly Report with Batch Sending")
        
        st.markdown("""
        Configure email settings to send quarterly portfolio reports to clients.
        **Batch sending** helps avoid email throttling by spreading emails over time.
        """)
        
        # Email configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Email Server Settings")
            sender_email = st.text_input("Sender Email", placeholder="your-email@example.com")
            sender_password = st.text_input("Email Password", type="password", 
                                          help="Use app-specific password for Gmail")
            smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
            smtp_port = st.number_input("SMTP Port", value=587, min_value=1, max_value=65535)
        
        with col2:
            st.markdown("##### Batch Settings")
            batch_size = st.number_input(
                "Emails per batch", 
                min_value=1, 
                max_value=100, 
                value=10,
                help="Number of emails to send before pausing"
            )
            delay_minutes = st.number_input(
                "Delay between batches (minutes)", 
                min_value=1, 
                max_value=1440, 
                value=30,
                help="Wait time between batches to avoid throttling"
            )
            
            email_subject = st.text_input("Email Subject", 
                                         value=f"Quarterly Portfolio Report - {datetime.now().strftime('%B %Y')}")
        
        # Recipient information
        st.markdown("---")
        st.markdown("##### Recipients")
        
        if selected_client == 'All':
            # Group by client name and email
            recipient_groups = df.groupby(['c_name', 'Email'])
            unique_recipients = len(recipient_groups)
            
            # Calculate batches
            num_batches = (unique_recipients + batch_size - 1) // batch_size
            total_time_minutes = (num_batches - 1) * delay_minutes
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Recipients", unique_recipients)
            with col2:
                st.metric("Number of Batches", num_batches)
            with col3:
                estimated_hours = total_time_minutes / 60
                st.metric("Estimated Time", f"{estimated_hours:.1f} hours")
            
            # Show schedule preview
            with st.expander("📅 View Sending Schedule"):
                schedule_data = []
                current_time = datetime.now()
                for batch_num in range(num_batches):
                    batch_start = batch_num * batch_size
                    batch_end = min((batch_num + 1) * batch_size, unique_recipients)
                    batch_count = batch_end - batch_start
                    
                    send_time = current_time + timedelta(minutes=batch_num * delay_minutes)
                    schedule_data.append({
                        'Batch': f"Batch {batch_num + 1}",
                        'Recipients': f"{batch_start + 1} - {batch_end}",
                        'Count': batch_count,
                        'Scheduled Time': send_time.strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                schedule_df = pd.DataFrame(schedule_data)
                st.dataframe(schedule_df, use_container_width=True, hide_index=True)
        else:
            recipient_emails = df_filtered['Email'].unique().tolist()
            st.info(f"Will send to: {', '.join(recipient_emails)}")
        
        # Email body template
        st.markdown("---")
        st.markdown("##### Email Body Template")
        st.info("Use [CLIENT_NAME] as a placeholder - it will be replaced with each client's name")
        
        email_body = st.text_area("Email Message", value=f"""Dear [CLIENT_NAME],

Please find attached your quarterly portfolio report for {datetime.now().strftime('%B %Y')}.

This comprehensive report includes:
• Asset class allocation and distribution
• Fund-type breakdown
• Individual fund performance with rankings
• Returns analysis (Absolute & XIRR)

Your portfolio summary:
- Total Investment: ₹{total_investment:,.0f}
- Current Value: ₹{current_value:,.0f}
- Total Gain: ₹{total_gain:,.0f} ({gain_percentage:.2f}%)

If you have any questions about your portfolio, please don't hesitate to reach out.

Best regards,
Winrich Professional Services
Portfolio Management Team""", height=350)
        
        # Send controls
        st.markdown("---")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("📄 Preview PDF", use_container_width=True, key="Previewpdf2_btn"):
                if selected_client != 'All':
                    # Generate preview for selected client
                    pdf_buffer = generate_pdf_report(
                        df_filtered,
                        selected_client,
                        datetime.now().strftime('%B %d, %Y')
                    )
                    
                    st.download_button(
                        label="📥 Download Preview PDF",
                        data=pdf_buffer,
                        file_name=f"Preview_Report_{selected_client.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        key="download_preview_pdf_btn"
                    )
                else:
                    st.warning("Please select a specific client to preview PDF")
        
        with col2:
            clear_logs = st.button("🗑️ Clear Logs", use_container_width=True, key="clear_logs2_btn")
            if clear_logs:
                st.session_state.sent_emails = []
                st.session_state.failed_emails = []
                st.success("Logs cleared!")
        
        with col3:
            if st.button("📧 Send Batch Emails", type="primary", use_container_width=True, key="send_batch_emails_btn"):
                if not sender_email or not sender_password:
                    st.error("Please provide sender email and password")
                else:
                    # Prepare recipient data
                    recipients_data = []
                    
                    if selected_client == 'All':
                        for (client_name, email), group_df in df.groupby(['c_name', 'Email']):
                            recipients_data.append((email, client_name, group_df))
                    else:
                        for email in df_filtered['Email'].unique():
                            client_df = df_filtered[df_filtered['Email'] == email]
                            recipients_data.append((email, selected_client, client_df))
                    
                    # Create progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Send emails
                    delay_seconds = delay_minutes * 60
                    sent_count = send_batch_emails(
                        sender_email, sender_password, smtp_server, smtp_port,
                        email_subject, email_body, recipients_data,
                        batch_size, delay_seconds,
                        progress_bar, status_text
                    )
                    
                    # Final status
                    if sent_count == len(recipients_data):
                        st.success(f"✅ Successfully sent all {sent_count} emails!")
                    else:
                        st.warning(f"⚠️ Sent {sent_count} out of {len(recipients_data)} emails. Check logs below.")
        
        # Email logs
        st.markdown("---")
        st.markdown("##### Email Sending Logs")
        
        log_tab1, log_tab2 = st.tabs(["✅ Sent Emails", "❌ Failed Emails"])
        
        with log_tab1:
            if st.session_state.sent_emails:
                sent_df = pd.DataFrame(st.session_state.sent_emails)
                st.dataframe(sent_df, use_container_width=True, hide_index=True)
                
                # Download sent log
                csv_sent = sent_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Sent Log",
                    data=csv_sent,
                    file_name=f'sent_emails_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                    key="download_sent_log_btn"
                )
            else:
                st.info("No emails sent yet")
        
        with log_tab2:
            if st.session_state.failed_emails:
                failed_df = pd.DataFrame(st.session_state.failed_emails)
                st.dataframe(failed_df, use_container_width=True, hide_index=True)
                
                # Download failed log
                csv_failed = failed_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Failed Log",
                    data=csv_failed,
                    file_name=f'failed_emails_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                )
            else:
                st.info("No failed emails")
        
        # Download CSV report option
        st.markdown("---")
        st.markdown("##### Download Data")
        
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV Data",
            data=csv,
            file_name=f'portfolio_data_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
            key="download_csv_btn"
        )
    
    with tab2:
        st.subheader("Portfolio Overview")
        
        # Summary by Nature (Asset Type)
        nature_summary = df_filtered.groupby('Nature').agg({
            'InvAmt': 'sum',
            'CurValue': 'sum',
            'NotionalGain': 'sum'
        }).reset_index()
        nature_summary['Return %'] = (nature_summary['NotionalGain'] / nature_summary['InvAmt'] * 100).round(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Asset Type Summary")
            st.dataframe(nature_summary.style.format({
                'InvAmt': '₹{:,.0f}',
                'CurValue': '₹{:,.0f}',
                'NotionalGain': '₹{:,.0f}',
                'Return %': '{:.2f}%'
            }), use_container_width=True)
        
        with col2:
            # Pie chart for current value by nature
            fig_pie = px.pie(nature_summary, values='CurValue', names='Nature', 
                            title='Current Value Distribution by Asset Type',
                            color_discrete_sequence=px.colors.qualitative.Set3)
            
            st.plotly_chart(fig_pie, use_container_width=True, key="nature_pie_chart")
        
        # Top funds by value
        st.markdown("##### Top 10 Funds by Current Value")
        top_funds = df_filtered.nlargest(10, 'CurValue')[['c_name', 's_name', 'Nature', 'InvAmt', 'CurValue', 'NotionalGain', 'absReturn']]
        st.dataframe(top_funds.style.format({
            'InvAmt': '₹{:,.0f}',
            'CurValue': '₹{:,.0f}',
            'NotionalGain': '₹{:,.0f}',
            'absReturn': '{:.2f}%'
        }), use_container_width=True)
    
    with tab2:
        st.subheader("Portfolio Overview")
        
        if selected_client == 'All':
            st.info("📊 Please select a specific client to view portfolio overview details")
        else:
            # Summary by Nature (Asset Type)
            nature_summary = df_filtered.groupby('Nature').agg({
                'InvAmt': 'sum',
                'CurValue': 'sum',
                'NotionalGain': 'sum'
            }).reset_index()
            nature_summary['Return %'] = (nature_summary['NotionalGain'] / nature_summary['InvAmt'] * 100).round(2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Asset Type Summary")
                st.dataframe(nature_summary.style.format({
                    'InvAmt': '₹{:,.0f}',
                    'CurValue': '₹{:,.0f}',
                    'NotionalGain': '₹{:,.0f}',
                    'Return %': '{:.2f}%'
                }), use_container_width=True)
            
            with col2:
                # Pie chart for current value by nature
                fig_pie = px.pie(nature_summary, values='CurValue', names='Nature', 
                                title='Current Value Distribution by Asset Type',
                                color_discrete_sequence=px.colors.qualitative.Set3)
                
                st.plotly_chart(fig_pie, use_container_width=True, key="current_value_by_nature_pie_chart")
            
            # Top funds by value
            st.markdown("##### Top 10 Funds by Current Value")
            top_funds = df_filtered.nlargest(10, 'CurValue')[['c_name', 's_name', 'Nature', 'InvAmt', 'CurValue', 'NotionalGain', 'absReturn']]
            st.dataframe(top_funds.style.format({
                'InvAmt': '₹{:,.0f}',
                'CurValue': '₹{:,.0f}',
                'NotionalGain': '₹{:,.0f}',
                'absReturn': '{:.2f}%'
            }), use_container_width=True)
    
    with tab3:
        st.subheader("Asset Allocation Analysis")
        
        if selected_client == 'All':
            st.info("🎯 Please select a specific client to view asset allocation details")
        else:
            col1, col2 = st.columns(2)
            
            # Summary by Nature (Asset Type)
            nature_summary = df_filtered.groupby('Nature').agg({
                'InvAmt': 'sum',
                'CurValue': 'sum',
                'NotionalGain': 'sum'
            }).reset_index()
            nature_summary['Return %'] = (nature_summary['NotionalGain'] / nature_summary['InvAmt'] * 100).round(2)
            
            with col1:
                # Investment amount by nature
                fig_inv = px.bar(nature_summary, x='Nature', y='InvAmt', 
                               title='Investment Amount by Asset Type',
                               color='Nature',
                               color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_inv.data[0].uid = "investment_bar_chart"   # or str(uuid.uuid4())
                st.plotly_chart(fig_inv, use_container_width=True)
            
            with col2:
                # Current value by nature
                fig_cur = px.bar(nature_summary, x='Nature', y='CurValue', 
                               title='Current Value by Asset Type',
                               color='Nature',
                               color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_cur.data[0].uid = "current_value_bar_chart"   # or str(uuid.uuid4())
                st.plotly_chart(fig_cur, use_container_width=True)
            
            # Detailed breakdown
            st.markdown("##### Scheme-wise Allocation")
            scheme_summary = df_filtered.groupby(['Nature', 's_name']).agg({
                'InvAmt': 'sum',
                'CurValue': 'sum',
                'NotionalGain': 'sum'
            }).reset_index()
            scheme_summary['Return %'] = (scheme_summary['NotionalGain'] / scheme_summary['InvAmt'] * 100).round(2)
            
            st.dataframe(scheme_summary.style.format({
                'InvAmt': '₹{:,.0f}',
                'CurValue': '₹{:,.0f}',
                'NotionalGain': '₹{:,.0f}',
                'Return %': '{:.2f}%'
            }), use_container_width=True)
    
    with tab4:
        st.subheader("Top Performers")
        
        if selected_client == 'All':
            st.info("🏆 Please select a specific client to view top performers")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Best Returns (Absolute %)")
                best_returns = df_filtered.nlargest(10, 'absReturn')[['c_name', 's_name', 'InvAmt', 'CurValue', 'absReturn']]
                st.dataframe(best_returns.style.format({
                    'InvAmt': '₹{:,.0f}',
                    'CurValue': '₹{:,.0f}',
                    'absReturn': '{:.2f}%'
                }), use_container_width=True)
            
            with col2:
                st.markdown("##### Highest Gains (₹)")
                highest_gains = df_filtered.nlargest(10, 'NotionalGain')[['c_name', 's_name', 'InvAmt', 'NotionalGain', 'absReturn']]
                st.dataframe(highest_gains.style.format({
                    'InvAmt': '₹{:,.0f}',
                    'NotionalGain': '₹{:,.0f}',
                    'absReturn': '{:.2f}%'
                }), use_container_width=True)
            
            # Return distribution
            fig_returns = px.histogram(df_filtered, x='absReturn', nbins=30,
                                      title='Distribution of Returns (%)',
                                      labels={'absReturn': 'Absolute Return (%)'},
                                      color_discrete_sequence=['#636EFA'])
            fig_returns.data[0].uid = "returns_histogram_chart"   # or str(uuid.uuid4())
            st.plotly_chart(fig_returns, use_container_width=True)
        st.subheader("📧 Email Quarterly Report with Batch Sending")
        
        st.markdown("""
        Configure email settings to send quarterly portfolio reports to clients.
        **Batch sending** helps avoid email throttling by spreading emails over time.
        """)
        
        # Email configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Email Server Settings")
            sender_email = st.text_input("Sender Email", placeholder="your-email@example.com", key="sender_email")
            sender_password = st.text_input("Email Password", type="password", 
                                          help="Use app-specific password for Gmail",
                                          key="sender_password")
            smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com",key="smtp_server")
            smtp_port = st.number_input("SMTP Port", value=587, min_value=1, max_value=65535, key="smtp_port")
        
        with col2:
            st.markdown("##### Batch Settings")
            batch_size = st.number_input(
                "Emails per batch", 
                min_value=1, 
                max_value=100, 
                value=10,
                help="Number of emails to send before pausing",
                key="batch_size"
            )
            delay_minutes = st.number_input(
                "Delay between batches (minutes)", 
                min_value=1, 
                max_value=1440, 
                value=30,
                help="Wait time between batches to avoid throttling",
                key="delay_minutes"
            )
            
            email_subject = st.text_input("Email Subject", 
                                         value=f"Quarterly Portfolio Report - {datetime.now().strftime('%B %Y')}",
                                         key="email_subject")
        
        # Recipient information
        st.markdown("---")
        st.markdown("##### Recipients")
        
        if selected_client == 'All':
            # Group by client name and email
            recipient_groups = df.groupby(['c_name', 'Email'])
            unique_recipients = len(recipient_groups)
            
            # Calculate batches
            num_batches = (unique_recipients + batch_size - 1) // batch_size
            total_time_minutes = (num_batches - 1) * delay_minutes
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Recipients", unique_recipients)
            with col2:
                st.metric("Number of Batches", num_batches)
            with col3:
                estimated_hours = total_time_minutes / 60
                st.metric("Estimated Time", f"{estimated_hours:.1f} hours")
            
            # Show schedule preview
            with st.expander("📅 View Sending Schedule"):
                schedule_data = []
                current_time = datetime.now()
                for batch_num in range(num_batches):
                    batch_start = batch_num * batch_size
                    batch_end = min((batch_num + 1) * batch_size, unique_recipients)
                    batch_count = batch_end - batch_start
                    
                    send_time = current_time + timedelta(minutes=batch_num * delay_minutes)
                    schedule_data.append({
                        'Batch': f"Batch {batch_num + 1}",
                        'Recipients': f"{batch_start + 1} - {batch_end}",
                        'Count': batch_count,
                        'Scheduled Time': send_time.strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                schedule_df = pd.DataFrame(schedule_data)
                st.dataframe(schedule_df, use_container_width=True, hide_index=True)
        else:
            recipient_emails = df_filtered['Email'].unique().tolist()
            st.info(f"Will send to: {', '.join(recipient_emails)}")
        
        # Email body template
        st.markdown("---")
        st.markdown("##### Email Body Template")
        st.info("Use [CLIENT_NAME] as a placeholder - it will be replaced with each client's name")
        
        email_body = st.text_area("Email Message", key="email_body", value=f"""Dear [CLIENT_NAME],

Please find attached your quarterly portfolio report for {datetime.now().strftime('%B %Y')}.

This comprehensive report includes:
• Asset class allocation and distribution
• Fund-type breakdown
• Individual fund performance with rankings
• Returns analysis (Absolute & XIRR)

Your portfolio summary:
- Total Investment: ₹{total_investment:,.0f}
- Current Value: ₹{current_value:,.0f}
- Total Gain: ₹{total_gain:,.0f} ({gain_percentage:.2f}%)

If you have any questions about your portfolio, please don't hesitate to reach out.

Best regards,
Winrich Professional Services
Portfolio Management Team""", height=350)
        
        # Send controls
        st.markdown("---")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("📄 Preview PDF", use_container_width=True, key="Previewpdf_btn"):
                if selected_client != 'All':
                    # Generate preview for selected client
                    pdf_buffer = generate_pdf_report(
                        df_filtered,
                        selected_client,
                        datetime.now().strftime('%B %d, %Y')
                    )
                    
                    st.download_button(
                        label="📥 Download Preview PDF",
                        data=pdf_buffer,
                        file_name=f"Preview_Report_{selected_client.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        key="download_preview_pdf_btn"
                    )
                else:
                    st.warning("Please select a specific client to preview PDF")
        
        with col2:
            clear_logs = st.button("🗑️ Clear Logs", use_container_width=True, key="clear_logs_btn")
            if clear_logs:
                st.session_state.sent_emails = []
                st.session_state.failed_emails = []
                st.success("Logs cleared!")
        
        with col3:
            if st.button("📧 Send Batch Emails", type="primary", use_container_width=True, key="send_batch_emails2_btn"):
                if not sender_email or not sender_password:
                    st.error("Please provide sender email and password")
                else:
                    # Prepare recipient data
                    recipients_data = []
                    
                    if selected_client == 'All':
                        for (client_name, email), group_df in df.groupby(['c_name', 'Email']):
                            recipients_data.append((email, client_name, group_df))
                    else:
                        for email in df_filtered['Email'].unique():
                            client_df = df_filtered[df_filtered['Email'] == email]
                            recipients_data.append((email, selected_client, client_df))
                    
                    # Create progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Send emails
                    delay_seconds = delay_minutes * 60
                    sent_count = send_batch_emails(
                        sender_email, sender_password, smtp_server, smtp_port,
                        email_subject, email_body, recipients_data,
                        batch_size, delay_seconds,
                        progress_bar, status_text
                    )
                    
                    # Final status
                    if sent_count == len(recipients_data):
                        st.success(f"✅ Successfully sent all {sent_count} emails!")
                    else:
                        st.warning(f"⚠️ Sent {sent_count} out of {len(recipients_data)} emails. Check logs below.")
        
        # Email logs
        st.markdown("---")
        st.markdown("##### Email Sending Logs")
        
        log_tab1, log_tab2 = st.tabs(["✅ Sent Emails", "❌ Failed Emails"])
        
        with log_tab1:
            if st.session_state.sent_emails:
                sent_df = pd.DataFrame(st.session_state.sent_emails)
                st.dataframe(sent_df, use_container_width=True, hide_index=True)
                
                # Download sent log
                csv_sent = sent_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Sent Log",
                    data=csv_sent,
                    file_name=f'sent_emails_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                    key="download_sent_log_btn"
                )
            else:
                st.info("No emails sent yet")
        
        with log_tab2:
            if st.session_state.failed_emails:
                failed_df = pd.DataFrame(st.session_state.failed_emails)
                st.dataframe(failed_df, use_container_width=True, hide_index=True)
                
                # Download failed log
                csv_failed = failed_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Failed Log",
                    data=csv_failed,
                    file_name=f'failed_emails_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                    key="download_failed_log_btn"
                )
            else:
                st.info("No failed emails")
        
        # Download CSV report option
        st.markdown("---")
        st.markdown("##### Download Data")
        
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV Data",
            data=csv,
            file_name=f'portfolio_data_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
            key="download_csv_data_btn"
        )

else:
    st.warning(f"⚠️ Waiting for CSV file: `{CSV_FILENAME}`")
    st.markdown(f"""
    ### Quick Setup:
    1. Place your CSV file in the `{DATA_FOLDER}/` folder
    2. Name it: `{CSV_FILENAME}`
    3. Refresh this page
    
    ### Features:
    - 📊 **Interactive Dashboard** - View portfolio metrics and visualizations
    - 🎯 **Asset Allocation** - Analyze distribution across Equity, Debt, and Balanced funds
    - 🏆 **Top Performers** - Identify best performing funds
    - 📧 **Batch Email Sending** - Send quarterly reports without throttling
    - 📄 **Professional PDF Reports** - WinRich format with tables and rankings
    - 📥 **Export Data** - Download reports in PDF and CSV formats
    
    ### Expected CSV Columns:
    `c_name`, `s_name`, `Nature`, `InvAmt`, `CurValue`, `NotionalGain`, `absReturn`, `Email`, `FolioXIRR`
    
    ### Folder Structure:
    - `{DATA_FOLDER}/` - Must contain `{CSV_FILENAME}`
    - `{REPORTS_FOLDER}/` - Generated PDF reports are saved here
    """)
    
    # Show current data folder status
    if os.path.exists(DATA_FOLDER):
        files_in_data = os.listdir(DATA_FOLDER)
        if files_in_data:
            st.info(f"Files currently in `{DATA_FOLDER}/`: {', '.join(files_in_data)}")
        else:
            st.info(f"The `{DATA_FOLDER}/` folder is empty")
    else:
        st.info(f"The `{DATA_FOLDER}/` folder will be created automatically")

# Footer
st.markdown("---")
st.markdown("*Portfolio Report Generator v2.0*")
