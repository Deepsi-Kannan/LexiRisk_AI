"""Streamlit interface; classification and review rules remain separate."""
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from pdf_extractor import extract_text_from_pdf
from clause_segmenter import segment_contract
from classify_contract import load_model, classify_clauses
from risk_engine import analyze_clause_risk
from ui_helpers import (apply_styles, render_page_header, render_panel,
                        render_disclaimer, render_pipeline, render_empty_state,
                        render_clause_card, render_brand, render_hero,
                        render_section_title, render_document_card, render_summary,
                        render_distribution, html)

st.set_page_config(page_title='LexiRisk AI', page_icon='⚖️', layout='wide')


@st.cache_resource
def get_model_components():
    return load_model()


def analyse_contract(pdf_path):
    pdf = extract_text_from_pdf(pdf_path)
    if not pdf['text'].strip():
        raise ValueError('No readable text was found. Please upload a text-based PDF; scanned documents require OCR.')
    clauses = segment_contract(pdf['text'])
    if not clauses:
        raise ValueError('No candidate clauses were detected. Try a contract with readable clause text.')
    predictions = classify_clauses(clauses, *get_model_components())
    if not predictions:
        raise ValueError('No clause analysis was returned. Please try another contract.')
    rows = []
    for result in predictions:
        risk = analyze_clause_risk(result['clause_text'])
        rules = risk['matched_rules']
        rows.append({
            'Clause Number': result['clause_number'],
            'Section': result['main_section'], 'Subsection': result['subsection'],
            'Clause Text': result['clause_text'],
            'CUAD Categories': ' | '.join(result['predicted_categories']),
            'Decision Margin': ' | '.join(f'{score:.4f}' for score in result['prediction_scores']),
            'Top CUAD Category': result['top_category'],
            'Top Decision Margin': result['top_decision_score'],
            'Risk Level': risk['risk_level'],
            'Matched Rule IDs': ' | '.join(rule['rule_id'] for rule in rules),
            'Matched Risk Rules': ' | '.join(rule['rule_name'] for rule in rules),
            'Matched Text': ' | '.join(rule['matched_text'] for rule in rules),
            'Risk Explanation': ' | '.join(rule['explanation'] for rule in rules),
        })
    return pdf, pd.DataFrame(rows)


def filter_clauses(df, search, risk, category, sort):
    """Filter cached results only; never call the classification model."""
    result = df.copy()
    if search.strip():
        mask = result[['Clause Text', 'Section', 'CUAD Categories', 'Matched Risk Rules']].fillna('').apply(
            lambda column: column.astype(str).str.contains(search.strip(), case=False, regex=False))
        result = result[mask.any(axis=1)]
    if risk != 'ALL':
        result = result[result['Risk Level'] == risk]
    if category == 'No CUAD prediction':
        result = result[result['CUAD Categories'] == '']
    elif category != 'All categories':
        result = result[result['CUAD Categories'].apply(lambda value: category in value.split(' | '))]
    if sort == 'Review Priority':
        result = result.assign(_priority=result['Risk Level'].map({'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'NONE': 0}))
        result = result.sort_values(['_priority', 'Clause Number'], ascending=[False, True]).drop(columns='_priority')
    else:
        result = result.sort_values('Clause Number')
    return result


def export_csv(df, label='Download Full Analysis CSV', key='full_csv'):
    st.download_button(label, df.to_csv(index=False).encode('utf-8'),
                       file_name='lexirisk_contract_analysis.csv', mime='text/csv', key=key, use_container_width=True)


def show_analysis():
    render_hero()
    with st.container(border=True, key='upload_panel'):
        render_section_title('Analyse a Contract', 'document')
        st.write('Upload a text-based PDF contract to identify clause categories and transparent review indicators.')
        uploaded = st.file_uploader('Upload a contract PDF', type=['pdf'],
                                   help='Text-based PDF required. Scanned image PDFs are not supported.',
                                   label_visibility='collapsed')
        if uploaded is not None:
            known = st.session_state.analysis_complete and uploaded.name == st.session_state.uploaded_name
            render_document_card(uploaded.name, st.session_state.pdf_pages if known else None, selected=True)
        if st.button('Analyse Contract', type='primary', disabled=uploaded is None, use_container_width=True):
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as handle:
                    handle.write(uploaded.getvalue())
                    temp_path = Path(handle.name)
                with st.spinner('Extracting text, identifying clause categories and applying review rules…'):
                    pdf, df = analyse_contract(temp_path)
                st.session_state.update(analysis_df=df, pdf_pages=pdf['page_count'],
                                        pdf_characters=pdf['character_count'], uploaded_name=uploaded.name,
                                        analysis_complete=True)
                st.success('Analysis complete. Open Results or Clause Review to explore your document.')
            except Exception as error:
                st.error(f'Analysis could not be completed: {error}')
                if st.session_state.analysis_complete:
                    st.info('The previous completed analysis is still available. Its filename is shown in the sidebar.')
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
        st.caption('ⓘ Scanned/image-only PDFs require OCR and are not fully supported in this prototype.')
        if st.session_state.analysis_complete:
            st.button('View Results', on_click=lambda: st.session_state.update(page='Results'), use_container_width=True)
    render_pipeline()
    st.caption('Analysis runs only when you select Analyse Contract. Completed results persist across pages in this session.')


def show_results():
    render_page_header('Analysis Results', 'Contract analysis summary and review insights.', 'chart')
    if not st.session_state.analysis_complete:
        render_empty_state()
        return
    df = st.session_state.analysis_df
    render_document_card(st.session_state.uploaded_name, st.session_state.pdf_pages, st.session_state.pdf_characters)
    with st.container(border=True, key='summary_panel'):
        render_section_title('Analysis Summary', 'layers')
        render_summary(df)
    with st.container(border=True, key='distribution_panel'):
        render_section_title('Risk Distribution', 'ring')
        render_distribution(df)
        st.caption('Review Priority from transparent heuristic rules. No Rule Match is not an assurance of legal safety.')
    with st.container(border=True, key='export_panel'):
        render_section_title('Export Analysis', 'upload')
        st.write('Download the complete analysis with clause text, CUAD predictions, decision margins, matched rules and explanations.')
        export_csv(df)
    st.button('Explore Clause Review', on_click=lambda: st.session_state.update(page='Clause Review'))


def show_review():
    render_page_header('Clause Review', 'Explore and review individual contract clauses.', 'list')
    if not st.session_state.analysis_complete:
        render_empty_state()
        return
    df = st.session_state.analysis_df
    st.caption(st.session_state.uploaded_name)
    with st.container(border=True, key='filter_toolbar'):
        columns = st.columns([2.6, 1.1, 1.6, 1.25])
        search = columns[0].text_input('Search clauses', placeholder='Search clause text…')
        risk = columns[1].selectbox('Review Priority', ['ALL', 'HIGH', 'MEDIUM', 'LOW', 'NONE'])
        categories = sorted({category for value in df['CUAD Categories'] for category in value.split(' | ') if category})
        category = columns[2].selectbox('CUAD category', ['All categories', 'No CUAD prediction'] + categories)
        sort = columns[3].selectbox('Sort by', ['Review Priority', 'Clause order'])
        filtered = filter_clauses(df, search, risk, category, sort)
        html(f'<span class="showing">Showing {len(filtered)} of {len(df)} clauses</span>')
    if filtered.empty:
        render_panel('No matching clauses', 'Try a broader search or choose ALL review priorities and All categories.')
    else:
        for _, row in filtered.iterrows():
            render_clause_card(row)
    export_csv(df)
    export_csv(filtered, 'Download Filtered CSV', 'filtered_csv')


def show_methodology():
    render_page_header('About / Methodology', 'Two distinct components. Transparent interpretation. Human judgment at every stage.')
    render_panel('CUAD Clause Classification', 'LexiRisk AI uses a multi-label Linear SVM trained using CUAD clause annotations for clause-category identification. A clause may receive multiple categories, or none when no category meets the decision threshold.')
    render_panel('Risk Review Indicator', 'A separate transparent rule-based system matches clause wording against custom rules and assigns a Review Priority. These rules are heuristic review indicators, not learned legal-risk labels. CUAD categories and risk-review levels are separate components; category predictions do not determine review priority.')
    render_panel('Interpreting a Decision Margin', 'SVM scores are decision margins, not probabilities or confidence percentages. The current classification threshold is 0.00. A larger margin reflects the classifier’s position relative to its decision boundary, not legal significance.')
    render_panel('Human review is required', 'Read the complete contract and assess context, jurisdiction and the parties’ circumstances. Segmentation, classification and keyword rules can miss relevant wording or produce false matches. No category prediction or no rule match does not establish legal safety. This prototype does not perform OCR.')
    path = PROJECT_ROOT / 'outputs' / 'final_test_metrics.csv'
    if path.exists():
        try:
            metrics = pd.read_csv(path).iloc[0]
            st.subheader('Recorded clause-classification evaluation')
            st.caption('Source: outputs/final_test_metrics.csv. Saved evaluation results, not a new evaluation or a measure of rule-based legal-risk performance.')
            columns = st.columns(3)
            for col, label, field in zip(columns, ['Micro F1', 'Macro F1', 'Weighted F1'], ['micro_f1', 'macro_f1', 'weighted_f1']):
                col.metric(label, f'{metrics[field]:.4f}')
            st.caption(f"Recorded test set: {int(metrics['test_samples']):,} samples across {int(metrics['test_contracts'])} contracts. Decision threshold: {metrics['decision_threshold']:.2f}.")
        except (ValueError, KeyError, IndexError, pd.errors.ParserError):
            st.caption('Saved evaluation metrics are unavailable in a supported format.')


def main():
    for key, value in {'page': 'Analyse Contract', 'analysis_df': None, 'pdf_pages': 0,
                       'pdf_characters': 0, 'uploaded_name': None, 'analysis_complete': False}.items():
        st.session_state.setdefault(key, value)
    apply_styles()
    with st.sidebar:
        render_brand()
        st.radio('Workspace', ['Analyse Contract', 'Results', 'Clause Review', 'About / Methodology'], key='page',
                 format_func=lambda value: value)
        with st.container(key='sidebar_status'):
            if st.session_state.analysis_complete:
                st.caption('COMPLETED ANALYSIS')
                st.text(st.session_state.uploaded_name)
                st.caption(f'{len(st.session_state.analysis_df)} clauses · ready for review')
            else:
                st.caption('No contract analysed yet')
            st.caption('CUAD categories + transparent review rules. Human review required.')
    {'Analyse Contract': show_analysis, 'Results': show_results,
     'Clause Review': show_review, 'About / Methodology': show_methodology}[st.session_state.page]()
    render_disclaimer()


if __name__ == '__main__':
    main()
