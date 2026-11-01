"""Presentation helpers. All document content is escaped before HTML rendering."""
from html import escape
from pathlib import Path
from urllib.parse import quote
import pandas as pd
import streamlit as st

DISCLAIMER = ('LexiRisk AI is an educational decision-support prototype. '
              'It does not provide legal advice and should not replace review '
              'by a qualified legal professional.')

# Original, compact line icons; no external image or icon dependencies.
ICONS = {
    'document': '<path d="M6 2h8l4 4v16H6z M14 2v5h4 M9 11h6 M9 15h6 M9 18h4"/>',
    'upload': '<path d="M12 16V3m-5 5 5-5 5 5 M4 14v7h16v-7"/>',
    'layers': '<path d="m2 7 10-5 10 5-10 5z m0 5 10 5 10-5 M2 17l10 5 10-5"/>',
    'list': '<path d="M8 5h14 M8 12h14 M8 19h14 M2 5h1 M2 12h1 M2 19h1"/>',
    'chip': '<rect x="5" y="5" width="14" height="14" rx="2"/><path d="M9 1v4m6-4v4M9 19v4m6-4v4M1 9h4m-4 6h4m14-6h4m-4 6h4"/>',
    'shield': '<path d="m12 2 9 4v6c0 5-9 10-9 10S3 17 3 12V6z m-5 9 4 4 6-7"/>',
    'chart': '<path d="M4 21v-7h3v7 M11 21V8h3v13 M18 21V2h3v19"/>',
    'warning': '<path d="m12 2 10 19H2z M12 8v6 M12 17v1"/>',
    'info': '<circle cx="12" cy="12" r="10"/><path d="M12 10v7 M12 6v1"/>',
    'minus': '<circle cx="12" cy="12" r="10"/><path d="M6 12h12"/>',
    'scales': '<path d="M12 2v19 M7 22h10 M3 6h18 M5 6 1 16h8L5 6 M19 6l-4 10h8L19 6 M1 16c0 4 8 4 8 0 M15 16c0 4 8 4 8 0"/>',
    'ring': '<circle cx="12" cy="12" r="9"/><path d="M12 3v9l7 5 M12 12l-8 5"/>',
}


def icon(name):
    return f'<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>'


def html(content):
    st.markdown(content, unsafe_allow_html=True)


def apply_styles():
    courthouse = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" fill="none" stroke="#8cd8d3" stroke-width="2"><path d="M20 100 200 10l180 90z M35 110h330 M30 265h340 M20 278h360 M10 290h380 M70 115v145m20-145v145m55-145v145m20-145v145m65-145v145m20-145v145m55-145v145m20-145v145 M55 120h50m25 0h50m35 0h50m25 0h50 M55 255h50m25 0h50m35 0h50m25 0h50 M75 85l125-60 125 60z"/></svg>'
    waves = ''.join(f'<path d="M-100 {80+i*8} C200 {250+i*8}, 450 {-200+i*8}, 850 {40+i*8} S1300 {300+i*8},1700 {100+i*8}"/>' for i in range(18))
    art = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 1000" fill="none" stroke="#31c4b0" stroke-width=".65">{waves}<g transform="translate(15 20) scale(.8)">{courthouse[courthouse.index("<path"):courthouse.rindex("</svg>")]}</g><g transform="translate(1320 55) scale(6)" stroke-width=".25">{ICONS["scales"]}</g></svg>'
    variables = f':root{{--court-art:url("data:image/svg+xml,{quote(courthouse)}");--legal-art:url("data:image/svg+xml,{quote(art)}");}}'
    for name in ['document', 'chart', 'list', 'info', 'upload']:
        svg = icon(name).replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ').replace('currentColor', '#ffffff')
        variables += f':root{{--icon-{name}:url("data:image/svg+xml,{quote(svg)}");}}'
    css = Path(__file__).with_name('ui_styles.css').read_text(encoding='utf-8-sig')
    html(f'<style>{variables}{css}</style>')


def render_brand():
    html(f'<div class="brand">{icon("scales")}LexiRisk <em>AI</em></div>')


def render_hero():
    html('<div class="hero"><div class="hero-title">LexiRisk <em>AI</em></div><div class="hero-subtitle">Intelligent Contract Review &amp; Risk Insights</div><span class="prototype">Educational AI Prototype</span></div>')


def render_page_header(title, description, symbol='info'):
    html('<div class="top-badge"><span class="prototype">Educational AI Prototype</span></div>')
    html(f'<div class="page-heading">{icon(symbol)}<h1>{escape(title)}</h1></div><div class="intro">{escape(description)}</div>')


def render_section_title(title, symbol):
    html(f'<div class="section-title">{icon(symbol)}<span>{escape(title)}</span></div>')


def render_panel(title, body):
    html(f'<div class="panel"><h3>{escape(title)}</h3><p>{escape(body)}</p></div>')


def render_document_card(filename, pages=None, characters=None, selected=False):
    meta = []
    if pages is not None:
        meta.append(f'Pages: {pages}')
    if characters is not None:
        meta.append(f'Extracted characters: {characters:,}')
    if not meta:
        meta.append('PDF selected · ready to analyse')
    html(f'<div class="document-card">{icon("document")}<div><div class="document-name">{escape(filename)}</div><div class="document-meta">{" &nbsp; | &nbsp; ".join(meta)}</div></div>{"<span class=file-check>✓</span>" if selected else ""}</div>')


def render_summary(df):
    cards = [('Total Clauses', len(df), 'total', 'document'),
             ('CUAD Classified', int(df['CUAD Categories'].ne('').sum()), 'cuad', 'layers')]
    cards += [(label, int(df['Risk Level'].eq(level).sum()), tone, symbol)
              for level, label, tone, symbol in [('HIGH', 'High Review', 'high', 'warning'),
              ('MEDIUM', 'Medium Review', 'medium', 'warning'), ('LOW', 'Low Review', 'low', 'info'),
              ('NONE', 'No Rule Match', 'none', 'minus')]]
    html('<div class="summary-grid">' + ''.join(
        f'<div class="summary-card tone-{tone}">{icon(symbol)}<div><div class="summary-number">{value}</div><div class="summary-label">{label}</div></div></div>'
        for label, value, tone, symbol in cards) + '</div>')


def render_distribution(df):
    total = len(df)
    counts = df['Risk Level'].value_counts()
    stops, rows, start = [], [], 0.0
    for level, label, color in [('HIGH', 'High', '#ed7773'), ('MEDIUM', 'Medium', '#e8b94e'),
                                 ('LOW', 'Low', '#38aee2'), ('NONE', 'No Match', '#bacbd3')]:
        count = int(counts.get(level, 0))
        percentage = count / total * 100 if total else 0
        stops.append(f'{color} {start:.5f}% {start+percentage:.5f}%')
        start += percentage
        rows.append(f'<div class="legend-row"><span class="legend-dot" style="background:{color}"></span>{label}<span class="legend-value">{count} ({percentage:.1f}%)</span></div>')
    gradient = ','.join(stops) if total else '#30434d 0% 100%'
    html(f'<div class="distribution"><div class="donut" role="img" aria-label="Review Priority distribution" style="background:conic-gradient({gradient})"><div class="donut-hole"><strong>{total}</strong><span>Clauses</span></div></div><div class="legend">{"".join(rows)}</div></div>')


def render_disclaimer():
    html(f'<div class="disclaimer">{DISCLAIMER}</div>')


def render_risk_badge(level):
    return f'<span class="badge {escape(level.lower())}">{escape(level)}</span>'


def render_pipeline():
    stages = [('PDF Upload', 'upload'), ('Text Extraction', 'document'),
              ('Clause Segmentation', 'list'), ('CUAD Classification', 'chip'), ('Risk Review', 'shield')]
    content = '<span class="stage-arrow">→</span>'.join(f'<div class="stage">{icon(symbol)}{title}</div>' for title, symbol in stages)
    html(f'<div class="pipeline"><div class="section-title">{icon("layers")}Processing Pipeline</div><div class="pipeline-stages">{content}</div></div>')


def render_empty_state():
    render_panel('Your review workspace is ready', 'Analyse a text-based contract PDF to see its clause categories and transparent review indicators here.')
    st.button('Go to Analyse Contract', on_click=lambda: st.session_state.update(page='Analyse Contract'))


def clean(value, fallback='—'):
    return fallback if value is None or pd.isna(value) or str(value) == '' else str(value)


def render_clause_card(row):
    section = clean(row['Section'], 'Unsectioned').replace('*', '').replace('[', '').replace(']', '')
    subsection = clean(row['Subsection']).replace('*', '').replace('[', '').replace(']', '')
    level = row['Risk Level']
    color = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'blue', 'NONE': 'gray'}[level]
    title = f"**Clause {row['Clause Number']}**　 Section: {section}　 |　 Subsection: {subsection}　 :{color}[{level}]"
    with st.expander(title):
        html(f'<div class="clause-text"><h3 class="detail-heading">{icon("document")} Clause Text</h3><div class="clause-copy">{escape(row["Clause Text"])}</div></div>')
        left, right = st.columns(2)
        with left:
            if row['CUAD Categories']:
                categories = f'<p><b>CUAD categories</b><br>{escape(row["CUAD Categories"])}</p><p><b>Decision Margin</b> (same category order)<br>{escape(row["Decision Margin"])}</p>'
            else:
                categories = '<p>No CUAD category exceeded the selected decision threshold.</p>'
            html(f'<div class="detail-card cuad"><h3 class="detail-heading">{icon("layers")} CUAD Clause Classification</h3>{categories}<p class="detail-note">ⓘ &nbsp; SVM decision margins are not probabilities.</p></div>')
        with right:
            if row['Matched Risk Rules']:
                details = ''.join(f'<p><b>{label}:</b> {escape(clean(row[field]))}</p>' for label, field in [
                    ('Matched Rule', 'Matched Risk Rules'), ('Matched Text', 'Matched Text'),
                    ('Explanation', 'Risk Explanation'), ('Rule IDs', 'Matched Rule IDs')])
            else:
                details = '<p>No custom risk-review rule matched this clause.</p><p class="detail-note">No match does not establish legal safety.</p>'
            html(f'<div class="detail-card risk"><h3 class="detail-heading">{icon("shield")} Risk Review Indicator</h3><p><b>Review Level:</b> {render_risk_badge(level)}</p>{details}</div>')
