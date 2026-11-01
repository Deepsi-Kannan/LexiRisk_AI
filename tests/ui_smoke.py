"""Integration checks using the real saved model and Streamlit's app harness."""
import hashlib
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


def run():
    protected = [Path('src') / name for name in ['risk_engine.py', 'clause_segmenter.py', 'classify_contract.py', 'pdf_extractor.py']]
    protected += list(Path('models').glob('*'))
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected if path.is_file()}
    pdf, df = app.analyse_contract(Path('data/raw/sample_contract.pdf'))
    assert len(df) and pdf['page_count'] > 0
    print(f"Sample PDF: {pdf['page_count']} pages, {pdf['character_count']} characters, {len(df)} clauses")
    assert app.filter_clauses(df, 'zzzz_no_match', 'ALL', 'All categories', 'Review Priority').empty
    for level in ['HIGH', 'MEDIUM', 'LOW', 'NONE']:
        subset = app.filter_clauses(df, '', level, 'All categories', 'Review Priority')
        assert subset['Risk Level'].eq(level).all()
    categories = {cat for value in df['CUAD Categories'] for cat in value.split(' | ') if cat}
    for category in categories:
        subset = app.filter_clauses(df, '', 'ALL', category, 'Clause order')
        assert subset['CUAD Categories'].apply(lambda value: category in value.split(' | ')).all()
    sorted_df = app.filter_clauses(df, '', 'ALL', 'All categories', 'Review Priority')
    priority = sorted_df['Risk Level'].map({'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'NONE': 0})
    assert priority.is_monotonic_decreasing
    exported = pd.read_csv(__import__('io').StringIO(df.to_csv(index=False)))
    assert len(exported) == len(df) and list(exported.columns) == list(df.columns)
    print('Starting AppTest', flush=True)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
    assert not at.exception
    print('Initial page passed', flush=True)
    assert at.button[0].disabled
    for page in ['Results', 'Clause Review', 'About / Methodology', 'Analyse Contract']:
        print('Page: ' + page, flush=True)
        at.radio[0].set_value(page).run()
        assert not at.exception, at.exception
    at.radio[0].set_value('Results').run()
    at.button[0].click().run()
    assert at.session_state['page'] == 'Analyse Contract'
    for key, value in dict(analysis_df=df, pdf_pages=pdf['page_count'], pdf_characters=pdf['character_count'], uploaded_name='sample_contract.pdf', analysis_complete=True).items():
        at.session_state[key] = value
    with patch('classify_contract.classify_clauses', side_effect=AssertionError('Navigation reran classifier')):
        for page in ['Results', 'Clause Review', 'About / Methodology', 'Analyse Contract', 'Results']:
            at.radio[0].set_value(page).run(timeout=30)
            assert not at.exception, at.exception
            pd.testing.assert_frame_equal(at.session_state['analysis_df'], df)
        assert len(at.get('download_button')) == 1
        at.radio[0].set_value('Clause Review').run()
        assert len(at.get('download_button')) == 2
        at.text_input[0].set_value('zzzz_no_match').run()
        assert not at.exception and not at.expander
        at.text_input[0].set_value('').run()
        at.selectbox[0].set_value('HIGH').run()
        assert len(at.expander) == df['Risk Level'].eq('HIGH').sum()
        at.selectbox[0].set_value('ALL').run()
        if categories:
            category = sorted(categories)[0]
            at.selectbox[1].set_value(category).run()
            assert len(at.expander) == df['CUAD Categories'].apply(lambda value: category in value.split(' | ')).sum()
    with patch.object(app, 'extract_text_from_pdf', return_value={'text': '  '}):
        try:
            app.analyse_contract('empty.pdf')
            raise AssertionError('Empty text accepted')
        except ValueError:
            pass
    with patch.object(app, 'extract_text_from_pdf', return_value={'text': 'readable'}), patch.object(app, 'segment_contract', return_value=[]):
        try:
            app.analyse_contract('empty.pdf')
            raise AssertionError('Empty clauses accepted')
        except ValueError:
            pass
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        invalid = Path(temp) / 'invalid.pdf'
        invalid.write_bytes(b'not a pdf')
        try:
            app.analyse_contract(invalid)
            raise AssertionError('Invalid PDF accepted')
        except ValueError:
            pass
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected if path.is_file()}
    assert before == after
    print('PASS: all pages, empty states, navigation buttons, session persistence, real-model sample analysis, search, priority/category filters, sorting, CSV round-trip and download controls, invalid/empty input. Backend/model hashes unchanged.')


if __name__ == '__main__':
    run()




