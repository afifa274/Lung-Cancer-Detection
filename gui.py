"""
Lung Cancer Detection — Flask web application
Serves the assessment form, runs model inference, and persists results.
"""
import csv
import io
import os
import pickle
import shutil
import platform
import logging
from datetime import datetime
from functools import wraps

import numpy as np
from flask import (
    Flask, render_template, request, redirect,
    url_for, send_file, flash, jsonify, abort,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'change-me-in-production-please')

# ── Load model & scaler ───────────────────────────────────────────────────────
def _load_or_retrain():
    """Try to load the saved model; if it fails, retrain from scratch."""
    import pickle as _pickle
    import numpy as _np
    import pandas as _pd
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.preprocessing import StandardScaler as _SS
    from tensorflow import keras as _keras
    from tensorflow.keras.models import Sequential as _Seq
    from tensorflow.keras.layers import Dense as _Dense, Dropout as _Drop, BatchNormalization as _BN

    # ── Try loading existing artefacts ────────────────────────────────────
    try:
        _m = _keras.models.load_model('cancer_snus_model.h5')
        with open('scaler.pkl', 'rb') as _f:
            _sc = _pickle.load(_f)
        # quick sanity-check: predict on a dummy input
        _m.predict(_np.zeros((1, 23)), verbose=0)
        logger.info('Model and scaler loaded from disk.')
        return _m, _sc
    except Exception as _e:
        logger.warning('Could not load model (%s) — retraining now...', _e)

    # ── Retrain ───────────────────────────────────────────────────────────
    _df = _pd.read_csv('cancer patient data sets.csv')
    if _df['Level'].dtype == object:
        _df['Level'] = _df['Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
    _X = _df.iloc[:, 2:-1].values
    _y = _df.iloc[:, -1].values.astype(int)
    _sc = _SS()
    _X = _sc.fit_transform(_X)
    _X_tr, _X_te, _y_tr, _y_te = _tts(_X, _y, test_size=0.2, random_state=42, stratify=_y)

    _m = _Seq([
        _Dense(128, activation='relu', input_dim=23),
        _BN(), _Drop(0.3),
        _Dense(64, activation='relu'),
        _BN(), _Drop(0.2),
        _Dense(32, activation='relu'),
        _Dense(3, activation='softmax'),
    ])
    _m.compile(optimizer=_keras.optimizers.Adam(1e-3),
               loss='categorical_crossentropy', metrics=['accuracy'])
    _m.fit(_X_tr, _keras.utils.to_categorical(_y_tr, 3),
           epochs=60, batch_size=32,
           validation_data=(_X_te, _keras.utils.to_categorical(_y_te, 3)),
           callbacks=[_keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
           verbose=0)
    _m.save('cancer_snus_model.h5')
    with open('scaler.pkl', 'wb') as _f:
        _pickle.dump(_sc, _f)
    logger.info('Model retrained and saved successfully.')
    return _m, _sc

try:
    model, scaler = _load_or_retrain()
except Exception as exc:
    logger.error('Fatal: could not load or retrain model: %s', exc)
    model, scaler = None, None

# ── CSV helpers ───────────────────────────────────────────────────────────────
CSV_PATH = 'user_inputs.csv'
CSV_HEADER = [
    'Index', 'Patient Id',
    'Age', 'Gender', 'Air Pollution', 'Alcohol Use', 'Dust Allergy',
    'Occupational Hazards', 'Genetic Risk', 'Chronic Lung Disease',
    'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker',
    'Chest Pain', 'Coughing Blood', 'Fatigue', 'Weight Loss',
    'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
    'Clubbing', 'Frequent Cold', 'Dry Cough', 'Snoring', 'Level',
]

def _ensure_csv_header() -> None:
    """Write the header row if the CSV file is missing or empty."""
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        with open(CSV_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(CSV_HEADER)

def _next_index() -> int:
    """Return the next sequential record number."""
    _ensure_csv_header()
    with open(CSV_PATH, 'r', newline='') as f:
        return max(0, len(list(f)) - 1) + 1   # minus header line

def _append_row(row: list) -> None:
    _ensure_csv_header()
    with open(CSV_PATH, 'a', newline='') as f:
        csv.writer(f).writerow(row)

def _read_patient(patient_id: str) -> list[dict]:
    _ensure_csv_header()
    results = []
    try:
        with open(CSV_PATH, 'r', newline='') as f:
            for row in csv.DictReader(f):
                if row.get('Patient Id') == patient_id:
                    results.append(row)
    except FileNotFoundError:
        pass
    return results

# ── Input validation ──────────────────────────────────────────────────────────
_SCORE_FIELDS = [
    'air_pollution', 'alcohol_use', 'dust_allergy', 'occupational_hazards',
    'genetic_risk', 'chronic_lung_disease', 'balanced_diet', 'obesity',
    'smoking', 'passive_smoker', 'chest_pain', 'coughing_blood', 'fatigue',
    'weight_loss', 'shortness_of_breath', 'wheezing', 'swallowing_difficulty',
    'clubbing', 'frequent_cold', 'dry_cough', 'snoring',
]

def _validate_form(form) -> tuple[dict | None, list[str]]:
    """
    Parse and validate the submitted form.
    Returns (parsed_values_dict, errors_list).
    """
    errors = []
    values = {}

    # Age
    try:
        age = int(form.get('age', ''))
        if not (1 <= age <= 120):
            errors.append('Age must be between 1 and 120.')
        values['age'] = age
    except (ValueError, TypeError):
        errors.append('Age is required and must be a whole number.')

    # Gender
    gender_raw = (form.get('gender') or '').strip().lower()
    if gender_raw not in ('male', 'female'):
        errors.append('Gender must be Male or Female.')
    values['gender'] = 1 if gender_raw == 'male' else 2

    # 1–9 scored fields
    for field in _SCORE_FIELDS:
        try:
            val = int(form.get(field, ''))
            if not (1 <= val <= 9):
                errors.append(f'{field.replace("_", " ").title()} must be between 1 and 9.')
            values[field] = val
        except (ValueError, TypeError):
            errors.append(f'{field.replace("_", " ").title()} is required (1–9).')

    return (values if not errors else None, errors)

# ── Model inference helper ────────────────────────────────────────────────────
def _predict(values: dict) -> tuple[str, float, float, float]:
    """Run inference and return (outcome, pct_low, pct_medium, pct_high)."""
    data = np.array([[
        values['age'], values['gender'],
        *[values[f] for f in _SCORE_FIELDS],
    ]])
    scaled = scaler.transform(data)
    probs = model.predict(scaled, verbose=0)[0]
    pct_low    = round(float(probs[0]) * 100, 1)
    pct_medium = round(float(probs[1]) * 100, 1)
    pct_high   = round(float(probs[2]) * 100, 1)
    outcome = ['Low', 'Medium', 'High'][int(np.argmax(probs))]
    return outcome, pct_low, pct_medium, pct_high

# ── Advice engine ─────────────────────────────────────────────────────────────
def _build_advice(v: dict) -> list[str]:
    advice = []
    if v['coughing_blood'] > 5:
        advice.append('You reported coughing blood — please see a doctor as soon as possible.')
    if v['smoking'] >= 8:
        advice.append('Heavy smoking is one of the strongest risk factors for lung cancer. Quitting is the single most impactful step you can take.')
    elif v['smoking'] > 5:
        advice.append('Smoking is a major risk factor for lung cancer. Consider speaking to your doctor about cessation programmes.')
    elif v['smoking'] > 3:
        advice.append('Reducing smoking frequency can meaningfully lower your long-term lung cancer risk.')
    if v['obesity'] > 6:
        advice.append('Maintaining a healthy weight supports overall health and may reduce cancer risk.')
    if v['fatigue'] > 7:
        advice.append('Persistent fatigue can signal a variety of conditions. Consult a doctor if it has lasted more than two weeks.')
    if v['air_pollution'] > 6:
        advice.append('High air pollution exposure is a known environmental risk factor. Consider using air purifiers indoors and checking local air-quality indices.')
    if v['passive_smoker'] > 6:
        advice.append('Regular exposure to second-hand smoke raises risk significantly. Try to limit time in heavily smoky environments.')
    if v['genetic_risk'] > 6:
        advice.append('A strong family history of lung cancer warrants regular screening conversations with your physician.')
    if v['balanced_diet'] < 3:
        advice.append('A diet rich in fruits and vegetables is associated with lower cancer risk. Consider consulting a nutritionist.')
    if not advice:
        advice.append('Your reported risk factors appear relatively low. Maintaining a healthy lifestyle is your best long-term protection.')
    return advice

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method != 'POST':
        return redirect(url_for('home'))

    if model is None or scaler is None:
        flash('The prediction model is not available. Please contact the administrator.', 'error')
        return redirect(url_for('form'))

    try:
        values, errors = _validate_form(request.form)
        if errors:
            for err in errors:
                flash(err, 'error')
            return redirect(url_for('form'))

        outcome, pct_low, pct_medium, pct_high = _predict(values)

        share_name   = request.form.get('share_name')
        patient_name = (request.form.get('patient_name') or '').strip()

        # CSV persistence — non-fatal: read-only filesystems (e.g. Render free) skip silently
        patient_id = 'anonymous'
        try:
            next_index = _next_index()
            patient_id = patient_name if (share_name == 'yes' and patient_name) else str(next_index)
            row = [next_index, patient_id,
                   values['age'], values['gender'],
                   *[values[f] for f in _SCORE_FIELDS],
                   outcome]
            _append_row(row)
            logger.info('Saved result for patient_id=%s outcome=%s', patient_id, outcome)
        except Exception as csv_err:
            logger.warning('Could not save to CSV (read-only fs?): %s', csv_err)
            patient_id = patient_name if (share_name == 'yes' and patient_name) else 'guest'

        advice = _build_advice(values)

        return render_template(
            'results.html',
            outcome=outcome,
            percent_low=pct_low,
            percent_medium=pct_medium,
            percent_high=pct_high,
            advice=advice,
            patient_id=patient_id,
            timestamp=datetime.now().strftime('%d %b %Y, %H:%M'),
        )
    except Exception as e:
        logger.exception('Error in /results: %s', e)
        flash('Something went wrong processing your results. Please try again.', 'error')
        return redirect(url_for('form'))

@app.route('/history/<patient_id>')
def history(patient_id):
    patient_results = _read_patient(patient_id)
    return render_template('history.html', results=patient_results, patient_id=patient_id)

@app.route('/history/<patient_id>/delete/<int:record_index>', methods=['POST'])
def delete_record(patient_id, record_index):
    """Delete a specific record by its Index value."""
    _ensure_csv_header()
    rows = []
    with open(CSV_PATH, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if not (r.get('Patient Id') == patient_id and int(r.get('Index', -1)) == record_index)]

    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    flash('Record deleted.', 'info')
    return redirect(url_for('history', patient_id=patient_id))

@app.route('/download_pdf/<patient_id>')
def download_pdf(patient_id):
    patient_results = _read_patient(patient_id)
    if not patient_results:
        flash('No records found for that patient ID.', 'error')
        return redirect(url_for('home'))

    rendered = render_template('history.html', results=patient_results, patient_id=patient_id)

    # Try pdfkit (wkhtmltopdf) first, fall back to WeasyPrint, then plain HTML
    pdf_bytes = None
    try:
        import pdfkit
        _wp = shutil.which('wkhtmltopdf')
        if not _wp and platform.system().lower().startswith('win'):
            _default = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
            _wp = _default if os.path.exists(_default) else None
        cfg = pdfkit.configuration(wkhtmltopdf=_wp) if _wp else None
        pdf_bytes = pdfkit.from_string(rendered, False, configuration=cfg)
    except Exception:
        pass

    if pdf_bytes is None:
        try:
            from weasyprint import HTML
            pdf_bytes = HTML(string=rendered, base_url=request.host_url).write_pdf()
        except Exception:
            pass

    if pdf_bytes:
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'report_{patient_id}.pdf',
        )

    # Ultimate fallback: return the HTML as a downloadable file
    flash('PDF generation is unavailable. Downloading HTML report instead.', 'info')
    return send_file(
        io.BytesIO(rendered.encode()),
        mimetype='text/html',
        as_attachment=True,
        download_name=f'report_{patient_id}.html',
    )

# ── REST API ──────────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    JSON API endpoint for programmatic access.
    Accepts the same fields as the HTML form (application/json).
    Returns { outcome, percent_low, percent_medium, percent_high, advice }.
    """
    if model is None or scaler is None:
        return jsonify({'error': 'Model not available'}), 503

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({'error': 'JSON body required'}), 400

    # Re-use the same validator (it reads from a Mapping)
    values, errors = _validate_form(payload)
    if errors:
        return jsonify({'errors': errors}), 422

    outcome, pct_low, pct_medium, pct_high = _predict(values)
    return jsonify({
        'outcome': outcome,
        'percent_low': pct_low,
        'percent_medium': pct_medium,
        'percent_high': pct_high,
        'advice': _build_advice(values),
    })

# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.exception('Internal server error')
    return render_template('500.html'), 500

if __name__ == '__main__':
    _ensure_csv_header()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', '0') == '1')
