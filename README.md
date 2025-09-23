# LungProfilaxe — Lung Cancer Risk Assessment Assistant

Welcome to **LungProfilaxe** — an interactive web service that helps patients and clinicians estimate the probability of developing lung cancer based on a standardized questionnaire. The project combines a trained neural network, a clean user interface, and auto-generated recommendations to make screening understandable and accessible.

## 📌 Key Features

- **Accurate Prediction**: A Keras model analyzes 23 clinical and behavioral factors, normalizes them, and outputs risk probabilities across three classes (Low / Medium / High).
- **Modern Interface**: Pages _Home_, _Assessment_, _Results_, and _History_ are styled consistently with cards, tooltips, and smooth risk indicators.
- **History Tracking**: Saved questionnaires are grouped per patient, displayed in scrollable cards, and visualized with interactive Chart.js graphs to show risk dynamics.
- **PDF Export**: Generate patient reports in one click. The app auto-detects `wkhtmltopdf` and creates a `report.pdf`.
- **Personalized Advice**: Based on the most concerning factors, the app generates tailored recommendations for next steps.

## 🧠 Model Details

1. **Data**: Uses the `cancer patient data sets.csv` dataset, where each record contains a patient ID, demographics, and answers on a 1–9 scale.
2. **Preprocessing**: Features are standardized with `StandardScaler`. The fitted scaler is stored as `scaler.pkl` and reused in the web app.
3. **Architecture**: A sequential neural net (Dense → Dense → Dense) trained for 10 epochs with Adam optimizer and `binary_crossentropy` in a one-hot setup.
4. **Inference**: In `gui.py` the model (`cancer_snus_model.h5`) loads, accepts normalized answers, and returns class probabilities.

To retrain, run `model.py` after updating the dataset. It will overwrite both the model weights and scaler.

## 🖥️ Interface Pages

- **Home** — Introduces the service, explains the scoring system, and guides the user to the assessment.
- **Assessment** — A multi-page form grouped by lifestyle, symptoms, and background factors, with tooltips and a 1–9 scale.
- **Results** — Displays the risk badge, animated progress bars, and tailored recommendations.
- **History** — Scrollable table of past assessments and an adaptive Chart.js risk trend graph.

## ⚙️ Installation & Run

```bash
# 1. (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install wkhtmltopdf for PDF export
#   macOS: brew install --cask wkhtmltopdf
#   Ubuntu/Debian: sudo apt install wkhtmltopdf
#   Windows: download installer from https://wkhtmltopdf.org/downloads.html

# 4. Run the web app
python gui.py
The app will be available at http://127.0.0.1:5000/. Results are saved in user_inputs.csv so history can be reviewed or exported later.

 Project Structure
 .
├── gui.py                     # Flask app and routes
├── model.py                   # Training script for the neural net
├── cancer_snus_model.h5       # Trained model weights
├── scaler.pkl                 # Feature scaler
├── user_inputs.csv            # Collected user answers
├── static/
│   ├── css/main.css           # Global styles
│   └── images/favicon.svg     # App icon
└── templates/
    ├── base.html              # Base template and navigation
    ├── home.html              # Home page
    ├── form.html              # Assessment form
    ├── results.html           # Results screen
    └── history.html           # History table and chart
⚠️ cancer_snus_model.h5 and scaler.pkl must be placed in the same folder as gui.py. If you clone the repo fresh, make sure artifacts are copied correctly.

🧪 Useful Commands

python -m compileall . — quickly check syntax for all Python files.

python model.py — retrain the model after dataset update.

python gui.py — launch the web interface.

🙌 Acknowledgements

The dataset was provided by the authors of the Lung Cancer Patient Data Sets. Thanks for making this data public, enabling the creation of this interactive tool.

If you spot issues or want to suggest improvements, feel free to open an issue or submit a pull request — contributions are welcome!
```
