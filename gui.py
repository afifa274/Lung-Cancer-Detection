import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import csv
import os
import pdfkit
import shutil
import platform


path_wkhtmltopdf = shutil.which('wkhtmltopdf')
if not path_wkhtmltopdf and platform.system().lower().startswith('win'):
    default_windows_path = r'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'
    if os.path.exists(default_windows_path):
        path_wkhtmltopdf = default_windows_path

config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf) if path_wkhtmltopdf else None

app = Flask(__name__)

model = load_model('cancer_snus_model.h5')

scaler_file = 'scaler.pkl'
with open(scaler_file, 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        # Extract the form data
        age = int(request.form.get('age'))
        gender = request.form.get('gender')
        gender = 1 if gender.lower() == 'male' else 2
        air_pollution = int(request.form.get('air_pollution'))
        alcohol_use = int(request.form.get('alcohol_use'))
        dust_allergy = int(request.form.get('dust_allergy'))
        occupational_hazards = int(request.form.get('occupational_hazards'))
        genetic_risk = int(request.form.get('genetic_risk'))
        chronic_lung_disease = int(request.form.get('chronic_lung_disease'))
        balanced_diet = int(request.form.get('balanced_diet'))
        obesity = int(request.form.get('obesity'))
        smoking = int(request.form.get('smoking'))
        passive_smoker = int(request.form.get('passive_smoker'))
        chest_pain = int(request.form.get('chest_pain'))
        coughing_blood = int(request.form.get('coughing_blood'))
        fatigue = int(request.form.get('fatigue'))
        weight_loss = int(request.form.get('weight_loss'))
        shortness_of_breath = int(request.form.get('shortness_of_breath'))
        wheezing = int(request.form.get('wheezing'))
        swallowing_difficulty = int(request.form.get('swallowing_difficulty'))
        clubbing = int(request.form.get('clubbing'))
        frequent_cold = int(request.form.get('frequent_cold'))
        dry_cough = int(request.form.get('dry_cough'))
        snoring = int(request.form.get('snoring'))

        
        share_name = request.form.get('share_name')
        patient_name = request.form.get('patient_name', '').strip()

        data = np.zeros((23))
        data[0] = age
        data[1] = gender
        data[2] = air_pollution
        data[3] = alcohol_use
        data[4] = dust_allergy
        data[5] = occupational_hazards
        data[6] = genetic_risk
        data[7] = chronic_lung_disease
        data[8] = balanced_diet
        data[9] = obesity
        data[10] = smoking
        data[11] = passive_smoker
        data[12] = chest_pain
        data[13] = coughing_blood
        data[14] = fatigue
        data[15] = weight_loss
        data[16] = shortness_of_breath
        data[17] = wheezing
        data[18] = swallowing_difficulty
        data[19] = clubbing
        data[20] = frequent_cold
        data[21] = dry_cough
        data[22] = snoring

        new_data = np.array([data])
        new_data_scaled = scaler.transform(new_data)
        predictions = model.predict(new_data_scaled)[0]
        percent_low = round(predictions[0] * 100, 1)
        percent_medium = round(predictions[1] * 100, 1)
        percent_high = round(predictions[2] * 100, 1)

        predicted_class = np.argmax(predictions)

        if predicted_class == 0:
            outcome = 'Low'
        elif predicted_class == 1:
            outcome = 'Medium'
        else:
            outcome = 'High'

     
        csv_path = 'user_inputs.csv'
        if os.path.exists(csv_path):
            with open(csv_path, 'r', newline='') as f:
                lines = list(f)
                next_index = len(lines)  
        else:
            next_index = 1

        
        patient_id = patient_name if (share_name == 'yes' and patient_name) else str(next_index)

       
        row = [next_index, patient_id] + list(data) + [outcome]
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        

        advice = []
        if coughing_blood > 5:
            advice.append("Visit a doctor immediately if you have blood in your cough.")
        if smoking > 3:
            advice.append("Try to quit smoking to reduce the risk of lung cancer.")
        if smoking > 5:
            advice.append("Smoking is a major risk factor for lung cancer. Consider quitting.")
        if smoking >= 8:
            advice.append("You should quit smoking or at least minimize it as soon as possible to lower your risk.") 
        if obesity > 6:
            advice.append("Maintaining a healthy weight can reduce the risk of various diseases, including cancer.")
        if fatigue > 7:
            advice.append("Fatigue can be a symptom of various conditions. Consult a doctor if it persists.")

        return render_template(
            'results.html',
            outcome=outcome,
            percent_low=percent_low,
            percent_medium=percent_medium,
            percent_high=percent_high,
            advice=advice,
            patient_id=patient_id
        )
    else:
        return redirect('/')

@app.route('/history/<patient_id>')
def history(patient_id):
    results = []
    with open('user_inputs.csv', 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Patient Id'] == patient_id:
                results.append(row)
    return render_template('history.html', results=results, patient_id=patient_id)

@app.route('/download_pdf/<patient_id>')
def download_pdf(patient_id):
    results = []
    with open('user_inputs.csv', 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Patient Id'] == patient_id:
                results.append(row)
    rendered = render_template('history.html', results=results, patient_id=patient_id)
    pdf_kwargs = {'configuration': config} if config else {}
    pdfkit.from_string(rendered, 'report.pdf', **pdf_kwargs)
    return send_file('report.pdf', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
