"""Daily-life advice based on the patient's inputs and predicted risk.
Pure Python (no Streamlit) so it can be tested and reused.
General lifestyle guidance only - NOT a medical prescription."""

EMERGENCY_NUMBER = "112"   # change for your country

WARNING_SIGNS = [
    "Chest pain, pressure, tightness or burning that lasts more than a few minutes or keeps coming back",
    "Pain or discomfort spreading to the left arm, shoulder, neck, jaw or back",
    "Sudden shortness of breath, even at rest",
    "Cold sweat, nausea or vomiting together with chest discomfort",
    "Sudden dizziness, fainting, or a racing / irregular heartbeat with weakness",
    "Unusual, sudden tiredness that is new for you",
]

EMERGENCY_STEPS = [
    f"Stop what you are doing, sit down and stay calm.",
    f"Call your local emergency number ({EMERGENCY_NUMBER}) immediately.",
    "Do NOT drive yourself. Ask someone to stay with you.",
    "Follow any emergency instructions your own doctor has already given you.",
]


# ------------------------------------------------------------------ risk band
def risk_band(probability: float, threshold: float = 0.5) -> dict:
    """Low / Moderate / High based on the model's tuned threshold."""
    high_cut = min(0.95, max(0.65, threshold + 0.15))
    if probability < threshold:
        return {"level": 0, "label": "Low", "emoji": "🟢",
                "summary": "The model sees a lower chance of heart disease. Keep your healthy habits going."}
    if probability < high_cut:
        return {"level": 1, "label": "Moderate", "emoji": "🟠",
                "summary": "Some risk signs are present. Lifestyle changes plus a doctor check-up are a good idea."}
    return {"level": 2, "label": "High", "emoji": "🔴",
            "summary": "Several risk signs are present. Please see a doctor (preferably a cardiologist) soon."}


# ------------------------------------------------------------------ personalised findings
def personal_findings(p: dict) -> list:
    """One entry per thing in the patient's numbers that deserves attention."""
    f = []

    bp = p["trestbps"]
    if bp >= 140:
        f.append({"icon": "🩸", "title": "Resting blood pressure", "value": f"{bp} mmHg (high)",
                  "why": "Persistently high blood pressure makes the heart work harder and damages arteries over time.",
                  "actions": ["Keep salt under 5 g (about 1 teaspoon) a day: go easy on pickles, papad, chips, namkeen, packaged food.",
                              "Measure BP at home at the same time daily and note it down for your doctor.",
                              "Follow a DASH-style diet: fruits, vegetables, low-fat dairy, whole grains, nuts.",
                              "If readings stay at or above 140/90, see your doctor. Do not start or stop BP medicines on your own."]})
    elif bp >= 130:
        f.append({"icon": "🩸", "title": "Resting blood pressure", "value": f"{bp} mmHg (slightly high)",
                  "why": "This is in the 'elevated' range. It is the best time to fix it with lifestyle alone.",
                  "actions": ["Cut down on salt and processed food.",
                              "Walk briskly 30 minutes most days and keep your weight in a healthy range.",
                              "Check your BP regularly (at least once a month)."]})

    ch = p["chol"]
    if ch >= 240:
        f.append({"icon": "🧈", "title": "Cholesterol", "value": f"{ch} mg/dl (high)",
                  "why": "High cholesterol can build up plaque in the arteries that feed the heart.",
                  "actions": ["Avoid trans fats and cut saturated fat: less deep-fried food, bakery items, vanaspati, and excess ghee/butter/red meat.",
                              "Eat more fibre: oats, dal, beans, sprouts, fruits, vegetables. Add a handful of nuts daily.",
                              "Ask your doctor for a full lipid profile (LDL, HDL, triglycerides) and whether you need medicine."]})
    elif ch >= 200:
        f.append({"icon": "🧈", "title": "Cholesterol", "value": f"{ch} mg/dl (borderline)",
                  "why": "Borderline cholesterol usually improves well with diet and exercise.",
                  "actions": ["Reduce fried and oily food; choose whole grains over refined flour.",
                              "Repeat a lipid profile in 3 to 6 months."]})

    if p["fbs"] == 1:
        f.append({"icon": "🍬", "title": "Fasting blood sugar", "value": "above 120 mg/dl",
                  "why": "High blood sugar (or diabetes) damages blood vessels and strongly raises heart risk.",
                  "actions": ["Cut sugary drinks, sweets and refined carbs (maida, white bread, white rice in large portions).",
                              "Walk for 10 to 15 minutes after meals; it lowers blood sugar.",
                              "Get an HbA1c and fasting glucose test and discuss the result with your doctor."]})

    if p["exang"] == 1:
        f.append({"icon": "🏃", "title": "Chest pain during exercise", "value": "present",
                  "why": "Pain or tightness on exertion can mean the heart is not getting enough blood during effort.",
                  "actions": ["Do NOT start hard exercise before a doctor has checked you (ECG / stress test).",
                              "Stop immediately and rest if you get chest pain, breathlessness or dizziness.",
                              "Ask your doctor about a supervised programme such as cardiac rehabilitation."]})

    if p["oldpeak"] >= 1.0:
        f.append({"icon": "📉", "title": "ST depression (oldpeak)", "value": f"{p['oldpeak']}",
                  "why": "ST depression on an exercise ECG can be a sign that the heart muscle is short of blood during stress.",
                  "actions": ["Show this ECG result to a cardiologist and ask whether more tests are needed."]})

    if p["ca"] >= 1:
        f.append({"icon": "🫀", "title": "Major vessels highlighted", "value": f"{p['ca']}",
                  "why": "A higher number here usually points to narrowing in more of the heart's blood vessels.",
                  "actions": ["Regular follow-up with a cardiologist is important; keep all scheduled appointments."]})

    if p["restecg"] != 0:
        f.append({"icon": "📈", "title": "Resting ECG", "value": f"code {p['restecg']} (not normal)",
                  "why": "The resting ECG was not fully normal.",
                  "actions": ["Keep your ECG reports and discuss them with your doctor at the next visit."]})

    if p["age"] >= 50:
        f.append({"icon": "🎂", "title": "Age", "value": f"{p['age']} years",
                  "why": "Heart risk naturally rises with age, so regular screening matters more.",
                  "actions": ["Do a yearly check-up: BP, lipid profile, fasting sugar, weight and waist."]})
    return f


def urgent_flags(p: dict, band: dict) -> list:
    flags = []
    if band["level"] == 2:
        flags.append("Your predicted risk is HIGH")
    if p["exang"] == 1:
        flags.append("you get chest pain with exercise")
    if p["trestbps"] >= 180:
        flags.append("your blood pressure is very high (180 or more)")
    if p["oldpeak"] >= 2.0:
        flags.append("your ST depression is marked (2.0 or more)")
    return flags


# ------------------------------------------------------------------ plans
def exercise_plan(p: dict, band: dict) -> dict:
    max_hr = 220 - p["age"]
    lo, hi = round(0.5 * max_hr), round(0.7 * max_hr)
    needs_clearance = p["exang"] == 1 or band["level"] == 2
    if needs_clearance:
        plan = ["Get your doctor's OK (ECG / stress test) BEFORE starting any regular exercise.",
                "Until then: easy walking for 10 to 15 minutes at a comfortable pace, on flat ground.",
                "Add 5 minutes every week only if you feel completely fine."]
    else:
        plan = ["Aim for 150 minutes of moderate activity a week, e.g. 30 min brisk walking, 5 days a week.",
                "You can split it: 3 walks of 10 minutes after meals.",
                "Add muscle-strengthening (bodyweight, light weights, resistance bands) on 2 days a week.",
                "Warm up for 5 minutes and cool down for 5 minutes."]
    plan += ["Sit less: stand up and move for 2 to 3 minutes every hour.",
             "Use the 'talk test': you should be able to talk but not sing while exercising."]
    return {"needs_clearance": needs_clearance, "plan": plan, "hr_zone": (lo, hi),
            "hr_note": "Estimated from 220 minus age. It is not valid if you take heart-rate-lowering medicines "
                       "(e.g. beta blockers). Ask your doctor for your own target."}


FOOD_DO = [
    "Half your plate vegetables and salad; 2 to 3 servings of fruit a day.",
    "Whole grains (whole wheat roti, oats, brown rice, millets such as ragi/jowar/bajra).",
    "Protein from dal, beans, chana, sprouts, curd, fish, eggs or lean meat.",
    "A small handful of nuts (almonds, walnuts) and seeds daily.",
    "Cook in small amounts of oil (mustard, groundnut, olive); bake, steam, grill or roast instead of deep frying.",
    "Drink enough water; choose water, buttermilk, or unsweetened tea over sugary drinks.",
]
FOOD_AVOID = [
    "Deep-fried snacks, bakery items, and foods with trans fat / vanaspati.",
    "Too much salt: pickles, papad, chips, instant noodles, packaged soups.",
    "Sugary drinks, sweets and large portions of refined flour (maida).",
    "Processed and red meat in large amounts.",
    "Heavy late-night dinners; finish eating 2 to 3 hours before sleep.",
]
LIFESTYLE = [
    "Tobacco: do not smoke and avoid chewing tobacco or second-hand smoke. Quitting is the single biggest step.",
    "Alcohol: best avoided; if you drink, keep it minimal.",
    "Sleep 7 to 8 hours. Poor sleep raises BP and sugar.",
    "Stress: 10 minutes of deep breathing, walking, prayer/meditation or a hobby each day.",
    "Weight: aim for a healthy weight and a smaller waistline; even losing 5% of body weight helps.",
    "Take prescribed medicines exactly as advised. Never stop them on your own because you feel fine.",
]
MONITORING = [
    "Blood pressure: at home a few times a week (daily if it is high).",
    "Blood tests once a year (or as advised): lipid profile, fasting sugar / HbA1c.",
    "Weight and waist: once a week.",
    "Keep a simple diary of BP, weight, symptoms and medicines, and take it to every doctor visit.",
    "Follow up with a doctor or cardiologist as advised, more often if your risk is moderate or high.",
]

DAILY_ROUTINE = {
    "🌅 Morning": ["Drink a glass of water after waking up.",
                   "5 to 10 minutes of stretching and deep breathing.",
                   "Take your medicines as prescribed; check BP if your doctor asked you to.",
                   "Breakfast: oats / poha / vegetable upma / eggs with fruit. Avoid fried and sugary options."],
    "☀️ Daytime": ["Lunch: half plate vegetables, one part dal/protein, one part whole grain. Go easy on salt.",
                   "Walk 10 minutes after meals.",
                   "Move every hour; avoid sitting for long stretches.",
                   "Snack on fruit, nuts, sprouts or roasted chana instead of namkeen."],
    "🌙 Evening & night": ["Main exercise / brisk walk if your doctor has cleared you.",
                           "Light dinner 2 to 3 hours before bed.",
                           "10 minutes of relaxation; reduce screens before sleep.",
                           "Sleep 7 to 8 hours. No tobacco or alcohol."],
}

CHECKLIST = [
    "Walked / exercised as planned today",
    "Ate at least 5 servings of vegetables and fruit",
    "No tobacco, little or no alcohol",
    "Kept salt, fried food and sugar low",
    "Took my medicines on time (if prescribed)",
    "Did 10 minutes of relaxation / breathing",
    "Slept (or plan to sleep) 7 to 8 hours",
]


# ------------------------------------------------------------------ main entry
def build_advice(patient: dict, probability: float, threshold: float = 0.5) -> dict:
    band = risk_band(probability, threshold)
    return {
        "band": band,
        "findings": personal_findings(patient),
        "urgent": urgent_flags(patient, band),
        "exercise": exercise_plan(patient, band),
        "food_do": FOOD_DO, "food_avoid": FOOD_AVOID,
        "lifestyle": LIFESTYLE, "monitoring": MONITORING,
        "routine": DAILY_ROUTINE, "checklist": CHECKLIST,
        "warning_signs": WARNING_SIGNS, "emergency_steps": EMERGENCY_STEPS,
    }