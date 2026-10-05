import os
import warnings

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, session
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-secret-change-me")

print("Loading personality models...")
models = joblib.load(os.path.join(BASE_DIR, "models.pkl"))
print("Models loaded successfully.")

TRAITS = ['conf', 'disc', 'lead', 'neuro', 'open', 'agree', 'extra']

DEMO_USER = {
    "username": os.environ.get("DEMO_USERNAME", "admin"),
    "password": os.environ.get("DEMO_PASSWORD", "1234"),
    "name": "Demo User"
}

KEYWORD_MAP = {
    'conf': ['lead', 'decide', 'confident', 'bold', 'speak', 'face', 'direct', 'handle'],
    'disc': ['plan', 'organize', 'schedule', 'responsible', 'finish', 'honest', 'return', 'complete'],
    'lead': ['guide', 'lead', 'team', 'manage', 'coordinate', 'support all', 'take charge'],
    'neuro': ['afraid', 'panic', 'confused', 'stress', 'nervous', 'fear'],
    'open': ['new', 'learn', 'creative', 'explore', 'different', 'idea', 'change', 'risk'],
    'agree': ['help', 'kind', 'share', 'support', 'care', 'respect', 'fair', 'friendly'],
    'extra': ['talk', 'people', 'group', 'social', 'meet', 'speak', 'network', 'introduce']
}

BASE_FEATURES = [4] * 15

PERSONA_NAMES = {
    'conf': 'Decisive Thinker',
    'disc': 'Reliable Planner',
    'lead': 'Natural Coordinator',
    'neuro': 'Emotionally Sensitive',
    'open': 'Curious Explorer',
    'agree': 'Supportive Helper',
    'extra': 'Social Initiator'
}

IMPROVEMENT_TIPS = {
    'confidence': [
        'Practice speaking in front of a mirror for 5 minutes daily.',
        'Take one small decision quickly each day without overthinking.',
        'Join group discussions and share at least one opinion.'
    ],
    'discipline': [
        'Use a simple daily to-do list with 3 priority tasks.',
        'Set a fixed study or work start time every day.',
        'Break large goals into small tasks and complete them one by one.'
    ],
    'leadership': [
        'Volunteer to coordinate one small team activity each week.',
        'Practice giving clear instructions in group work.',
        'Learn to listen first, then guide the group calmly.'
    ],
    'neuroticism': [
        'Use deep breathing for 2 minutes before reacting under pressure.',
        'Write down stressful thoughts instead of holding them inside.',
        'Sleep and routine management can improve emotional balance.'
    ],
    'openness': [
        'Try one new skill, place, or activity every week.',
        'Read or watch something outside your normal interests.',
        'Challenge yourself to explore more than one solution.'
    ],
    'agreeableness': [
        'Practice active listening before giving your own answer.',
        'Offer help in one small situation each day.',
        'Use calm and respectful words during disagreement.'
    ],
    'extroversion': [
        'Start one short conversation with a new person when possible.',
        'Take part in team events instead of staying fully silent.',
        'Practice introducing yourself clearly and confidently.'
    ]
}


def score_text_to_feature(text, default_value=3):
    if not text:
        return default_value

    lower = text.lower()
    score = default_value

    positive_words = [
        'help', 'save', 'lead', 'plan', 'calm', 'honest', 'return', 'support',
        'confident', 'solve', 'organize', 'team', 'fair', 'learn', 'explore'
    ]
    negative_words = [
        'panic', 'avoid', 'hide', 'angry', 'leave', 'fear', 'afraid', 'confused'
    ]

    for word in positive_words:
        if word in lower:
            score += 0.25

    for word in negative_words:
        if word in lower:
            score -= 0.25

    return max(1, min(5, round(score)))


def enrich_trait_scores_from_text(text, trait_scores):
    if not text:
        return trait_scores

    lower = text.lower()
    for trait, keywords in KEYWORD_MAP.items():
        for keyword in keywords:
            if keyword in lower:
                trait_scores[trait] += 3

    if 'not sure' in lower or 'confused' in lower or 'afraid' in lower:
        trait_scores['conf'] -= 4
        trait_scores['neuro'] += 6

    return trait_scores


def get_tip(score, trait):
    if score >= 80:
        return f"Excellent {trait}! You show this trait strongly."
    elif score >= 60:
        return f"Good {trait}. You show a healthy level here."
    elif score >= 40:
        return f"Moderate {trait}. There is room to improve further."
    return f"Low {trait}. This may need more development and reflection."


def get_neuro_tip(score):
    if score <= 20:
        return "Very emotionally stable. You seem calm in pressure situations."
    elif score <= 40:
        return "Emotionally balanced in most situations."
    elif score <= 60:
        return "Moderate emotional sensitivity."
    elif score <= 80:
        return "Higher stress sensitivity. Calm decision practice may help."
    return "Very high emotional reactivity. Stress handling exercises are recommended."


def build_persona(scores):
    ranking = sorted([
        ('conf', scores['conf']),
        ('disc', scores['disc']),
        ('lead', scores['lead']),
        ('open', scores['open']),
        ('agree', scores['agree']),
        ('extra', scores['extra']),
        ('neuro', 100 - scores['neuro'])
    ], key=lambda x: x[1], reverse=True)
    return f"{PERSONA_NAMES[ranking[0][0]]} with {PERSONA_NAMES[ranking[1][0]]}"


def build_changes_needed(scores):
    mapping = [
        ('confidence', scores['conf']),
        ('discipline', scores['disc']),
        ('leadership', scores['lead']),
        ('neuroticism', 100 - scores['neuro']),
        ('openness', scores['open']),
        ('agreeableness', scores['agree']),
        ('extroversion', scores['extra'])
    ]
    mapping = sorted(mapping, key=lambda x: x[1])[:3]
    result = []
    for trait, _ in mapping:
        result.append({
            'trait': trait.title(),
            'how_to_change': IMPROVEMENT_TIPS[trait]
        })
    return result


@app.route('/')
def home():
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/health')
def health():
    return jsonify(status='ok', models_loaded=len(models)), 200


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = (data.get('username') or '').strip()
    password = (data.get('password') or '').strip()

    if username == DEMO_USER['username'] and password == DEMO_USER['password']:
        session['logged_in'] = True
        session['username'] = username
        session['name'] = DEMO_USER['name']
        session.setdefault('attempts', 0)
        return jsonify({
            'success': True,
            'name': DEMO_USER['name'],
            'message': 'Login successful'
        })

    return jsonify({'success': False, 'message': 'Invalid username or password'}), 401


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('logged_in'):
        return jsonify({'error': 'Please login first'}), 401

    payload = request.get_json()
    if not payload:
        return jsonify({'error': 'No input data received'}), 400

    answers = payload.get('answers') if isinstance(payload, dict) else payload
    if not isinstance(answers, list) or len(answers) != 15:
        return jsonify({'error': 'Exactly 15 answers are required'}), 400

    feature_values = []
    text_blob = []
    total_points = 0

    for i, answer in enumerate(answers):
        if isinstance(answer, dict):
            answer_type = answer.get('type', 'choice')
            if answer_type == 'other':
                text = (answer.get('text') or '').strip()
                value = score_text_to_feature(text, BASE_FEATURES[i] if i < len(BASE_FEATURES) else 3)
                feature_values.append(value)
                total_points += int(value)
                if text:
                    text_blob.append(text)
            else:
                value = float(answer.get('value', BASE_FEATURES[i] if i < len(BASE_FEATURES) else 3))
                feature_values.append(value)
                total_points += int(value)
        else:
            value = float(answer)
            feature_values.append(value)
            total_points += int(value)

    X = np.array(feature_values).reshape(1, -1)

    scores = {}
    for trait in TRAITS:
        scores[trait] = int(models[trait].predict_proba(X)[0][1] * 100)

    combined_text = ' '.join(text_blob)
    scores = enrich_trait_scores_from_text(combined_text, scores)

    for key in scores:
        scores[key] = int(max(0, min(100, scores[key])))

    session['attempts'] = session.get('attempts', 0) + 1
    attempt_number = session['attempts']
    persona = build_persona(scores)
    changes_needed = build_changes_needed(scores)

    return jsonify({
        'confidence': scores['conf'],
        'discipline': scores['disc'],
        'leadership': scores['lead'],
        'neuroticism': scores['neuro'],
        'openness': scores['open'],
        'agreeableness': scores['agree'],
        'extroversion': scores['extra'],
        'persona': persona,
        'feature_values': feature_values,
        'total_points': total_points,
        'max_points': 75,
        'average_points': round(total_points / 15, 2),
        'attempt_number': attempt_number,
        'changes_needed': changes_needed,
        'suggestions': {
            'confidence': get_tip(scores['conf'], 'confidence'),
            'discipline': get_tip(scores['disc'], 'discipline'),
            'leadership': get_tip(scores['lead'], 'leadership'),
            'neuroticism': get_neuro_tip(scores['neuro']),
            'openness': get_tip(scores['open'], 'openness'),
            'agreeableness': get_tip(scores['agree'], 'agreeableness'),
            'extroversion': get_tip(scores['extra'], 'extroversion')
        }
    })


if __name__ == '__main__':
    print('Starting Scenario Personality Analyzer on http://127.0.0.1:5000')
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5000)
