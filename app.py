from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Create Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Sample dataset for training the category prediction model
sample_resumes = {
    'Software Developer': [
        "Experienced Python developer with Flask and Django. Proficient in web development, APIs, and databases.",
        "Full-stack developer skilled in JavaScript, React, Node.js, and backend technologies.",
        "Software engineer with expertise in C++, algorithms, and system design."
    ],
    'Data Analyst': [
        "Data analyst with experience in SQL, Python, pandas, and data visualization tools like Tableau.",
        "Business intelligence analyst skilled in Excel, Power BI, and statistical analysis.",
        "Data scientist proficient in machine learning, R, and big data technologies."
    ],
    'HR': [
        "Human resources professional with experience in recruitment, employee relations, and HR policies.",
        "Talent acquisition specialist focused on sourcing and interviewing candidates.",
        "HR manager overseeing employee development, performance management, and compliance."
    ],
    'Marketing': [
        "Digital marketing specialist with SEO, SEM, and social media expertise.",
        "Content marketer creating engaging content for blogs, websites, and campaigns.",
        "Brand manager developing marketing strategies and managing product launches."
    ]
}

# Prepare training data
X_train = []
y_train = []
for category, resumes in sample_resumes.items():
    for resume in resumes:
        X_train.append(resume)
        y_train.append(category)

# Create and train the category prediction model
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])
model.fit(X_train, y_train)

def extract_text_from_pdf(file_stream):
    """
    Extract text from a PDF file using pdfplumber.
    
    Args:
        file_stream: File-like object of the PDF
        
    Returns:
        str: Extracted text from the PDF
    """
    with pdfplumber.open(file_stream) as pdf:
        text = ''
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def calculate_match_score(resume_text, job_description):
    """
    Calculate the match score between resume text and job description using TF-IDF and cosine similarity.
    
    Args:
        resume_text (str): Text extracted from the resume
        job_description (str): Job description text
        
    Returns:
        float: Match score as percentage (0-100)
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 2)

def extract_skills(resume_text, required_skills):
    """
    Extract matched and missing skills from resume text.
    
    Args:
        resume_text (str): Text from the resume
        required_skills (str): Comma-separated skills string
        
    Returns:
        tuple: (matched_skills list, missing_skills list)
    """
    if not required_skills:
        return [], []
    
    skills_list = [skill.strip().lower() for skill in required_skills.split(',') if skill.strip()]
    resume_lower = resume_text.lower()
    
    matched = []
    missing = []
    
    for skill in skills_list:
        if skill in resume_lower:
            matched.append(skill.title())
        else:
            missing.append(skill.title())
    
    return matched, missing

def calculate_skill_score(matched_skills, total_skills):
    """
    Calculate skill match score as percentage.
    
    Args:
        matched_skills (list): List of matched skills
        total_skills (int): Total required skills
        
    Returns:
        float: Skill match percentage
    """
    if total_skills == 0:
        return 0
    return (len(matched_skills) / total_skills) * 100

def predict_category(resume_text):
    return model.predict([resume_text])[0] 
@app.route('/analyze', methods=['POST'])
def analyze():
    """
    API endpoint to analyze a resume against a job description.
    
    Expects FormData with:
        - resume (PDF file, optional)
        - pasted_text (string, optional if no file)
        - job_description (string, required)
        - hr_name (string)
        - candidate_name (string)
        - skills (comma-separated string)
        - threshold (number)
    
    Returns:
        JSON with analysis results or error message
    """
    try:
        # Get form data
        resume_file = request.files.get('resume')
        pasted_text = request.form.get('pasted_text', '').strip()
        job_description = request.form.get('job_description', '').strip()
        hr_name = request.form.get('hr_name', '').strip()
        candidate_name = request.form.get('candidate_name', '').strip()
        skills = request.form.get('skills', '').strip()
        threshold_str = request.form.get('threshold', '70')
        
        # Validate required fields
        if not job_description:
            return jsonify({'error': 'Job description is required'}), 400
        
        if not resume_file and not pasted_text:
            return jsonify({'error': 'Either resume file or pasted text must be provided'}), 400
        
        # Get resume text
        if pasted_text:
            resume_text = pasted_text
        else:
            resume_text = extract_text_from_pdf(resume_file.stream)
        
        if not resume_text.strip():
            return jsonify({'error': 'Could not extract text from resume'}), 400
        
        # Parse threshold
        try:
            threshold = float(threshold_str)
        except ValueError:
            threshold = 70.0
        
        # Calculate match score
        text_score = calculate_match_score(resume_text, job_description)
        
        # Predict category
        category = predict_category(resume_text)
        
        # Extract skills
        matched_skills, missing_skills = extract_skills(resume_text, skills)
        
        # Calculate combined score (40% text, 60% skills)
        total_required_skills = len([s for s in skills.split(',') if s.strip()])
        skill_score = calculate_skill_score(matched_skills, total_required_skills)
        match_score = round((0.4 * text_score) + (0.6 * skill_score), 2)
        
        # Determine status
        status = "PASS" if match_score >= threshold else "REJECT"
        
        # Return response
        return jsonify({
            'hr_name': hr_name,
            'candidate_name': candidate_name,
            'match_score': match_score,
            'category': category,
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'status': status
        })
    
    except Exception as e:
        # Error handling
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(debug=True, port=5000)