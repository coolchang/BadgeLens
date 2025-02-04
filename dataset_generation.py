import random
import pandas as pd
from faker import Faker

fake = Faker()

NUM_RECORDS = 10000

# Sample information for badge data
badge_names = [
    "Python Certification", "Data Science Expert", "AI Specialist",
    "Machine Learning Pro", "Deep Learning Master", "Cloud Computing Guru",
    "Big Data Analyst", "Cybersecurity Professional", "DevOps Engineer", "UI/UX Designer"
]
issuers = ["Coursera", "Google", "IBM", "Udacity", "edX"]
learning_opportunities = [
    "AI Professional Course", "Data Science Bootcamp", "Cloud Computing Course",
    "Big Data Analysis Program", "Intro to Machine Learning"
]
criteria_templates = [
    "Completion of 40 hours of lectures and project submission",
    "Passing all module assessments",
    "Completion of hands-on project and final evaluation"
]
employment_outcomes = [
    "Eligible for Machine Learning Engineer positions",
    "Eligible for Data Analyst roles",
    "Eligible for Software Engineer roles",
    "Eligible for Cloud Architect positions"
]
alignments = ["Industry Standard Alignment", "Academic Alignment", "Professional Alignment"]
skills_pool = [
    "Python", "Data Analysis", "Machine Learning", "Deep Learning",
    "SQL", "Cloud Computing", "DevOps", "UI/UX", "Cybersecurity"
]
competency_pool = ["Problem Solving", "Analytical Skills", "Creativity", "Teamwork", "Leadership"]

# Sample information for user data
user_goals = [
    "Become a Data Scientist", "Become a Software Engineer", "Become an AI Expert",
    "Become a Machine Learning Researcher", "Become a Cloud Engineer"
]
competency_levels = ["Beginner", "Intermediate", "Advanced"]
education_levels = ["High School", "Bachelor's Degree", "Master's Degree", "PhD"]
engagement_levels = ["Highly Active", "Moderate", "Low"]

# Generate badge dataset
badge_list = []
for i in range(1, NUM_RECORDS + 1):
    badge_id = f"B{i:05d}"
    name = random.choice(badge_names)
    issuer = random.choice(issuers)
    description = f"{name} badge issued by {issuer}"
    criteria = random.choice(criteria_templates)
    alignment = random.choice(alignments)
    employmentOutcome = random.choice(employment_outcomes)
    skillsValidated = random.sample(skills_pool, k=random.randint(1, 3))
    competency = random.sample(skills_pool + competency_pool, k=random.randint(1, 3))
    learningOpportunity = random.choice(learning_opportunities)
    
    # Assign related badges randomly ensuring no duplicates and not self-referential
    related_badges = []
    for _ in range(random.randint(0, 2)):
        rel_id = f"B{random.randint(1, NUM_RECORDS):05d}"
        if rel_id != badge_id and rel_id not in related_badges:
            related_badges.append(rel_id)
    
    badge_list.append({
        "badge_id": badge_id,
        "name": name,
        "issuer": issuer,
        "description": description,
        "criteria": criteria,
        "alignment": alignment,
        "employmentOutcome": employmentOutcome,
        "skillsValidated": skillsValidated,
        "competency": competency,
        "learningOpportunity": learningOpportunity,
        "related_badges": related_badges
    })

badge_df = pd.DataFrame(badge_list)

# Generate user dataset
user_list = []
for i in range(1, NUM_RECORDS + 1):
    user_id = f"U{i:05d}"
    name = fake.name()
    email = fake.email()
    goal = random.choice(user_goals)
    skills = random.sample(skills_pool, k=random.randint(1, 4))
    competency_level = random.choice(competency_levels)
    acquired_badges = random.sample([f"B{j:05d}" for j in range(1, NUM_RECORDS + 1)], k=random.randint(0, 5))
    learning_history = f"Completed the '{random.choice(badge_names)}' course at {random.choice(issuers)}"
    employment_history = f"Worked at {fake.company()} for {random.randint(1, 5)} years"
    education_level = random.choice(education_levels)
    engagement_metrics = random.choice(engagement_levels)
    recommendation_history = random.sample([f"B{j:05d}" for j in range(1, NUM_RECORDS + 1)], k=random.randint(0, 5))
    
    user_list.append({
        "user_id": user_id,
        "name": name,
        "email": email,
        "goal": goal,
        "skills": skills,
        "competency_level": competency_level,
        "acquired_badges": acquired_badges,
        "learning_history": learning_history,
        "employment_history": employment_history,
        "education_level": education_level,
        "engagement_metrics": engagement_metrics,
        "recommendation_history": recommendation_history
    })

user_df = pd.DataFrame(user_list)

# Create relationship dataset
# We'll extract relationships from the acquired_badges and recommendation_history fields in the user dataset.
relationships = []
for user in user_list:
    uid = user["user_id"]
    # Acquired badges relationship
    for bid in user["acquired_badges"]:
        relationships.append({
            "user_id": uid,
            "badge_id": bid,
            "relationship": "acquired"
        })
    # Recommended badges relationship
    for bid in user["recommendation_history"]:
        relationships.append({
            "user_id": uid,
            "badge_id": bid,
            "relationship": "recommended"
        })

relationship_df = pd.DataFrame(relationships)

# Save datasets to independent Excel files
badge_df.to_excel("openbadge_dataset.xlsx", index=False)
user_df.to_excel("user_dataset.xlsx", index=False)
relationship_df.to_excel("relationship_dataset.xlsx", index=False)

print("Three Excel files have been created:")
print("1. openbadge_dataset.xlsx")
print("2. user_dataset.xlsx")
print("3. relationship_dataset.xlsx")
