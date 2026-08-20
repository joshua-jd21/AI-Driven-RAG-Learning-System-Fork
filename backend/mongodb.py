import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")

client = MongoClient(MONGODB_URI)

db = client["adaptive_learning"]

learner_profiles = db["learner_profiles"]


def save_learner_profile(profile: dict):
    learner_id = profile.get("learner_id")

    if not learner_id:
        raise ValueError("learner_id is required")

    learner_profiles.update_one(
        {"learner_id": learner_id},
        {"$set": profile},
        upsert=True
    )


def get_learner_profile(learner_id: str):
    return learner_profiles.find_one(
        {"learner_id": learner_id},
        {"_id": 0}
    )