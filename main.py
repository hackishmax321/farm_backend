# main.py
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import bcrypt
import os
import shutil
from typing import List, Optional, Dict, Union
from firestore_db import get_firestore_client
import joblib
import pandas as pd
from google.cloud import firestore
from datetime import datetime, timedelta
from collections import Counter
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import uuid
import firebase_admin
from firebase_admin import credentials, db as FireDB
import traceback
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import requests
from inference_sdk import InferenceHTTPClient
from typing import Literal

app = FastAPI()
origins = [
    "http://localhost:3000",
    "http://localhost:3001"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load MOdels Health
MODEL_HEALTH = joblib.load('cow_health_model.joblib')

# Growth Model
MODEL_GROWTH = joblib.load('model_growth_in_weight.joblib')

# Feeding Patter
MODEL_FEED = joblib.load('feeding_pattern_model.joblib')

# Load Breed detection Model
MODEL_BREED = load_model("model_breed_detect.h5")
CLASS_BREED = ['Ayrshire', 'Friesian', 'Jersey', 'Sahiwal', 'Local Lankan (Lankan White)', 'Zebu']

# Load Pest detection Models
MODEL_PESTS = load_model("model_pests_detect.h5")
CLASS_PESTS = ['Mastitis', ' Tick Infestation', 'Dermatophytosis (RINGWORM)', 'Fly Strike (MYIASIS)', 'Foot and Mouth disease', 'Lumpy Skin', 'Black Quarter (BQ)', 'Parasitic Mange']

# Db connection
db = get_firestore_client()
cred = credentials.Certificate("cattletracking-d9d96-firebase-adminsdk.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://cattletracking-d9d96-default-rtdb.firebaseio.com/"  # Your Firebase Realtime Database URL
})

class User(BaseModel):
    username: str
    full_name: str
    email:str
    contact: str
    password: str
    nic: str

class LoginUser(BaseModel):
    username: str
    password: str


users_db = {}

@app.post("/register")
async def register_user(user: User):
    user_ref = db.collection("users").document(user.username)
    if user_ref.get().exists:
        raise HTTPException(status_code=400, detail="Username already registered")

    # Hash the password before storing it
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    user_data = user.dict()
    user_data["password"] = hashed_password.decode('utf-8')

    user_ref.set(user_data)
    return {"message": "User registered successfully", "user": user_data}

@app.post("/login")
async def login_user(user: LoginUser):
    user_ref = db.collection("users").document(user.username)
    user_doc = user_ref.get()

    if not user_doc.exists:
        raise HTTPException(status_code=400, detail="Invalid username or password")

    user_data = user_doc.to_dict()
    
    # Check the hashed password
    if not bcrypt.checkpw(user.password.encode('utf-8'), user_data["password"].encode('utf-8')):
        raise HTTPException(status_code=400, detail="Invalid username or password")

    user_data.pop("password")  # Remove the password field from the response

    return {"message": "Login successful", "user": user_data}

# Adding Cattles / Animals

class Animal(BaseModel):
    name: str
    owner: str
    type: str
    dob: str
    gender: str
    milk_ability: bool
    status: str
    health: str
    image: str 

@app.post("/add-animal")
async def add_animal(animal:Animal):
    # Verify owner exists
    owner_ref = db.collection("users").document(animal.owner)
    # if not owner_ref.get().exists:
    #     raise HTTPException(status_code=404, detail="Owner not found")

    # Prepare animal data
    animal_data = {
        "name": animal.name,
        "owner": animal.owner,
        "type": animal.type,
        "dob": animal.dob,
        "gender": animal.gender,
        "milk_ability": animal.milk_ability,
        "status": animal.status,
        "health": animal.health,
        "image": animal.image,
    }

    # Store the animal record in Firestore
    animal_ref = db.collection("animals").document()
    animal_ref.set(animal_data)

    return JSONResponse(
        content={
            "message": "Animal added successfully",
            "animal_id": animal_ref.id,
            "image_path": "image_path",
        },
        status_code=200,
    )

@app.get("/animals/{owner}")
async def get_animals_by_owner(owner: str):
    animals_ref = db.collection("animals").where("owner", "==", owner).stream()
    animals = []
    for animal in animals_ref:
        animal_data = animal.to_dict()
        animal_data["id"] = animal.id  # Add Firestore document ID
        animals.append(animal_data)

    if not animals:
        raise HTTPException(status_code=404, detail="No animals found for this owner")
    return {"owner": owner, "animals": animals}

@app.get("/animal/{animal_id}")
async def get_animal_by_id(animal_id: str):
    animal_ref = db.collection("animals").document(animal_id)
    animal = animal_ref.get()
    
    if not animal.exists:
        raise HTTPException(status_code=404, detail="Animal not found")

    animal_data = animal.to_dict()
    animal_data["id"] = animal.id  # Include Firestore document ID
    return animal_data

class UpdateAnimal(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    dob: Optional[str] = None
    gender: Optional[str] = None
    milk_ability: Optional[bool] = None
    status: Optional[str] = None
    health: Optional[str] = None
    image: Optional[str] = None

@app.delete("/delete-animal/{animal_id}")
async def delete_animal(animal_id: str):
    animal_ref = db.collection("animals").document(animal_id)
    if not animal_ref.get().exists:
        raise HTTPException(status_code=404, detail="Animal not found")

    animal_ref.delete()
    return JSONResponse(content={"message": "Animal deleted successfully"}, status_code=200)

@app.patch("/update-animal/{animal_id}")
async def update_animal(animal_id: str, updates: UpdateAnimal):
    animal_ref = db.collection("animals").document(animal_id)
    if not animal_ref.get().exists:
        raise HTTPException(status_code=404, detail="Animal not found")

    update_data = {k: v for k, v in updates.dict().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No valid fields provided for update")

    animal_ref.update(update_data)
    return JSONResponse(content={"message": "Animal updated successfully"}, status_code=200)

# Predict Health
class HealthStatusRequest(BaseModel):
    reproductive_status: str
    feeding_amount_KG_1: float
    feeding_amount_KG_2: float
    average_food_weight_KG: float
    travel_distance_per_day_KM: float

health_status_mapping = {0: 'Healthy', 1: 'Sick', 2: 'Underweight'}
reproductive_status_mapping = {'Breeding': 0, 'Lactating': 1, 'Non-reproductive': 2, 'Pregnant': 3}



# Define the endpoint for health status prediction
@app.post("/predict-health-status")
def predict_health_status(request: HealthStatusRequest):
    # Validate and encode the reproductive status
    if request.reproductive_status not in reproductive_status_mapping:
        raise HTTPException(status_code=400, detail=f"Invalid reproductive_status: {request.reproductive_status}")
    reproductive_status_encoded = reproductive_status_mapping[request.reproductive_status]

    # Prepare the input features as a numpy array
    input_features = np.array([[reproductive_status_encoded, 
                                request.feeding_amount_KG_1, 
                                request.feeding_amount_KG_2, 
                                request.average_food_weight_KG, 
                                request.travel_distance_per_day_KM]])

    # Make the prediction
    try:
        health_status_encoded = MODEL_HEALTH.predict(input_features)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")
    health_status = health_status_mapping.get(health_status_encoded, "Unknown")

    return {"health_status": health_status}

class FeedPredictionRequest(BaseModel):
    cattle_breed: str
    health_status: str
    status: str
    feeding_amount_KG_morning: float
    score_morning: float
    feeding_amount_KG_noon: float
    score_noon: float
    feeding_amount_KG_evening: float
    score_evening: float
    travel_distance_per_day_KM: float

encodings = {
    'cattle_breed': {'Zebu': 0, 'Ayrshire': 1, 'Friesian': 2, 'Jersey': 3, 'Lanka White': 4, 'Sahiwal': 5},
    'health_status': {'Healthy': 0, 'Sick': 1},
    'status': {'Breeding': 0, 'Bulls': 1, 'Calves': 2, 'Heifers': 3, 'Lactating': 4, 'Pregnant': 5, 'Active': 3},
    'food_type_morning': {'Coconut Poonac': 0, 'Coconut Poonac, Grass': 1, 'Milk': 2},
    'food_type_noon': {'Coconut Poonac, Grass': 0, 'Grass, Paddy Straw': 1, 'Napier Grass, Guinea grass': 2,
                       'Napier Grass, Guinea grass, Para grass': 3, 'Napier Grass, Guinea grass,Gliricidia': 4,
                       'Paddy Straw, Grass (Chopped)': 5, 'Para grass, Gliricidia': 6},
    'food_type_evening': {'Milk': 0, 'Paddy Straw': 1, 'Paddy Straw, Corn': 2, 'Paddy Straw, Grass': 3, 'Paddy Straw, Legumes': 4},
    'time_of_day': {
        'evening': 0, 'morning': 1, 'noon': 2
    }
}

@app.post("/predict_food_type")
async def predict_food_type(request: FeedPredictionRequest):
    print(request)
    if request.status == 'Active':
        request.status = 'Heifers'
    
    # Prepare the input list and identify invalid fields
    encoded_input = [
        encodings['cattle_breed'].get(request.cattle_breed, -1),
        encodings['health_status'].get(request.health_status, -1),
        encodings['status'].get(request.status, -1),
        request.feeding_amount_KG_morning,
        request.score_morning,
        request.feeding_amount_KG_noon,
        request.score_noon,
        request.feeding_amount_KG_evening,
        request.score_evening,
        request.travel_distance_per_day_KM
    ]

    # Identify which field has an invalid value (-1)
    invalid_fields = []
    for i, value in enumerate(encoded_input[:3]):
        if value == -1:
            if i == 0:
                invalid_fields.append("cattle_breed")
            elif i == 1:
                invalid_fields.append("health_status")
            elif i == 2:
                invalid_fields.append("status")
    print(invalid_fields)
    if invalid_fields:
        raise HTTPException(status_code=400, detail=f"Invalid input value(s) for: {', '.join(invalid_fields)}")

    encoded_input = np.array(encoded_input).reshape(1, -1)

    try:
        prediction = MODEL_FEED.predict(encoded_input)
        morning_pred, noon_pred, evening_pred = prediction[0].split('-')
        food_type_morning = [key for key, value in encodings['food_type_morning'].items() if value == int(morning_pred)][0]
        food_type_noon = [key for key, value in encodings['food_type_noon'].items() if value == int(noon_pred)][0]
        food_type_evening = [key for key, value in encodings['food_type_evening'].items() if value == int(evening_pred)][0]

        return {
            'morning': food_type_morning,
            'noon': food_type_noon,
            'evening': food_type_evening
        }
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}. See server logs for more details.")

# Feed patern save
class CattleData(BaseModel):
    cattle_name: str
    health_status: str
    status: str
    food_type_morning: str
    feeding_amount_KG_morning: float
    score_morning: int
    food_type_noon: str
    feeding_amount_KG_noon: float
    score_noon: int
    food_type_evening: str
    feeding_amount_KG_evening: float
    score_evening: int
    feed_platform: str
    # feeding_amount_KG_L: float
    travel_distance_per_day_KM: float
    # farmers_id: str
    farmer_name: str
    feed_date: str

# Endpoint to save cattle data
@app.post("/feed-patterns")
async def save_cattle_data(cattle_data: CattleData):
    try:
        # Add the cattle data to Firestore
        feed_patterns_collection = db.collection("feed_patterns")
        new_doc = feed_patterns_collection.document()
        new_doc.set(cattle_data.dict())

        return {"message": "Cattle data successfully saved", "id": new_doc.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving cattle data: {str(e)}")

# Endpoint to retrieve all cattle data
@app.get("/feed-patterns")
async def get_all_cattle_data():
    try:
        feed_patterns_collection = db.collection("feed_patterns")
        docs = feed_patterns_collection.stream()

        cattle_data_list = [doc.to_dict() for doc in docs]

        return {"data": cattle_data_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching cattle data: {str(e)}")

@app.get("/feed-patterns/{cattle_name}")
async def get_cattle_data_by_name(cattle_name: str):
    try:
        feed_patterns_collection = db.collection("feed_patterns")
        query = (
            feed_patterns_collection
            .where("cattle_name", "==", cattle_name)
            .order_by("feed_date", direction=firestore.Query.DESCENDING)
            .stream()
        )

        results = [{"id": doc.id, **doc.to_dict()} for doc in query]

        if not results:
            raise HTTPException(status_code=404, detail="No records found for the given cattle name")

        return {"data": results}
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}. See server logs for more details.")
    
@app.get("/feed-patterns/{cattle_name}/summary")
async def get_cattle_feed_summary(cattle_name: str):
    try:
        # Query all feed records for this cattle
        feed_patterns_collection = db.collection("feed_patterns")
        query = (feed_patterns_collection
                 .where("cattle_name", "==", cattle_name)
                 .order_by("feed_date", direction=firestore.Query.DESCENDING))
        
        docs = query.stream()
        records = [doc.to_dict() for doc in docs]
        
        if not records:
            raise HTTPException(
                status_code=404,
                detail=f"No feed records found for cattle: {cattle_name}"
            )
        
        last_record = records[0]
        
        # Calculate total daily feed from last record
        morning_feed = last_record.get("feeding_amount_KG_morning", 0)
        noon_feed = last_record.get("feeding_amount_KG_noon", 0)
        evening_feed = last_record.get("feeding_amount_KG_evening", 0)
        total_daily_feed = morning_feed + noon_feed + evening_feed
        
        # Calculate average scores across all records
        total_records = len(records)
        avg_morning_score = sum(r.get("score_morning", 0) for r in records) / total_records
        avg_noon_score = sum(r.get("score_noon", 0) for r in records) / total_records
        avg_evening_score = sum(r.get("score_evening", 0) for r in records) / total_records
        
        # Prepare response
        response = {
            "cattle_name": cattle_name,
            "last_feed_date": last_record.get("feed_date"),
            "morning_feed_kg": morning_feed,
            "noon_feed_kg": noon_feed,
            "evening_feed_kg": evening_feed,
            "total_daily_feed_kg": total_daily_feed,
            "average_morning_score": round(avg_morning_score, 1),
            "average_noon_score": round(avg_noon_score, 1),
            "average_evening_score": round(avg_evening_score, 1),
            "feed_platform": last_record.get("feed_platform"),
            "health_status": last_record.get("health_status"),
            "farmer_name": last_record.get("farmer_name"),
            "total_records_analyzed": total_records
        }
        
        return response
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {str(e)}\n{error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing feed data: {str(e)}"
        )

class ScoreUpdate(BaseModel):
    score_morning: int | None = None
    score_noon: int | None = None
    score_evening: int | None = None

@app.patch("/feed-patterns/{doc_id}")
async def update_cattle_scores(doc_id: str, update_data: ScoreUpdate):
    try:
        feed_patterns_collection = db.collection("feed_patterns")
        doc_ref = feed_patterns_collection.document(doc_id)
        doc = doc_ref.get()

        if not doc.exists:
            raise HTTPException(status_code=404, detail="Record not found")

        update_fields = {k: v for k, v in update_data.dict().items() if v is not None}

        if not update_fields:
            raise HTTPException(status_code=400, detail="No valid fields provided for update")

        doc_ref.update(update_fields)

        return {"message": "Record updated successfully", "updated_fields": update_fields}
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}. See server logs for more details.")

# Endpoint to retrieve cattle data by farmer
@app.get("/feed-patterns/farmer/{farmer_id}")
async def get_cattle_data_by_farmer(farmer_id: str):
    try:
        feed_patterns_collection = db.collection("feed_patterns")
        docs = feed_patterns_collection.where("farmer_name", "==", farmer_id).stream()

        cattle_data_list = []
        
        for doc in docs:
            cattle_data = doc.to_dict()
            
            # Fetch animal details using cattle_name as animal_id
            animal_id = cattle_data.get("cattle_name")
            if animal_id:
                # Call the get_animal_by_id function or equivalent to fetch animal info
                animal_data = await get_animal_by_id(animal_id)  # Assuming the function is async
                # Merge animal data with cattle data
                cattle_data.update(animal_data)

            cattle_data_list.append(cattle_data)

        if not cattle_data_list:
            raise HTTPException(status_code=404, detail="No cattle data found for the given farmer")

        return {"data": cattle_data_list}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching cattle data by farmer: {str(e)}")

@app.get("/feed-patterns/farmer/{farmer_id}/last-30-days")
async def get_cattle_data_last_30_days(farmer_id: str):
    try:
        # Get the current date and the date 30 days ago
        current_date = datetime.utcnow()
        thirty_days_ago = current_date - timedelta(days=30)

        feed_patterns_collection = db.collection("feed_patterns")
        docs = feed_patterns_collection.where("farmer_name", "==", farmer_id).stream()

        cattle_data_list = []
        
        # Collect data for the last 30 days
        for doc in docs:
            cattle_data = doc.to_dict()
            feed_date = datetime.strptime(cattle_data.get("feed_date"), "%Y-%m-%d")

            if feed_date >= thirty_days_ago:
                # Fetch animal details using cattle_name as animal_id
                animal_id = cattle_data.get("cattle_name")
                if animal_id:
                    # Call the get_animal_by_id function or equivalent to fetch animal info
                    animal_data = await get_animal_by_id(animal_id)  # Assuming the function is async
                    # Merge animal data with cattle data
                    cattle_data.update(animal_data)

                cattle_data_list.append(cattle_data)

        if not cattle_data_list:
            raise HTTPException(status_code=404, detail="No cattle data found for the given farmer in the last 30 days")

        # Statistical Details
        food_types = []
        meal_scores = {'morning': {}, 'noon': {}, 'evening': {}}
        
        for cattle_data in cattle_data_list:
            # Collect food types for each meal
            food_types.extend([cattle_data["food_type_morning"], cattle_data["food_type_noon"], cattle_data["food_type_evening"]])

            # Collect scores for each meal
            meal_scores['morning'][cattle_data["food_type_morning"]] = max(meal_scores['morning'].get(cattle_data["food_type_morning"], 0), cattle_data["score_morning"])
            meal_scores['noon'][cattle_data["food_type_noon"]] = max(meal_scores['noon'].get(cattle_data["food_type_noon"], 0), cattle_data["score_noon"])
            meal_scores['evening'][cattle_data["food_type_evening"]] = max(meal_scores['evening'].get(cattle_data["food_type_evening"], 0), cattle_data["score_evening"])

        # Popular food types
        food_type_counter = Counter(food_types)
        total_meals = len(cattle_data_list) * 3  # 3 meals per cattle
        popular_foods = [{'food_type': food, 'percentage': (count / total_meals) * 100} for food, count in food_type_counter.items()]

        # Most scored food item for each meal
        most_scored_food = {
            'morning': max(meal_scores['morning'], key=meal_scores['morning'].get, default=None),
            'noon': max(meal_scores['noon'], key=meal_scores['noon'].get, default=None),
            'evening': max(meal_scores['evening'], key=meal_scores['evening'].get, default=None),
        }

        return {
            "data": cattle_data_list,
            "statistics": {
                "popular_foods": popular_foods,
                "most_scored_food": most_scored_food
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching cattle data for the last 30 days: {str(e)}")

@app.get("/feed-patterns/{username}/by_date/{date}")
async def get_feeding_records_by_date(username: str, date: str):
    try:
        # Fetch feeding records filtered by date and farmer_name
        feed_patterns_collection = (
            db.collection("feed_patterns")
            .where("feed_date", "==", date)
            .where("farmer_name", "==", username)
        )
        records = [doc.to_dict() for doc in feed_patterns_collection.stream()]

        return {"feeding_records": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching feeding records: {str(e)}")

@app.get("/feed-patterns/by_date/{date}")
async def get_feeding_records_by_date(date: str):
    try:
        # Fetch feeding records filtered by date
        feed_patterns_collection = db.collection("feed_patterns").where("feed_date", "==", date)
        records = [doc.to_dict() for doc in feed_patterns_collection.stream()]

        return {"feeding_records": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching feeding records: {str(e)}")


# Feed Type Weight Predict
CattleBreed = Literal['Zebu', 'Ayrshire', 'Friesian', 'Jersey', 'Lanka White', 'Sahiwal']
HealthStatus = Literal['Healthy', 'Sick']
CattleStatus = Literal['Breeding', 'Bulls', 'Calves', 'Heifers', 'Lactating', 'Pregnant', 'Active']
TimeOfDay = Literal['morning', 'noon', 'evening']

class WeightPredictionRequest(BaseModel):
    cattle_breed: CattleBreed
    health_status: HealthStatus
    status: CattleStatus
    food_type: str  
    time_of_day: TimeOfDay

class WeightPredictionResponse(BaseModel):
    predicted_amount: float
    units: str = "kg"

def predict_feeding_amount(cattle_breed: CattleBreed, 
                           health_status: HealthStatus, 
                           status: CattleStatus, 
                           food_type: str, 
                           time_of_day: TimeOfDay) -> float:
    # Determine which food_type encoding to use based on time_of_day
    food_type_encoding_key = f'food_type_{time_of_day}'

    # Safely get encoded values with default 0 if key not found
    encoded_values = [
        encodings['cattle_breed'].get(cattle_breed, 0),
        encodings['health_status'].get(health_status, 0),
        encodings['status'].get(status, 0),
        encodings.get(food_type_encoding_key, {}).get(food_type, 0),
        encodings['time_of_day'].get(time_of_day, 0)
    ]

    # Convert to DataFrame (1 sample, 5 features)
    input_df = pd.DataFrame([encoded_values], columns=[
        'cattle_breed', 'health_status', 'status', 'food_type', 'time_of_day'
    ])

    # Predict (replace with actual model prediction)
    prediction = model.predict(input_df)[0]  # Assuming model is loaded
    return round(prediction, 2)


@app.post("/predict-weight-feeding", response_model=WeightPredictionResponse)
async def predict_feeding(request: WeightPredictionRequest):
    try:
        prediction = predict_feeding_amount(
            cattle_breed=request.cattle_breed,
            health_status=request.health_status,
            status=request.status,
            food_type=request.food_type,
            time_of_day=request.time_of_day
        )
        
        return {
            "predicted_amount": prediction,
            "units": "kg"
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

# Breed Related

class Breed(BaseModel):
    image: str

# Predict Cattle Breed
@app.post("/predict-breed")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to classify an uploaded image using the loaded model.
    """
    # Save the uploaded file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Preprocess the image
        image = Image.open(file_path).convert("RGB")  # Ensure RGB format
        image = image.resize((48, 48))
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict the label
        predictions = MODEL_BREED.predict(image_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))  # Highest probability

        # Map the predicted index to the class name
        predicted_label = CLASS_BREED[predicted_index]

        # Clean up the uploaded file
        os.remove(file_path)

        return {
            "predicted_label": predicted_label,
            "confidence": confidence
        }
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}. See server logs for more details.")

ROBOFLOW_API_KEY = "X8oP99fDpU6RHWwHo4d2"
WORKSPACE_NAME = "hackishmax321"
WORKFLOW_ID = "custom-workflow-2"



client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

@app.post("/predict-breed-roboflow")
async def predict_roboflow(file: UploadFile = File(...)):
    """
    Endpoint to classify an uploaded image using Roboflow workflow API via inference_sdk.
    Returns predictions in Roboflow's standard format.
    """
    # Save the uploaded file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Run workflow prediction
        result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": file_path},
            use_cache=True  # cache workflow definition for 15 minutes
        )
        
        # Clean up the uploaded file
        os.remove(file_path)

        print(result)
        
        return {
            "results": result
        }
        
    except Exception as e:
        # Clean up the uploaded file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

# Predict Pests
@app.post("/predict-pest")
async def predict_pest(file: UploadFile = File(...)):
    """
    Endpoint to classify an uploaded image for pest attack detection.
    """
    # Save the uploaded file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Preprocess the image
        image = Image.open(file_path).convert("RGB")  # Ensure RGB format
        image = image.resize((48, 48))  # Resize to model's input size
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict the pest attack label
        predictions = MODEL_PESTS.predict(image_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))  # Highest probability

        # Map the predicted index to the pest attack label
        predicted_label = CLASS_PESTS[predicted_index]

        # Clean up the uploaded file
        os.remove(file_path)

        return {
            "predicted_label": predicted_label,
            "confidence": confidence
        }
    except Exception as e:
        os.remove(file_path)  # Clean up the uploaded file in case of error
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# Breed Information
file_path = 'breed_dataset/cattle_breed_idintification_dataset.xlsx'
df = pd.read_excel(file_path)

# Combine text-heavy fields into a single feature
df['text_data'] = (
    df['Cattle Breed Name'] + " " +
    df['Pedigree/Lineage'] + " " +
    df['Optimal Rearing Conditions'] + " " +
    df['Physical Characteristics'] + " " +
    df['Temperament'] + " " +
    df['Productivity Metrics']
)

# # Target variable
# target = 'Origin'

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     df['text_data'],
#     df[target],
#     test_size=0.2,
#     random_state=42
# )

# # Vectorize text data
# tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
# X_train_tfidf = tfidf.fit_transform(X_train)
# X_test_tfidf = tfidf.transform(X_test)

# # Train a Naive Bayes classifier
# model = MultinomialNB()
# model.fit(X_train_tfidf, y_train)

# Save the model and vectorizer
model = joblib.load('cattle_breed_nlp_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')

# Define request model
# class InsightRequest(BaseModel):
#     breed_name: str
#     adopted: bool

# Define output model for FastAPI
# class InsightResponse(BaseModel):
#     Breed: str
#     Adopted: str
#     Predicted_Origin: str
#     Rearing_Conditions: str
#     Temperament: str
#     Milk_Production: int
#     Lifespan: int

class AdoptionInfo(BaseModel):
    is_adopted: bool
    farm_size: str  # 'small', 'medium', 'large'
    climate: str    # 'tropical', 'temperate', 'cold'
    purpose: str    # 'dairy', 'beef', 'dual'

class InsightRequest(BaseModel):
    breed_name: str
    adoption_info: AdoptionInfo

class BreedIdentification(BaseModel):
    name: str
    origin: str
    predicted_origin: str
    pedigree: str
    physical_characteristics: str

class CareRequirements(BaseModel):
    optimal_conditions: str
    temperament: str
    lifespan: str
    climate_adaptability: str

class ProductivityAnalysis(BaseModel):
    milk_production: str
    production_category: str
    primary_purpose: str

class AdoptionAnalysis(BaseModel):
    current_status: str
    suitability_score: int
    suitability_rating: str
    feedback: List[str]
    recommendation: str
    farm_size_compatibility: str
    climate_compatibility: str
    purpose_alignment: str

class InsightResponse(BaseModel):
    breed_identification: BreedIdentification
    care_requirements: CareRequirements
    productivity_analysis: ProductivityAnalysis
    adoption_analysis: AdoptionAnalysis

class ErrorResponse(BaseModel):
    error: str
    suggestions: Optional[List[str]] = None

pipeline = joblib.load('cattle_breed_classifier_tunned.joblib')

@app.post("/insights", response_model=Union[InsightResponse, ErrorResponse])
async def get_breed_insights(request: InsightRequest):
    """
    Get comprehensive insights about a cattle breed including adoption suitability analysis.
    """
    try:
        # Find breed in dataset (case insensitive)
        breed_mask = df['Cattle Breed Name'].str.lower() == request.breed_name.lower()
        breed_data = df[breed_mask]
        
        if breed_data.empty:
            closest_matches = df[df['Cattle Breed Name'].str.lower().str.contains(request.breed_name.lower())]
            if not closest_matches.empty:
                suggestions = closest_matches['Cattle Breed Name'].unique().tolist()
                return ErrorResponse(error="Breed not found", suggestions=suggestions)
            return ErrorResponse(error="Breed not found in dataset")

        # Extract breed information
        breed_info = breed_data.iloc[0]
        
        # Make prediction using full pipeline
        tfidf_vec = pipeline['vectorizer'].transform([breed_info['text_data']])
        svd_transformed = pipeline['svd'].transform(tfidf_vec)
        predicted_origin = pipeline['model'].predict(svd_transformed)[0]

        # Calculate adoption suitability score (0-100)
        suitability_score = 0
        feedback = []
        
        # Climate compatibility check
        rearing_conditions = breed_info['Optimal Rearing Conditions'].lower()
        if request.adoption_info.climate in rearing_conditions:
            suitability_score += 30
            feedback.append(f"Excellent climate match for {request.adoption_info.climate} conditions")
        else:
            feedback.append(f"Potential climate mismatch - breed prefers {rearing_conditions}")

        # Farm size compatibility
        if request.adoption_info.farm_size == 'large':
            suitability_score += 25
        elif request.adoption_info.farm_size == 'medium':
            suitability_score += 15
        else:  # small
            if 'small' in rearing_conditions.lower():
                suitability_score += 10
            else:
                suitability_score -= 5
                feedback.append("This breed may require more space than your small farm provides")

        # Purpose compatibility
        productivity = breed_info['Productivity Metrics'].lower()
        if request.adoption_info.purpose in productivity:
            suitability_score += 30
            feedback.append(f"Excellent match for {request.adoption_info.purpose} production")
        elif 'dual' in productivity:
            suitability_score += 20
            feedback.append("Good match - breed serves multiple purposes")
        else:
            feedback.append(f"Potential purpose mismatch - breed specializes in {productivity}")

        # Adjust score if already adopted
        if request.adoption_info.is_adopted:
            suitability_score += 15
            feedback.append("Positive experience with this breed reported")
        else:
            suitability_score -= 5

        # Ensure score is within bounds
        suitability_score = max(0, min(100, suitability_score))

        # Prepare response
        milk_prod = breed_info['Milk Production Ability (Liters/Year)']
        return InsightResponse(
            breed_identification=BreedIdentification(
                name=breed_info['Cattle Breed Name'],
                origin=breed_info['Origin'],
                predicted_origin=predicted_origin,
                pedigree=breed_info['Pedigree/Lineage'],
                physical_characteristics=breed_info['Physical Characteristics']
            ),
            care_requirements=CareRequirements(
                optimal_conditions=breed_info['Optimal Rearing Conditions'],
                temperament=breed_info['Temperament'],
                lifespan=f"{breed_info['Lifespan (Years)']} years",
                climate_adaptability=rearing_conditions
            ),
            productivity_analysis=ProductivityAnalysis(
                milk_production=f"{milk_prod} liters/year",
                production_category="High yield" if milk_prod > 5000 else 
                                  "Medium yield" if milk_prod > 3000 else 
                                  "Low yield",
                primary_purpose="Dairy" if 'milk' in productivity.lower() else 
                               "Beef" if 'meat' in productivity.lower() else 
                               "Dual purpose"
            ),
            adoption_analysis=AdoptionAnalysis(
                current_status="Adopted" if request.adoption_info.is_adopted else "Not adopted",
                suitability_score=suitability_score,
                suitability_rating="Excellent" if suitability_score >= 80 else 
                                 "Good" if suitability_score >= 60 else 
                                 "Moderate" if suitability_score >= 40 else 
                                 "Poor",
                feedback=feedback,
                recommendation="Highly recommended" if suitability_score >= 70 else 
                              "Recommended with conditions" if suitability_score >= 50 else 
                              "Not recommended",
                farm_size_compatibility=request.adoption_info.farm_size,
                climate_compatibility=request.adoption_info.climate,
                purpose_alignment=request.adoption_info.purpose
            )
        )

    except Exception as e:
        return ErrorResponse(error=f"An error occurred: {str(e)}")


# Growth Monitor
CATTLE_BREED_ENCODING = {
    ' ': 0,
    'AUSTRALIAN MILKING ZEBU': 1,
    'AYRSHIRE': 2,
    'FRIESIAN': 3,
    'JERSEY': 4,
    'LANKA WHITE': 5,
    'SAHIWAL': 6,
    'nan': 7
}

LACTATION_STAGE_ENCODING = {
    'EARLY': 0,
    'LATE': 1,
    'MID': 2,
    'nan': 3
}

REPRODUCTIVE_STATUS_ENCODING = {
    ' PREGNANT': 0,
    'NOT PREGNANT': 1,
    'PREGNANT': 2,
    'nan': 3
}

# Define input data model
class CattleData(BaseModel):
    cattle_breed: str
    height_cm: float
    age_years: float
    feed_kg_per_day: float
    lactation_stage: str
    reproductive_status: str


@app.post("/predict-growth-weight")
async def predict_weight(data: CattleData):
    
    try:
        # Encode categorical variables
        cattle_breed_encoded = CATTLE_BREED_ENCODING.get(data.cattle_breed, 0)  # Default to 0 if not found
        lactation_stage_encoded = LACTATION_STAGE_ENCODING.get(data.lactation_stage, 3)  # Default to 3 if not found
        reproductive_status_encoded = REPRODUCTIVE_STATUS_ENCODING.get(data.reproductive_status, 3)  # Default to 3 if not found

        # Prepare input for the model
        input_data = np.array([[cattle_breed_encoded, data.height_cm, data.age_years, data.feed_kg_per_day, lactation_stage_encoded, reproductive_status_encoded]])

        # Predict weight using the model
        predicted_weight = MODEL_GROWTH.predict(input_data)
        print(predicted_weight[0])

        return {"predicted_weight": predicted_weight[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Growth Records
class GrowthRecord(BaseModel):
    owner: str
    cattle: str
    breed: str
    age: int
    weight: float
    height: float

# In-memory storage (replace with a database in production)
growth_records_db: Dict[str, List[GrowthRecord]] = {}

# POST endpoint to add a growth record
@app.post("/growth-records")
def add_growth_record(record: GrowthRecord):
    cattle_id = record.cattle  # Animal ID

    # Check if the animal exists
    animal_ref = db.collection("animals").document(cattle_id)
    animal = animal_ref.get()
    if not animal.exists:
        raise HTTPException(status_code=404, detail="Animal not found")

    # Add growth record as a subcollection under the animal
    growth_ref = animal_ref.collection("growth_records")
    growth_ref.add(record.dict())  # Firestore will auto-generate a document ID

    return {"message": "Growth record added successfully", "data": record.dict()}

@app.get("/growth-records/{cattle_id}")
def get_growth_records(cattle_id: str):
    animal_ref = db.collection("animals").document(cattle_id)
    if not animal_ref.get().exists:
        raise HTTPException(status_code=404, detail="Animal not found")

    # First get all records ordered by age
    growth_ref = animal_ref.collection("growth_records").order_by("age").stream()
    
    # Group by age and keep last modified
    records_by_age = {}
    for doc in growth_ref:
        record = doc.to_dict()
        age = record['age']
        # If you have a timestamp field, use it to determine newest
        if 'timestamp' in record:
            if age not in records_by_age or record['timestamp'] > records_by_age[age].get('timestamp', 0):
                records_by_age[age] = record
        else:
            # Without timestamp, we can't reliably determine newest - just overwrite
            records_by_age[age] = record

    # Return sorted by age
    return {"growth_records": sorted(records_by_age.values(), key=lambda x: x['age'])}


# Milk Records
class MilkCollection(BaseModel):
    date_collected: str
    cattle: str
    amount: float
    status: str = "no issue" 

@app.post("/milk_collection/{username}")
async def create_milk_record(username: str, milk: MilkCollection):
    try:
        user_ref = db.collection("users").document(username)
        if not user_ref.get().exists:
            raise HTTPException(status_code=400, detail="User not found")

        # Create a reference to the milk collection for the user
        milk_ref = db.collection("users").document(username).collection("milk_collection").add(milk.dict())

        return {
            "message": "Milk collection record added successfully",
            "milk_record": milk.dict()
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {str(e)}\n{error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}. See server logs for more details."
        )

@app.get("/milk_collection/{username}")
async def get_all_milk_records(username: str):
    user_ref = db.collection("users").document(username)
    if not user_ref.get().exists:
        raise HTTPException(status_code=400, detail="User not found")

    # Fetch all milk collection records for the user
    records_ref = db.collection("users").document(username).collection("milk_collection")
    records = [doc.to_dict() for doc in records_ref.stream()]

    return {"milk_records": records}

@app.get("/milk_collection/{username}/{cattle}")
async def get_milk_records_by_cattle(username: str, cattle: str):
    user_ref = db.collection("users").document(username)
    if not user_ref.get().exists:
        raise HTTPException(status_code=400, detail="User not found")
    
    records_ref = (
        db.collection("users").document(username)
        .collection("milk_collection")
        .where("cattle", "==", cattle)
        .stream()
    )
    
    records = [doc.to_dict() for doc in records_ref]
    
    # Sorting records in descending order based on date_collected (YYYY-MM-DD)
    records.sort(key=lambda x: datetime.strptime(x["date_collected"], "%Y-%m-%d"), reverse=True)
    
    return {"milk_records": records}

@app.get("/milk_collection/{username}/by_date/{date}")
async def get_records_by_date(username: str, date: str):
    user_ref = db.collection("users").document(username)
    if not user_ref.get().exists:
        raise HTTPException(status_code=400, detail="User not found")

    # Fetch milk collection records filtered by date for the user
    records_ref = db.collection("users").document(username).collection("milk_collection").where("date_collected", "==", date)
    records = [doc.to_dict() for doc in records_ref.stream()]

    return {"milk_records": records}

@app.get("/milk_collection/{username}/by_cattle/{cattle}")
async def get_records_by_cattle(username: str, cattle: str):
    user_ref = db.collection("users").document(username)
    if not user_ref.get().exists:
        raise HTTPException(status_code=400, detail="User not found")

    # Fetch milk collection records filtered by cattle for the user
    records_ref = db.collection("users").document(username).collection("milk_collection").where("cattle", "==", cattle)
    records = [doc.to_dict() for doc in records_ref.stream()]

    return {"milk_records": records}

@app.delete("/milk_collection/{username}/delete/{record_id}")
async def delete_milk_record(username: str, record_id: str):
    user_ref = db.collection("users").document(username)
    if not user_ref.get().exists:
        raise HTTPException(status_code=400, detail="User not found")

    # Delete a specific milk collection record for the user
    milk_ref = db.collection("users").document(username).collection("milk_collection").document(record_id)
    if not milk_ref.get().exists:
        raise HTTPException(status_code=400, detail="Record not found")

    milk_ref.delete()
    
    return {"message": "Milk collection record deleted successfully"}

# Get milk collections anual given breed
@app.get("/milk_collection/{owner}/{breed}")
async def get_milk_records_by_breed(owner: str, breed: str):
    # Step 1: Get cattle IDs of the specified breed
    cattle_ref = db.collection("animals").where("owner", "==", owner).where("type", "==", breed).stream()
    cattle_ids = [cattle.id for cattle in cattle_ref]

    if not cattle_ids:
        raise HTTPException(status_code=404, detail="No cattle found for this breed under the given owner")

    # Step 2: Retrieve milk records of the fetched cattle
    milk_records = []
    user_milk_ref = db.collection("users").document(owner).collection("milk_collection").stream()

    for record in user_milk_ref:
        milk_data = record.to_dict()
        if milk_data["cattle"] in cattle_ids:
            milk_records.append(milk_data)

    if not milk_records:
        raise HTTPException(status_code=404, detail="No milk records found for the given breed")

    # Step 3: Aggregate milk amount annually
    yearly_totals = defaultdict(float)
    
    for record in milk_records:
        try:
            year = datetime.strptime(record["date_collected"], "%Y-%m-%d").year
            yearly_totals[year] += record["amount"]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {record['date_collected']}")

    # Step 4: Return the totals in ascending order of year
    sorted_totals = sorted(yearly_totals.items())

    return {"owner": owner, "breed": breed, "annual_milk_collection": sorted_totals}

@app.get("/milk_collection_by_names/{owner}/{breed}")
async def get_milk_records_by_breed(owner: str, breed: str):
    print(owner)
    print(breed)
    # Step 1: Get cattle names of the specified breed
    cattle_ref = db.collection("animals").where("owner", "==", owner).where("type", "==", breed).stream()
    cattle_names = [cattle.to_dict().get("name") for cattle in cattle_ref]

    print(cattle_names)

    if not cattle_names:
        raise HTTPException(status_code=404, detail="No cattle found for this breed under the given owner")

    # Step 2: Retrieve milk records of the fetched cattle names
    milk_records = []
    user_milk_ref = db.collection("users").document(owner).collection("milk_collection").stream()

    for record in user_milk_ref:
        milk_data = record.to_dict()
        if milk_data["cattle"] in cattle_names:
            milk_records.append(milk_data)

    print(milk_records)
    if not milk_records:
        raise HTTPException(status_code=404, detail="No milk records found for the given breed")

    # Step 3: Aggregate milk amount annually
    yearly_totals = defaultdict(float)
    
    for record in milk_records:
        try:
            year = datetime.strptime(record["date_collected"], "%Y-%m-%d").year
            yearly_totals[year] += record["amount"]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {record['date_collected']}")

    # Step 4: Return the totals in ascending order of year
    sorted_totals = sorted(yearly_totals.items())

    return {"owner": owner, "breed": breed, "annual_milk_collection": sorted_totals}


# Milk Prediction
MOODEL_MILK_PROD = joblib.load('model_milk_prediction.joblib')

cattle_breed_encoding = {
    " ": 0,
    "AUSTRALIAN MILKING ZEBU": 1,
    "AYRSHIRE": 2,
    "FRIESIAN": 3,
    "JERSEY": 4,
    "LANKA WHITE": 5,
    "SAHIWAL": 6,
    "nan": 7
}

lactation_stage_encoding = {
    "EARLY": 0,
    "LATE": 1,
    "MID": 2,
    "nan": 3
}

reproductive_status_encoding = {
    " PREGNANT": 0,
    "NOT PREGNANT": 1,
    "PREGNANT": 2,
    "nan": 3
}

class MilkPredictionRequest(BaseModel):
    cattle_breed: str
    height_cm: float
    age_years: int
    feed_kg_per_day: float
    lactation_stage: str
    reproductive_status: str

# Prediction Endpoint
@app.post("/predict_milk")
async def predict_milk(data: MilkPredictionRequest):
    try:
        print(data)
        # Encode categorical variables
        cattle_breed_encoded = cattle_breed_encoding.get(data.cattle_breed, 0)
        lactation_stage_encoded = lactation_stage_encoding.get(data.lactation_stage, 3)
        reproductive_status_encoded = reproductive_status_encoding.get(data.reproductive_status, 3)

        # Prepare input data
        input_data = np.array([[cattle_breed_encoded, data.height_cm, data.age_years, data.feed_kg_per_day, lactation_stage_encoded, reproductive_status_encoded]])

        # Predict using the model
        predicted_milk = MOODEL_MILK_PROD.predict(input_data)

        return {"predicted_milk_production": round(predicted_milk[0], 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# GET IoT loaction
def calculate_duration(loc1, loc2, speed_kmph=10):
    """
    Calculate the straight-line distance and approximate duration between two locations.

    Args:
        loc1 (dict): {"latitude": float, "longitude": float} for the first location.
        loc2 (dict): {"latitude": float, "longitude": float} for the second location.
        speed_kmph (float): Assumed speed in kilometers per hour (default is 50 km/h).

    Returns:
        dict: {"distance_km": float, "duration_minutes": float}
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = radians(loc1["latitude"]), radians(loc1["longitude"])
    lat2, lon2 = radians(loc2["latitude"]), radians(loc2["longitude"])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Radius of the Earth in kilometers
    R = 6371.0
    distance_km = round((100 * 24 * R * c) / 1000, 3)

    # Calculate duration (time = distance / speed)
    duration_hours = distance_km / speed_kmph
    duration_minutes = duration_hours * 60 

    return {"distance_km": distance_km, "duration_minutes": duration_minutes}




# Locate Farm and Cattle (IOT)
@app.get("/location")
async def get_location():
    try:
        # Reference to the location node
        ref = FireDB.reference("location")
        ref24 = FireDB.reference("location24")
        location_data = ref.get()
        location24_data = ref24.get()

        print(location_data)

        
        
        if not location_data:
            raise HTTPException(status_code=404, detail="Location data not found")
        
        duration = calculate_duration(location_data, location24_data)
        
        return {"location": location_data, "location_24": location24_data, "duration": duration}
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error fetching location data: {str(e)}")

# Farm Location / border save
class FarmBorderRequest(BaseModel):
    user: str
    farm_name: str
    details: str
    border: list[dict]

@app.post("/mark-farm-border")
async def mark_farm_border(request: FarmBorderRequest):
    try:
        # Prepare farm data
        farm_data = {
            "user": request.user,
            "farm_name": request.farm_name,
            "details": request.details,
            "border": request.border,
        }
        
        # Save to Firestore
        farms_collection = db.collection("farms")
        new_farm_doc = farms_collection.document()
        new_farm_doc.set(farm_data)
        
        return {"message": "Farm border successfully saved", "farm_id": new_farm_doc.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving farm border: {str(e)}")

@app.get("/farms/{user}")
async def get_farms_by_user(user: str):
    try:
        farms_collection = db.collection("farms")
        query = farms_collection.where("user", "==", user).stream()

        farms = [{"id": doc.id, **doc.to_dict()} for doc in query]

        if not farms:
            raise HTTPException(status_code=404, detail="No farms found for the given user")

        return {"farms": farms}
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}. See server logs for more details.")

@app.delete("/farms/{farm_id}")
async def delete_farm_by_id(farm_id: str):
    try:
        farms_collection = db.collection("farms")
        farm_doc_ref = farms_collection.document(farm_id)
        farm_doc = farm_doc_ref.get()

        if not farm_doc.exists:
            raise HTTPException(status_code=404, detail="Farm not found")

        farm_doc_ref.delete()

        return {"message": "Farm successfully deleted"}
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}. See server logs for more details.")
