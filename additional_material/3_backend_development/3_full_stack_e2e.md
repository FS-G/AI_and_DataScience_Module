# Full-Stack Hospital Management System (End-to-End Tutorial)

This tutorial demonstrates building a complete full-stack application: a minimal hospital management system with FastAPI backend and a HTML/CSS/JavaScript frontend.

## System Overview

The application manages:
- **User Authentication (JWT)**: Register/login to access protected operations
- **Patient Registration**: Add new patients to the system (with current location)
- **Doctor Management**: Manage doctors and their specializations
- **Doctor-Patient Assignment**: Assign doctors to patients
- **Treatment Tracking**: Record treatment details for a patient (who, what, when)
- **Discharge Management**: Mark patients as discharged
- **Patient Detail View**: See location, status, assigned doctor, and treatment history

---

## Part 1: Database Design & ERD

### Entity Relationship Diagram

```mermaid
erDiagram
    USERS ||--o{ TREATMENTS : "recorded_by"
    PATIENTS ||--o{ ASSIGNMENTS : "has"
    DOCTORS ||--o{ ASSIGNMENTS : "assigned_to"
    PATIENTS ||--o{ TREATMENTS : "receives"

    USERS {
        int id PK
        string email
        string hashed_password
        boolean is_active
    }

    PATIENTS {
        int id PK
        string name
        int age
        string gender
        string contact
        date admission_date
        string status
        string location
    }
    
    DOCTORS {
        int id PK
        string name
        string specialization
        string contact
        boolean is_available
    }
    
    ASSIGNMENTS {
        int id PK
        int patient_id FK
        int doctor_id FK
        date assigned_date
        date discharge_date
    }

    TREATMENTS {
        int id PK
        int patient_id FK
        int doctor_id FK
        int recorded_by_user_id FK
        string description
        string medication
        string dosage
        datetime given_at
    }
```

### Database Schema Explanation

**Users Table:**
- `id`: Primary key
- `email`: Login identity (unique)
- `hashed_password`: Bcrypt-hashed password
- `is_active`: Whether the user can log in

**Patients Table:**
- `id`: Primary key (auto-increment)
- `name`: Patient full name
- `age`: Patient age
- `gender`: Patient gender (M/F/Other)
- `contact`: Contact number/email
- `admission_date`: Date of admission
- `status`: Current status (Admitted/Discharged)
- `location`: Current ward/room

**Doctors Table:**
- `id`: Primary key (auto-increment)
- `name`: Doctor full name
- `specialization`: Medical specialization
- `contact`: Contact information
- `is_available`: Whether doctor can accept new patients

**Assignments Table:**
- `id`: Primary key (auto-increment)
- `patient_id`: Foreign key to Patients
- `doctor_id`: Foreign key to Doctors
- `assigned_date`: When doctor was assigned
- `discharge_date`: When patient was discharged (NULL if active)

**Treatments Table:**
- `id`: Primary key
- `patient_id`: Patient receiving treatment
- `doctor_id`: Doctor associated
- `recorded_by_user_id`: User who recorded the treatment
- `description`: Treatment description
- `medication`: What medication was given
- `dosage`: Dosage information
- `given_at`: Timestamp (server time)

**Relationships:**
- One Patient can have multiple Assignments (historical records)
- One Doctor can have multiple Assignments
- Each Assignment links one Patient to one Doctor

---

## Part 2: FastAPI Backend Implementation

### Project Structure
```
hospital_backend/
├── main.py
├── config.py
├── database.py
├── models.py
├── schemas.py
├── routers/
│   ├── __init__.py
│   ├── patients.py
│   ├── doctors.py
│   └── assignments.py
├── requirements.txt
├── alembic/
└── .env
```

### Step 1: Dependencies (`requirements.txt`)

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
psycopg[binary]==3.1.14
pydantic==2.5.0
pydantic-settings==2.1.0
alembic==1.12.1
python-dotenv==1.0.0
python-multipart==0.0.6
bcrypt==4.0.1
PyJWT==2.8.0
```

### Step 2: Configuration (`config.py`)

```python
# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
```

### Step 3: Database Setup (`database.py`)

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Step 4: Database Models (`models.py`)

```python
# models.py
from sqlalchemy import Integer, String, Date, Boolean, ForeignKey, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime, date
from database import Base

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

class Patient(Base):
    __tablename__ = "patients"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String(10), nullable=False)
    contact: Mapped[str] = mapped_column(String(100), nullable=False)
    admission_date: Mapped[date] = mapped_column(Date, nullable=False, default=date.today)
    status: Mapped[str] = mapped_column(String(20), default="Admitted")
    location: Mapped[str] = mapped_column(String(100), default="Ward A - Room 1")
    
    assignments: Mapped[list["Assignment"]] = relationship("Assignment", back_populates="patient")

class Doctor(Base):
    __tablename__ = "doctors"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    specialization: Mapped[str] = mapped_column(String(100), nullable=False)
    contact: Mapped[str] = mapped_column(String(100), nullable=False)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True)
    
    assignments: Mapped[list["Assignment"]] = relationship("Assignment", back_populates="doctor")

class Assignment(Base):
    __tablename__ = "assignments"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(Integer, ForeignKey("patients.id"), nullable=False)
    doctor_id: Mapped[int] = mapped_column(Integer, ForeignKey("doctors.id"), nullable=False)
    assigned_date: Mapped[date] = mapped_column(Date, nullable=False, default=date.today)
    discharge_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    
    patient: Mapped["Patient"] = relationship("Patient", back_populates="assignments")
    doctor: Mapped["Doctor"] = relationship("Doctor", back_populates="assignments")

class Treatment(Base):
    __tablename__ = "treatments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(Integer, ForeignKey("patients.id"), nullable=False)
    doctor_id: Mapped[int] = mapped_column(Integer, ForeignKey("doctors.id"), nullable=False)
    recorded_by_user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    medication: Mapped[str] = mapped_column(String(200), nullable=True)
    dosage: Mapped[str] = mapped_column(String(100), nullable=True)
    given_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

### Step 5: Pydantic Schemas (`schemas.py`)

```python
# schemas.py
from pydantic import BaseModel, EmailStr
from datetime import date, datetime
from typing import Optional, List

# Patient Schemas
class PatientBase(BaseModel):
    name: str
    age: int
    gender: str
    contact: str
    admission_date: date
    location: str

class PatientCreate(PatientBase):
    pass

class PatientRead(PatientBase):
    id: int
    status: str
    class Config:
        from_attributes = True

# Doctor Schemas
class DoctorBase(BaseModel):
    name: str
    specialization: str
    contact: str
    is_available: bool = True

class DoctorCreate(DoctorBase):
    pass

class DoctorRead(DoctorBase):
    id: int
    class Config:
        from_attributes = True

# Assignment Schemas
class AssignmentBase(BaseModel):
    patient_id: int
    doctor_id: int
    assigned_date: date

class AssignmentCreate(AssignmentBase):
    pass

class AssignmentRead(AssignmentBase):
    id: int
    discharge_date: Optional[date] = None
    patient: PatientRead
    doctor: DoctorRead
    class Config:
        from_attributes = True

# Combined schemas for frontend
class PatientWithDoctor(BaseModel):
    patient: PatientRead
    doctor: Optional[DoctorRead] = None
    assignment_id: Optional[int] = None

# Auth Schemas
class UserCreate(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

# Treatment Schemas
class TreatmentCreate(BaseModel):
    patient_id: int
    doctor_id: int
    description: str
    medication: Optional[str] = None
    dosage: Optional[str] = None

class TreatmentRead(BaseModel):
    id: int
    patient_id: int
    doctor_id: int
    description: str
    medication: Optional[str] = None
    dosage: Optional[str] = None
    given_at: datetime
    class Config:
        from_attributes = True

class PatientDetails(BaseModel):
    patient: PatientRead
    assigned_doctor: Optional[DoctorRead] = None
    treatments: List[TreatmentRead]
```

### Step 6: Auth Router (`routers/auth.py`)

```python
# routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from database import get_db
from models import User
from schemas import UserCreate, Token
import bcrypt, jwt, os

JWT_SECRET = os.getenv("JWT_SECRET", "changeme")
JWT_ALG = "HS256"

router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

def verify_password(pw: str, hashed: str) -> bool:
    return bcrypt.checkpw(pw.encode(), hashed.encode())

def create_token(sub: str) -> str:
    return jwt.encode({"sub": sub}, JWT_SECRET, algorithm=JWT_ALG)

@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(400, detail="Email already registered")
    user = User(email=payload.email, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    token = create_token(user.email)
    return {"access_token": token, "token_type": "bearer"}

@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(400, detail="Invalid credentials")
    token = create_token(user.email)
    return {"access_token": token, "token_type": "bearer"}

def get_current_user_email(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload.get("sub")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
```

### Step 7: Patients Router (`routers/patients.py`)

```python
# routers/patients.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from database import get_db
from models import Patient, Assignment, Doctor, Treatment
from schemas import PatientCreate, PatientRead, PatientDetails
from routers.auth import get_current_user_email

router = APIRouter(prefix="/api/patients", tags=["patients"])

@router.post("", response_model=PatientRead, status_code=status.HTTP_201_CREATED)
def create_patient(payload: PatientCreate, db: Session = Depends(get_db), email: str = Depends(get_current_user_email)):
    patient = Patient(**payload.model_dump())
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient

@router.get("", response_model=List[PatientRead])
def list_patients(db: Session = Depends(get_db)):
    return db.query(Patient).all()

@router.get("/{patient_id}", response_model=PatientRead)
def get_patient(patient_id: int, db: Session = Depends(get_db)):
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

@router.put("/{patient_id}", response_model=PatientRead)
def update_patient(patient_id: int, payload: PatientCreate, db: Session = Depends(get_db), email: str = Depends(get_current_user_email)):
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    for field, value in payload.model_dump().items():
        setattr(patient, field, value)
    db.commit()
    db.refresh(patient)
    return patient

@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_patient(patient_id: int, db: Session = Depends(get_db), email: str = Depends(get_current_user_email)):
@router.get("/{patient_id}/details", response_model=PatientDetails)
def get_patient_details(patient_id: int, db: Session = Depends(get_db)):
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    assignment = db.query(Assignment).filter(Assignment.patient_id == patient_id, Assignment.discharge_date == None).first()
    assigned_doctor = db.get(Doctor, assignment.doctor_id) if assignment else None
    treatments = db.query(Treatment).filter(Treatment.patient_id == patient_id).order_by(Treatment.given_at.desc()).all()
    return {"patient": patient, "assigned_doctor": assigned_doctor, "treatments": treatments}
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    db.delete(patient)
    db.commit()
    return None
```

### Step 7: Doctors Router (`routers/doctors.py`)

```python
# routers/doctors.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from database import get_db
from models import Doctor
from schemas import DoctorCreate, DoctorRead
from routers.auth import get_current_user_email

router = APIRouter(prefix="/api/doctors", tags=["doctors"])

@router.post("", response_model=DoctorRead, status_code=status.HTTP_201_CREATED)
def create_doctor(payload: DoctorCreate, db: Session = Depends(get_db), email: str = Depends(get_current_user_email)):
    doctor = Doctor(**payload.model_dump())
    db.add(doctor)
    db.commit()
    db.refresh(doctor)
    return doctor

@router.get("", response_model=List[DoctorRead])
def list_doctors(db: Session = Depends(get_db), available_only: bool = False):
    query = db.query(Doctor)
    if available_only:
        query = query.filter(Doctor.is_available == True)
    return query.all()

@router.get("/{doctor_id}", response_model=DoctorRead)
def get_doctor(doctor_id: int, db: Session = Depends(get_db)):
    doctor = db.get(Doctor, doctor_id)
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    return doctor
```

### Step 8: Assignments Router (`routers/assignments.py`)

```python
# routers/assignments.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import date
from database import get_db
from models import Assignment, Patient, Doctor
from schemas import AssignmentCreate, AssignmentRead, PatientWithDoctor
from routers.auth import get_current_user_email

router = APIRouter(prefix="/api/assignments", tags=["assignments"])

@router.post("", response_model=AssignmentRead, status_code=status.HTTP_201_CREATED)
def assign_doctor(payload: AssignmentCreate, db: Session = Depends(get_db), email: str = Depends(get_current_user_email)):
    # Check if patient exists
    patient = db.get(Patient, payload.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Check if doctor exists and is available
    doctor = db.get(Doctor, payload.doctor_id)
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    if not doctor.is_available:
        raise HTTPException(status_code=400, detail="Doctor is not available")
    
    # Check if patient already has an active assignment
    active_assignment = db.query(Assignment).filter(
        Assignment.patient_id == payload.patient_id,
        Assignment.discharge_date == None
    ).first()
    if active_assignment:
        raise HTTPException(status_code=400, detail="Patient already has an assigned doctor")
    
    assignment = Assignment(**payload.model_dump())
    db.add(assignment)
    db.commit()
    db.refresh(assignment)
    return assignment

@router.get("", response_model=List[AssignmentRead])
def list_assignments(db: Session = Depends(get_db), active_only: bool = False):
    query = db.query(Assignment)
    if active_only:
        query = query.filter(Assignment.discharge_date == None)
    return query.all()

@router.put("/{assignment_id}/discharge", response_model=AssignmentRead)
def discharge_patient(assignment_id: int, db: Session = Depends(get_db), email: str = Depends(get_current_user_email)):
    assignment = db.get(Assignment, assignment_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")
    if assignment.discharge_date:
        raise HTTPException(status_code=400, detail="Patient already discharged")
    
    assignment.discharge_date = date.today()
    assignment.patient.status = "Discharged"
    db.commit()
    db.refresh(assignment)
    return assignment

@router.get("/patients-with-doctors", response_model=List[PatientWithDoctor])
def get_patients_with_doctors(db: Session = Depends(get_db)):
    """Get all patients with their currently assigned doctor (if any)"""
    patients = db.query(Patient).all()
    result = []
    for patient in patients:
        active_assignment = db.query(Assignment).filter(
            Assignment.patient_id == patient.id,
            Assignment.discharge_date == None
        ).first()
        
        if active_assignment:
            result.append(PatientWithDoctor(
                patient=patient,
                doctor=active_assignment.doctor,
                assignment_id=active_assignment.id
            ))
        else:
            result.append(PatientWithDoctor(
                patient=patient,
                doctor=None,
                assignment_id=None
            ))
    return result
```

### Step 10: Treatments Router (`routers/treatments.py`)

```python
# routers/treatments.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from models import Treatment, Patient, Doctor, User
from schemas import TreatmentCreate, TreatmentRead
from routers.auth import get_current_user_email

router = APIRouter(prefix="/api/treatments", tags=["treatments"])

@router.post("", response_model=TreatmentRead, status_code=status.HTTP_201_CREATED)
def create_treatment(payload: TreatmentCreate, db: Session = Depends(get_db), email: str = Depends(get_current_user_email)):
    patient = db.get(Patient, payload.patient_id)
    doctor = db.get(Doctor, payload.doctor_id)
    if not patient or not doctor:
        raise HTTPException(400, detail="Invalid patient or doctor")
    user = db.query(User).filter(User.email == email).first()
    treatment = Treatment(
        patient_id=payload.patient_id,
        doctor_id=payload.doctor_id,
        recorded_by_user_id=user.id,
        description=payload.description,
        medication=payload.medication,
        dosage=payload.dosage,
    )
    db.add(treatment)
    db.commit()
    db.refresh(treatment)
    return treatment

@router.get("/by-patient/{patient_id}", response_model=list[TreatmentRead])
def list_treatments(patient_id: int, db: Session = Depends(get_db)):
    return db.query(Treatment).filter(Treatment.patient_id == patient_id).order_by(Treatment.given_at.desc()).all()
```

### Step 11: Main Application (`main.py`)

```python
# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import Base, engine
from routers import patients, doctors, assignments, treatments, auth

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Hospital Management API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(doctors.router)
app.include_router(assignments.router)
app.include_router(treatments.router)

@app.get("/")
def root():
    return {"message": "Hospital Management API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}
```

### Step 12: Environment File (`.env`)

```
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/hospital_db
JWT_SECRET=supersecret
```

---

## Part 3: Frontend Implementation (Single HTML File)

### Complete Frontend (`index.html`)

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hospital Management System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }

        .section {
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .section h2 {
            color: #667eea;
            margin-bottom: 20px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }

        .form-group {
            margin-bottom: 15px;
        }

        label {
            display: block;
            margin-bottom: 5px;
            color: #333;
            font-weight: 500;
        }

        input, select {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            transition: border-color 0.3s;
        }

        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }

        .form-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }

        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 500;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }

        button:active {
            transform: translateY(0);
        }

        button.secondary {
            background: #6c757d;
        }

        button.danger {
            background: #dc3545;
        }

        button.success {
            background: #28a745;
        }

        .patient-card, .doctor-card {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .patient-info, .doctor-info {
            flex: 1;
        }

        .patient-info h3, .doctor-info h3 {
            color: #333;
            margin-bottom: 5px;
        }

        .patient-info p, .doctor-info p {
            color: #666;
            font-size: 14px;
            margin: 3px 0;
        }

        .patient-actions, .doctor-actions {
            display: flex;
            gap: 10px;
        }

        .status-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-top: 5px;
        }

        .status-admitted {
            background: #d4edda;
            color: #155724;
        }

        .status-discharged {
            background: #f8d7da;
            color: #721c24;
        }

        .status-unassigned {
            background: #fff3cd;
            color: #856404;
        }

        .error-message {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
            border-left: 4px solid #dc3545;
        }

        .success-message {
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
            border-left: 4px solid #28a745;
        }

        .loading {
            text-align: center;
            padding: 20px;
            color: #667eea;
        }

        select {
            cursor: pointer;
        }

        @media (max-width: 768px) {
            .form-row {
                grid-template-columns: 1fr;
            }

            .patient-card, .doctor-card {
                flex-direction: column;
                align-items: flex-start;
            }

            .patient-actions, .doctor-actions {
                width: 100%;
                margin-top: 10px;
            }

            button {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Hospital Management System</h1>

        <!-- Patient Registration Section -->
        <div class="section">
            <h2>Register New Patient</h2>
            <div id="patient-message"></div>
            <form id="patient-form">
                <div class="form-row">
                    <div class="form-group">
                        <label for="patient-name">Full Name *</label>
                        <input type="text" id="patient-name" required>
                    </div>
                    <div class="form-group">
                        <label for="patient-age">Age *</label>
                        <input type="number" id="patient-age" min="0" max="150" required>
                    </div>
                    <div class="form-group">
                        <label for="patient-gender">Gender *</label>
                        <select id="patient-gender" required>
                            <option value="">Select</option>
                            <option value="M">Male</option>
                            <option value="F">Female</option>
                            <option value="Other">Other</option>
                        </select>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label for="patient-contact">Contact *</label>
                        <input type="text" id="patient-contact" required>
                    </div>
                    <div class="form-group">
                        <label for="patient-admission-date">Admission Date *</label>
                        <input type="date" id="patient-admission-date" required>
                    </div>
                    <div class="form-group">
                        <label for="patient-location">Location (Ward/Room) *</label>
                        <input type="text" id="patient-location" placeholder="Ward A - Room 1" required>
                    </div>
                </div>
                <button type="submit">Register Patient</button>
            </form>
        </div>

        <!-- Doctor Assignment Section -->
        <div class="section">
            <h2>Assign Doctor to Patient</h2>
            <div id="assignment-message"></div>
            <form id="assignment-form">
                <div class="form-row">
                    <div class="form-group">
                        <label for="assignment-patient">Select Patient *</label>
                        <select id="assignment-patient" required>
                            <option value="">Loading patients...</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="assignment-doctor">Select Doctor *</label>
                        <select id="assignment-doctor" required>
                            <option value="">Loading doctors...</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="assignment-date">Assignment Date *</label>
                        <input type="date" id="assignment-date" required>
                    </div>
                </div>
                <button type="submit" class="success">Assign Doctor</button>
            </form>
        </div>

        <!-- Authentication Section -->
        <div class="section">
            <h2>Authentication</h2>
            <div id="auth-message"></div>
            <form id="register-form" class="form-row" style="margin-bottom:12px;">
                <div class="form-group">
                    <label for="reg-email">Register Email</label>
                    <input type="email" id="reg-email" placeholder="user@example.com">
                </div>
                <div class="form-group">
                    <label for="reg-password">Password</label>
                    <input type="password" id="reg-password">
                </div>
                <button type="submit">Register</button>
            </form>
            <form id="login-form" class="form-row">
                <div class="form-group">
                    <label for="login-email">Login Email</label>
                    <input type="email" id="login-email" placeholder="user@example.com">
                </div>
                <div class="form-group">
                    <label for="login-password">Password</label>
                    <input type="password" id="login-password">
                </div>
                <button type="submit" class="success">Login</button>
                <button type="button" id="logout-btn" class="secondary">Logout</button>
            </form>
            <p id="auth-status" style="margin-top:10px;color:#666;">Not authenticated</p>
        </div>

        <!-- Patients List Section -->
        <div class="section">
            <h2>Patients List</h2>
            <div id="patients-loading" class="loading">Loading patients...</div>
            <div id="patients-list"></div>
        </div>

        <!-- Add Doctor Section -->
        <div class="section">
            <h2>Add New Doctor</h2>
            <div id="doctor-message"></div>
            <form id="doctor-form">
                <div class="form-row">
                    <div class="form-group">
                        <label for="doctor-name">Full Name *</label>
                        <input type="text" id="doctor-name" required>
                    </div>
                    <div class="form-group">
                        <label for="doctor-specialization">Specialization *</label>
                        <input type="text" id="doctor-specialization" required>
                    </div>
                    <div class="form-group">
                        <label for="doctor-contact">Contact *</label>
                        <input type="text" id="doctor-contact" required>
                    </div>
                </div>
                <button type="submit">Add Doctor</button>
            </form>
        </div>

        <!-- Record Treatment Section -->
        <div class="section">
            <h2>Record Treatment</h2>
            <div id="treatment-message"></div>
            <form id="treatment-form">
                <div class="form-row">
                    <div class="form-group">
                        <label for="treatment-patient">Patient *</label>
                        <select id="treatment-patient" required>
                            <option value="">Loading patients...</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="treatment-doctor">Doctor *</label>
                        <select id="treatment-doctor" required>
                            <option value="">Loading doctors...</option>
                        </select>
                    </div>
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label for="treatment-desc">Description *</label>
                        <input type="text" id="treatment-desc" placeholder="IV fluids, monitoring" required>
                    </div>
                    <div class="form-group">
                        <label for="treatment-med">Medication</label>
                        <input type="text" id="treatment-med" placeholder="Paracetamol">
                    </div>
                    <div class="form-group">
                        <label for="treatment-dose">Dosage</label>
                        <input type="text" id="treatment-dose" placeholder="500mg">
                    </div>
                </div>
                <button type="submit">Save Treatment</button>
            </form>
        </div>
    </div>

    <script>
        // ============================================
        // CONFIGURATION: Change this base URL for different environments
        // ============================================
        const API_BASE_URL = 'http://localhost:8000';  // Change to your server URL: 'https://your-app.railway.app'
        // ============================================

        // Helper function to show messages
        function showMessage(elementId, message, type = 'error') {
            const element = document.getElementById(elementId);
            element.innerHTML = `<div class="${type}-message">${message}</div>`;
            setTimeout(() => {
                element.innerHTML = '';
            }, 5000);
        }

        // Helper function for API calls
        async function apiCall(endpoint, method = 'GET', body = null) {
            try {
                const options = {
                    method,
                    headers: {
                        'Content-Type': 'application/json',
                    }
                };
                const token = localStorage.getItem('token');
                if (token) {
                    options.headers['Authorization'] = `Bearer ${token}`;
                }
                if (body) {
                    options.body = JSON.stringify(body);
                }
                const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || 'An error occurred');
                }
                return data;
            } catch (error) {
                console.error('API Error:', error);
                throw error;
            }
        }

        // Auth handlers
        document.getElementById('register-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            try {
                const email = document.getElementById('reg-email').value;
                const password = document.getElementById('reg-password').value;
                const data = await apiCall('/auth/register', 'POST', { email, password });
                localStorage.setItem('token', data.access_token);
                showMessage('auth-message', 'Registered and logged in!', 'success');
                updateAuthStatus();
            } catch (err) {
                showMessage('auth-message', `Register error: ${err.message}`, 'error');
            }
        });

        document.getElementById('login-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            try {
                const email = document.getElementById('login-email').value;
                const password = document.getElementById('login-password').value;
                const form = new URLSearchParams();
                form.append('username', email);
                form.append('password', password);
                const resp = await fetch(`${API_BASE_URL}/auth/login`, { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: form });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || 'Login failed');
                localStorage.setItem('token', data.access_token);
                showMessage('auth-message', 'Logged in!', 'success');
                updateAuthStatus();
            } catch (err) {
                showMessage('auth-message', `Login error: ${err.message}`, 'error');
            }
        });

        document.getElementById('logout-btn').addEventListener('click', () => {
            localStorage.removeItem('token');
            updateAuthStatus();
        });

        function updateAuthStatus() {
            const token = localStorage.getItem('token');
            document.getElementById('auth-status').textContent = token ? 'Authenticated' : 'Not authenticated';
        }

        // Load patients for dropdown and list
        async function loadPatients() {
            try {
                const patients = await apiCall('/api/patients');
                
                // Populate patient dropdown
                const patientSelect = document.getElementById('assignment-patient');
                patientSelect.innerHTML = '<option value="">Select Patient</option>';
                patients.forEach(patient => {
                    if (patient.status === 'Admitted') {
                        const option = document.createElement('option');
                        option.value = patient.id;
                        option.textContent = `${patient.name} (Age: ${patient.age})`;
                        patientSelect.appendChild(option);
                    }
                });

                // Populate treatment patient dropdown
                const treatmentPatient = document.getElementById('treatment-patient');
                treatmentPatient.innerHTML = '<option value="">Select Patient</option>';
                patients.forEach(patient => {
                    const option = document.createElement('option');
                    option.value = patient.id;
                    option.textContent = `${patient.name} (#${patient.id})`;
                    treatmentPatient.appendChild(option);
                });

                // Load patients with doctors for display
                await loadPatientsWithDoctors();
            } catch (error) {
                console.error('Failed to load patients:', error);
            }
        }

        // Load patients with their assigned doctors
        async function loadPatientsWithDoctors() {
            try {
                document.getElementById('patients-loading').style.display = 'block';
                const patientsWithDoctors = await apiCall('/api/assignments/patients-with-doctors');
                document.getElementById('patients-loading').style.display = 'none';
                
                const listDiv = document.getElementById('patients-list');
                if (patientsWithDoctors.length === 0) {
                    listDiv.innerHTML = '<p>No patients registered yet.</p>';
                    return;
                }

                listDiv.innerHTML = patientsWithDoctors.map(item => {
                    const patient = item.patient;
                    const doctor = item.doctor;
                    const assignmentId = item.assignment_id;
                    const statusClass = patient.status === 'Admitted' ? 'status-admitted' : 'status-discharged';
                    const statusBadge = `<span class="status-badge ${statusClass}">${patient.status}</span>`;
                    
                    let doctorInfo = '';
                    if (doctor) {
                        doctorInfo = `<p><strong>Assigned Doctor:</strong> ${doctor.name} (${doctor.specialization})</p>`;
                    } else {
                        doctorInfo = `<p class="status-unassigned status-badge">No doctor assigned</p>`;
                    }

                    let dischargeButton = '';
                    if (assignmentId && patient.status === 'Admitted') {
                        dischargeButton = `<button class="danger" onclick="dischargePatient(${assignmentId})">Discharge Patient</button>`;
                    }

                    return `
                        <div class="patient-card">
                            <div class="patient-info">
                                <h3>${patient.name}</h3>
                                <p><strong>Age:</strong> ${patient.age} | <strong>Gender:</strong> ${patient.gender}</p>
                                <p><strong>Contact:</strong> ${patient.contact}</p>
                                <p><strong>Admission Date:</strong> ${patient.admission_date}</p>
                                <p><strong>Location:</strong> ${patient.location || 'N/A'}</p>
                                ${doctorInfo}
                                ${statusBadge}
                            </div>
                            <div class="patient-actions">
                                <button class="secondary" onclick="viewDetails(${patient.id})">View Details</button>
                                ${dischargeButton}
                            </div>
                        </div>
                    `;
                }).join('');
            } catch (error) {
                document.getElementById('patients-loading').style.display = 'none';
                document.getElementById('patients-list').innerHTML = 
                    `<div class="error-message">Failed to load patients: ${error.message}</div>`;
            }
        }

        // Load doctors for dropdown
        async function loadDoctors() {
            try {
                const doctors = await apiCall('/api/doctors?available_only=true');
                const doctorSelect = document.getElementById('assignment-doctor');
                doctorSelect.innerHTML = '<option value="">Select Doctor</option>';
                doctors.forEach(doctor => {
                    const option = document.createElement('option');
                    option.value = doctor.id;
                    option.textContent = `${doctor.name} - ${doctor.specialization}`;
                    doctorSelect.appendChild(option);
                });

                // Populate treatment doctor dropdown
                const tDoc = document.getElementById('treatment-doctor');
                tDoc.innerHTML = '<option value="">Select Doctor</option>';
                doctors.forEach(doctor => {
                    const option = document.createElement('option');
                    option.value = doctor.id;
                    option.textContent = `${doctor.name} - ${doctor.specialization}`;
                    tDoc.appendChild(option);
                });
            } catch (error) {
                console.error('Failed to load doctors:', error);
                document.getElementById('assignment-doctor').innerHTML = 
                    '<option value="">Error loading doctors</option>';
            }
        }

        // Register new patient
        document.getElementById('patient-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            try {
                const patientData = {
                    name: document.getElementById('patient-name').value,
                    age: parseInt(document.getElementById('patient-age').value),
                    gender: document.getElementById('patient-gender').value,
                    contact: document.getElementById('patient-contact').value,
                    admission_date: document.getElementById('patient-admission-date').value,
                    location: document.getElementById('patient-location').value
                };

                await apiCall('/api/patients', 'POST', patientData);
                showMessage('patient-message', 'Patient registered successfully!', 'success');
                document.getElementById('patient-form').reset();
                await loadPatients();
            } catch (error) {
                showMessage('patient-message', `Error: ${error.message}`, 'error');
            }
        });

        // Assign doctor to patient
        document.getElementById('assignment-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            try {
                const assignmentData = {
                    patient_id: parseInt(document.getElementById('assignment-patient').value),
                    doctor_id: parseInt(document.getElementById('assignment-doctor').value),
                    assigned_date: document.getElementById('assignment-date').value
                };

                await apiCall('/api/assignments', 'POST', assignmentData);
                showMessage('assignment-message', 'Doctor assigned successfully!', 'success');
                document.getElementById('assignment-form').reset();
                await loadPatients();
                await loadDoctors();
            } catch (error) {
                showMessage('assignment-message', `Error: ${error.message}`, 'error');
            }
        });

        // Add new doctor
        document.getElementById('doctor-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            try {
                const doctorData = {
                    name: document.getElementById('doctor-name').value,
                    specialization: document.getElementById('doctor-specialization').value,
                    contact: document.getElementById('doctor-contact').value,
                    is_available: true
                };

                await apiCall('/api/doctors', 'POST', doctorData);
                showMessage('doctor-message', 'Doctor added successfully!', 'success');
                document.getElementById('doctor-form').reset();
                await loadDoctors();
            } catch (error) {
                showMessage('doctor-message', `Error: ${error.message}`, 'error');
            }
        });

        // Save treatment
        document.getElementById('treatment-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            try {
                const payload = {
                    patient_id: parseInt(document.getElementById('treatment-patient').value),
                    doctor_id: parseInt(document.getElementById('treatment-doctor').value),
                    description: document.getElementById('treatment-desc').value,
                    medication: document.getElementById('treatment-med').value,
                    dosage: document.getElementById('treatment-dose').value,
                };
                await apiCall('/api/treatments', 'POST', payload);
                showMessage('treatment-message', 'Treatment recorded!', 'success');
                document.getElementById('treatment-form').reset();
                await loadPatients();
            } catch (err) {
                showMessage('treatment-message', `Error: ${err.message}`, 'error');
            }
        });

        // View details for a patient
        async function viewDetails(patientId) {
            try {
                const details = await apiCall(`/api/patients/${patientId}/details`);
                const treatments = details.treatments.map(t => `<li>${new Date(t.given_at).toLocaleString()}: ${t.description}${t.medication ? ' - ' + t.medication : ''} ${t.dosage ? '('+t.dosage+')' : ''}</li>`).join('');
                alert(`Patient: ${details.patient.name}\nStatus: ${details.patient.status}\nLocation: ${details.patient.location}\nDoctor: ${details.assigned_doctor ? details.assigned_doctor.name : 'None'}\n\nTreatments:\n- ${treatments || 'No treatments recorded.'}`);
            } catch (err) {
                showMessage('patient-message', `Error loading details: ${err.message}`, 'error');
            }
        }

        // Discharge patient
        async function dischargePatient(assignmentId) {
            if (!confirm('Are you sure you want to discharge this patient?')) {
                return;
            }
            try {
                await apiCall(`/api/assignments/${assignmentId}/discharge`, 'PUT');
                showMessage('patient-message', 'Patient discharged successfully!', 'success');
                await loadPatients();
            } catch (error) {
                showMessage('patient-message', `Error: ${error.message}`, 'error');
            }
        }

        // Set today's date as default for date inputs
        document.addEventListener('DOMContentLoaded', () => {
            const today = new Date().toISOString().split('T')[0];
            document.getElementById('patient-admission-date').value = today;
            document.getElementById('assignment-date').value = today;

            // Initial load
            updateAuthStatus();
            loadPatients();
            loadDoctors();
        });
    </script>
</body>
</html>
```

---

## Part 4: Setup and Deployment Instructions

### Local Setup

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Create Database:**
```bash
# Create PostgreSQL database
createdb hospital_db
```

3. **Set Environment Variables:**
```bash
# .env file
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/hospital_db
```

4. **Run Migrations (Optional - using Alembic):**
```bash
alembic revision --autogenerate -m "init hospital tables"
alembic upgrade head
```

Or create tables automatically (they're created in `main.py` using `Base.metadata.create_all`)

5. **Start Backend Server:**
```bash
uvicorn main:app --reload
```

6. **Open Frontend:**
- Open `index.html` in a web browser
- Or serve it using a simple HTTP server:
```bash
python -m http.server 8080
```
Then visit `http://localhost:8080`

### Testing the Application

1. **Add a Doctor:**
   - Fill in doctor details in "Add New Doctor" section
   - Click "Add Doctor"

2. **Register a Patient:**
   - Fill in patient details in "Register New Patient" section
   - Click "Register Patient"

3. **Assign Doctor:**
   - Select a patient and doctor from dropdowns
   - Click "Assign Doctor"

4. **View Patients:**
   - All patients are displayed in "Patients List" section
   - See assigned doctors and status

5. **Discharge Patient:**
   - Click "Discharge Patient" button on an admitted patient
   - Patient status changes to "Discharged"

### Switching Environments

To switch from localhost to production:

1. **In `index.html`**, change the `API_BASE_URL`:
```javascript
const API_BASE_URL = 'https://your-app.railway.app';
```

2. **Deploy backend to Railway** and update the frontend accordingly.

---

## Summary

This tutorial demonstrated:
- ✅ **Database Design**: ERD with proper relationships
- ✅ **FastAPI Backend**: Complete REST API with CRUD operations
- ✅ **Frontend**: Single HTML file with embedded CSS/JS
- ✅ **Full Functionality**: Patient registration, doctor assignment, discharge
- ✅ **Clean Architecture**: Separated routers, models, schemas
- ✅ **Production Ready**: CORS enabled, proper error handling

The application is minimal yet comprehensive, demonstrating real-world patterns for building full-stack applications with FastAPI and vanilla JavaScript.

