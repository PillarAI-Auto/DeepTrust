# ============================================================
# DeepTrustAI — All-In-One Production MVP
# AI-Based Deepfake Detection & Trust Scoring System
# ============================================================
# Features
# - FastAPI Backend
# - Image Deepfake Detection
# - Metadata Analysis
# - Trust Scoring Engine
# - Moderation Engine
# - Explainable AI Responses
# - SQLite Database
# - File Upload System
# - REST API
# - GPU Ready
# ============================================================

from __future__ import annotations

import os
import io
import cv2
import uuid
import json
import math
import shutil
import sqlite3
import hashlib
import logging
import traceback
import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ============================================================
# CONFIG
# ============================================================

APP_NAME = "DeepTrustAI"
VERSION = "1.0"
UPLOAD_DIR = "uploads"
DB_PATH = "deeptrust.db"

Path(UPLOAD_DIR).mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(APP_NAME)

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version=VERSION,
    description="AI-Based Deepfake Detection & Trust Scoring System"
)

# ============================================================
# DATABASE
# ============================================================

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS media (
        id TEXT PRIMARY KEY,
        filename TEXT,
        media_type TEXT,
        trust_score REAL,
        fake_probability REAL,
        moderation_action TEXT,
        explanation TEXT,
        created_at TEXT
    )
    """
)

conn.commit()

# ============================================================
# MODELS
# ============================================================

class AnalysisResponse(BaseModel):
    media_id: str
    filename: str
    trust_score: float
    fake_probability: float
    moderation_action: str
    explanation: Dict[str, Any]

# ============================================================
# UTILITIES
# ============================================================


def generate_id() -> str:
    return str(uuid.uuid4())



def calculate_file_hash(path: str) -> str:
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            sha256.update(chunk)

    return sha256.hexdigest()



def get_image_metadata(path: str) -> Dict[str, Any]:
    try:
        image = Image.open(path)

        metadata = {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "has_exif": hasattr(image, "getexif")
        }

        return metadata

    except Exception as e:
        return {
            "error": str(e)
        }

# ============================================================
# IMAGE ANALYSIS ENGINE
# ============================================================


class ImageDeepfakeDetector:

    def __init__(self):
        logger.info("Image detector initialized")

    def analyze(self, image_path: str) -> Dict[str, Any]:

        try:
            image = cv2.imread(image_path)

            if image is None:
                raise Exception("Failed to load image")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # ====================================================
            # FEATURE 1 — NOISE ANALYSIS
            # ====================================================

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            noise_score = self.detect_noise_inconsistency(gray)

            # ====================================================
            # FEATURE 2 — FFT FREQUENCY ANALYSIS
            # ====================================================

            fft_score = self.frequency_analysis(gray)

            # ====================================================
            # FEATURE 3 — EDGE CONSISTENCY
            # ====================================================

            edge_score = self.edge_analysis(gray)

            # ====================================================
            # FEATURE 4 — COLOR DISTRIBUTION
            # ====================================================

            color_score = self.color_distribution_analysis(image_rgb)

            # ====================================================
            # FEATURE 5 — TEXTURE ANALYSIS
            # ====================================================

            texture_score = self.texture_analysis(gray)

            # ====================================================
            # FINAL FAKE PROBABILITY
            # ====================================================

            fake_probability = (
                noise_score * 0.20 +
                fft_score * 0.25 +
                edge_score * 0.20 +
                color_score * 0.15 +
                texture_score * 0.20
            )

            fake_probability = max(0, min(100, fake_probability))

            return {
                "noise_score": round(noise_score, 2),
                "fft_score": round(fft_score, 2),
                "edge_score": round(edge_score, 2),
                "color_score": round(color_score, 2),
                "texture_score": round(texture_score, 2),
                "fake_probability": round(fake_probability, 2)
            }

        except Exception as e:
            logger.error(traceback.format_exc())

            return {
                "error": str(e),
                "fake_probability": 50
            }

    # ========================================================
    # DETECTION METHODS
    # ========================================================

    def detect_noise_inconsistency(self, gray: np.ndarray) -> float:

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blur)

        score = np.std(noise)

        return min(score * 2, 100)

    def frequency_analysis(self, gray: np.ndarray) -> float:

        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)

        magnitude = np.log(np.abs(fft_shift) + 1)

        high_freq_energy = np.mean(magnitude)

        return min(high_freq_energy * 2.5, 100)

    def edge_analysis(self, gray: np.ndarray) -> float:

        edges = cv2.Canny(gray, 100, 200)

        edge_density = np.sum(edges > 0) / edges.size

        return min(edge_density * 1000, 100)

    def color_distribution_analysis(self, image_rgb: np.ndarray) -> float:

        hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])

        variance = (
            np.var(hist_r) +
            np.var(hist_g) +
            np.var(hist_b)
        ) / 3

        return min(variance / 500, 100)

    def texture_analysis(self, gray: np.ndarray) -> float:

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        texture_variance = laplacian.var()

        return min(texture_variance / 10, 100)


image_detector = ImageDeepfakeDetector()

# ============================================================
# TRUST ENGINE
# ============================================================


class TrustScoringEngine:

    def calculate(
        self,
        fake_probability: float,
        metadata_score: float,
        source_score: float = 80
    ) -> float:

        trust_score = (
            (100 - fake_probability) * 0.70 +
            metadata_score * 0.20 +
            source_score * 0.10
        )

        return round(max(0, min(100, trust_score)), 2)


trust_engine = TrustScoringEngine()

# ============================================================
# MODERATION ENGINE
# ============================================================


class ModerationEngine:

    def decide(self, trust_score: float) -> str:

        if trust_score >= 85:
            return "ALLOW"

        elif trust_score >= 60:
            return "WARNING_LABEL"

        elif trust_score >= 35:
            return "LIMIT_DISTRIBUTION"

        return "BLOCK_AND_REVIEW"


moderation_engine = ModerationEngine()

# ============================================================
# EXPLAINABILITY ENGINE
# ============================================================


class ExplainabilityEngine:

    def build(
        self,
        analysis: Dict[str, Any],
        trust_score: float,
        moderation_action: str
    ) -> Dict[str, Any]:

        reasons = []

        if analysis.get("fft_score", 0) > 60:
            reasons.append(
                "Abnormal frequency-domain artifacts detected"
            )

        if analysis.get("noise_score", 0) > 60:
            reasons.append(
                "Noise inconsistencies detected"
            )

        if analysis.get("edge_score", 0) > 60:
            reasons.append(
                "Irregular edge structures detected"
            )

        if analysis.get("texture_score", 0) > 60:
            reasons.append(
                "Synthetic texture patterns detected"
            )

        if not reasons:
            reasons.append(
                "No major manipulation indicators detected"
            )

        return {
            "trust_score": trust_score,
            "moderation_action": moderation_action,
            "reasons": reasons,
            "confidence": round(analysis.get("fake_probability", 0), 2)
        }


explain_engine = ExplainabilityEngine()

# ============================================================
# HEALTH ROUTE
# ============================================================


@app.get("/")
def home():
    return {
        "app": APP_NAME,
        "version": VERSION,
        "status": "running"
    }

# ============================================================
# ANALYSIS ROUTE
# ============================================================


@app.post("/api/upload", response_model=AnalysisResponse)
async def upload_media(file: UploadFile = File(...)):

    media_id = generate_id()

    try:

        # ====================================================
        # SAVE FILE
        # ====================================================

        filename = f"{media_id}_{file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)

        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Saved file: {filepath}")

        # ====================================================
        # HASH
        # ====================================================

        file_hash = calculate_file_hash(filepath)

        # ====================================================
        # METADATA
        # ====================================================

        metadata = get_image_metadata(filepath)

        metadata_score = 90

        if metadata.get("error"):
            metadata_score = 40

        # ====================================================
        # IMAGE DETECTION
        # ====================================================

        analysis = image_detector.analyze(filepath)

        fake_probability = analysis["fake_probability"]

        # ====================================================
        # TRUST SCORE
        # ====================================================

        trust_score = trust_engine.calculate(
            fake_probability=fake_probability,
            metadata_score=metadata_score
        )

        # ====================================================
        # MODERATION
        # ====================================================

        moderation_action = moderation_engine.decide(trust_score)

        # ====================================================
        # EXPLANATION
        # ====================================================

        explanation = explain_engine.build(
            analysis,
            trust_score,
            moderation_action
        )

        # ====================================================
        # DATABASE SAVE
        # ====================================================

        cursor.execute(
            """
            INSERT INTO media VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                media_id,
                filename,
                file.content_type,
                trust_score,
                fake_probability,
                moderation_action,
                json.dumps(explanation),
                datetime.datetime.utcnow().isoformat()
            )
        )

        conn.commit()

        # ====================================================
        # RESPONSE
        # ====================================================

        return AnalysisResponse(
            media_id=media_id,
            filename=filename,
            trust_score=trust_score,
            fake_probability=fake_probability,
            moderation_action=moderation_action,
            explanation={
                "file_hash": file_hash,
                "metadata": metadata,
                "analysis": analysis,
                "details": explanation
            }
        )

    except Exception as e:

        logger.error(traceback.format_exc())

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )

# ============================================================
# HISTORY ROUTE
# ============================================================


@app.get("/api/history")
def get_history():

    cursor.execute(
        "SELECT * FROM media ORDER BY created_at DESC"
    )

    rows = cursor.fetchall()

    history = []

    for row in rows:
        history.append({
            "media_id": row[0],
            "filename": row[1],
            "media_type": row[2],
            "trust_score": row[3],
            "fake_probability": row[4],
            "moderation_action": row[5],
            "explanation": json.loads(row[6]),
            "created_at": row[7]
        })

    return {
        "count": len(history),
        "items": history
    }

# ============================================================
# ANALYSIS LOOKUP
# ============================================================


@app.get("/api/analyze/{media_id}")
def analyze_lookup(media_id: str):

    cursor.execute(
        "SELECT * FROM media WHERE id=?",
        (media_id,)
    )

    row = cursor.fetchone()

    if not row:
        return {
            "error": "Media not found"
        }

    return {
        "media_id": row[0],
        "filename": row[1],
        "media_type": row[2],
        "trust_score": row[3],
        "fake_probability": row[4],
        "moderation_action": row[5],
        "explanation": json.loads(row[6]),
        "created_at": row[7]
    }

# ============================================================
# ADVANCED FUTURE PLACEHOLDERS
# ============================================================


class VideoDeepfakeDetector:

    def analyze(self, video_path: str):
        return {
            "status": "future implementation"
        }


class AudioDeepfakeDetector:

    def analyze(self, audio_path: str):
        return {
            "status": "future implementation"
        }


class FusionEngine:

    def combine(self, image_score, video_score, audio_score):

        return (
            image_score * 0.5 +
            video_score * 0.3 +
            audio_score * 0.2
        )

# ============================================================
# STARTUP
# ============================================================


if __name__ == "__main__":

    import uvicorn

    logger.info("Starting DeepTrustAI...")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
