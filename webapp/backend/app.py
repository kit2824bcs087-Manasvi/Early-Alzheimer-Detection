import os
import uuid

import torch
import torch.nn as nn

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
from torchvision import transforms, models

import blockchain_service


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final",
    "alzheimer_mobilenetv2_final.pth"
)

IMAGE_SIZE = 224

CLASS_NAMES = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non Demented",
    "Very mild Dementia"
]

NUM_CLASSES = 4


# ============================================================
# DETECTION CONFIGURATION
# ============================================================

NON_DEMENTED_CLASS = "Non Demented"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("       ALZORA AI - ALZHEIMER MRI ANALYSIS API")
print("=" * 70)

print(
    "\nDevice:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    model = models.mobilenet_v2(
        weights=None
    )

    input_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = nn.Sequential(

        nn.Dropout(
            p=0.4
        ),

        nn.Linear(
            input_features,
            NUM_CLASSES
        )

    )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading final model...")

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Final model not found:\n{MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model = create_model()

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(
    DEVICE
)

model.eval()

print(
    "✓ Final model loaded successfully"
)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Checkpoint epoch:",
    checkpoint.get(
        "epoch",
        "Unknown"
    )
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )

])


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(

    title="Alzora AI - Alzheimer MRI Analysis API",

    description=(
        "Research/demo API for MRI-based dementia "
        "detection and four-class classification "
        "using MobileNetV2 with blockchain-based "
        "analysis integrity verification."
    ),

    version="2.1.0"

)


# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=[
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174"
],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]

)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {

        "message":
            "Alzora AI MRI Analysis API is running",

        "status":
            "online",

        "model":
            "MobileNetV2",

        "task":
            "Dementia detection and four-class classification",

        "blockchain":
            "Enabled",

        "classes":
            CLASS_NAMES

    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    blockchain_status = False

    try:

        blockchain_status = (
            blockchain_service.w3.is_connected()
        )

    except Exception:

        blockchain_status = False

    return {

        "status":
            "healthy",

        "device":
            str(DEVICE),

        "model_loaded":
            True,

        "model":
            "MobileNetV2",

        "blockchain_connected":
            blockchain_status

    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Check file type
    # --------------------------------------------------------

    allowed_types = [

        "image/jpeg",

        "image/png",

        "image/jpg",

        "image/bmp",

        "image/webp"

    ]

    if file.content_type not in allowed_types:

        raise HTTPException(

            status_code=400,

            detail=(
                "Please upload a valid MRI image "
                "(JPG, JPEG, PNG, BMP or WEBP)."
            )

        )

    try:

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image_bytes = await file.read()

        if not image_bytes:

            raise HTTPException(

                status_code=400,

                detail="Uploaded MRI image is empty."

            )

        from io import BytesIO

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        # ----------------------------------------------------
        # Preprocess image
        # ----------------------------------------------------

        image_tensor = transform(
            image
        )

        image_tensor = image_tensor.unsqueeze(
            0
        )

        image_tensor = image_tensor.to(
            DEVICE
        )

        # ----------------------------------------------------
        # MobileNetV2 prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_index = (
                torch.max(
                    probabilities,
                    dim=1
                )
            )

        # ----------------------------------------------------
        # Get predicted class
        # ----------------------------------------------------

        predicted_class = CLASS_NAMES[
            predicted_index.item()
        ]

        confidence_value = (
            confidence.item()
            * 100
        )

        # ====================================================
        # DETECTION LOGIC
        # ====================================================

        if predicted_class == NON_DEMENTED_CLASS:

            detection_status = "Not Detected"

            detection_label = (
                "Dementia Indication Not Detected"
            )

        else:

            detection_status = "Detected"

            detection_label = (
                "Dementia Indication Detected"
            )

        # ----------------------------------------------------
        # All class probabilities
        # ----------------------------------------------------

        class_probabilities = {}

        for index, class_name in enumerate(
            CLASS_NAMES
        ):

            class_probabilities[
                class_name
            ] = round(

                probabilities[
                    0,
                    index
                ].item()
                * 100,

                2

            )

        # ----------------------------------------------------
        # Detection probability
        # ----------------------------------------------------
        #
        # Detection probability is calculated as the combined
        # probability of the three dementia-related classes.
        #
        # Dementia-related classes:
        #   Mild Dementia
        #   Moderate Dementia
        #   Very mild Dementia
        #
        # Non Demented:
        #   Non Demented
        #
        # ----------------------------------------------------

        dementia_probability = round(

            class_probabilities["Mild Dementia"]
            +
            class_probabilities["Moderate Dementia"]
            +
            class_probabilities["Very mild Dementia"],

            2

        )

        non_demented_probability = (
            class_probabilities["Non Demented"]
        )

        # ====================================================
        # BLOCKCHAIN INTEGRATION
        # ====================================================
        #
        # The actual MRI image is NOT stored on blockchain.
        #
        # Only the SHA-256 hash and analysis metadata are
        # registered on the blockchain.
        #
        # This provides an integrity record for the analysis.
        #
        # ====================================================

        blockchain_record = None

        try:

            # ------------------------------------------------
            # Generate SHA-256 hash of MRI
            # ------------------------------------------------

            mri_hash = (
                blockchain_service.generate_mri_hash(
                    image_bytes
                )
            )

            # ------------------------------------------------
            # Generate unique analysis record ID
            # ------------------------------------------------

            record_id = (
                "ALZ-"
                +
                uuid.uuid4().hex[:10].upper()
            )

            # ------------------------------------------------
            # Register prediction on blockchain
            # ------------------------------------------------

            blockchain_record = (
                blockchain_service.register_analysis(

                    record_id=record_id,

                    mri_hash=mri_hash,

                    prediction=predicted_class,

                    confidence=confidence_value / 100,

                    model_version="MobileNetV2"

                )
            )

            print(
                "\n✓ Blockchain record created"
            )

            print(
                "Record ID:",
                blockchain_record.get(
                    "record_id"
                )
            )

            print(
                "Transaction:",
                blockchain_record.get(
                    "transaction_hash"
                )
            )

            print(
                "Block:",
                blockchain_record.get(
                    "block_number"
                )
            )

        except Exception as blockchain_error:

            # ------------------------------------------------
            # Blockchain failure should not stop the AI
            # prediction.
            # ------------------------------------------------

            print(
                "\n⚠ Blockchain registration failed:"
            )

            print(
                str(blockchain_error)
            )

            blockchain_record = {

                "status":
                    "failed",

                "record_id":
                    None,

                "mri_hash":
                    None,

                "transaction_hash":
                    None,

                "block_number":
                    None,

                "error":
                    "Blockchain registration failed"

            }

        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        return {

            "success":
                True,

            "filename":
                file.filename,

            # =================================================
            # Detection result
            # =================================================

            "detection":
                detection_status,

            "detection_label":
                detection_label,

            "dementia_probability":
                dementia_probability,

            "non_demented_probability":
                non_demented_probability,

            # =================================================
            # Classification result
            # =================================================

            "prediction":
                predicted_class,

            "confidence":
                round(
                    confidence_value,
                    2
                ),

            # =================================================
            # Four-class probabilities
            # =================================================

            "probabilities":
                class_probabilities,

            # =================================================
            # Model information
            # =================================================

            "model":
                "MobileNetV2",

            "input_size":
                "224x224",

            "num_classes":
                NUM_CLASSES,

            # =================================================
            # Blockchain information
            # =================================================

            "blockchain":
                blockchain_record,

            # =================================================
            # Note
            # =================================================

            "note":
                (
                    "This result represents model output "
                    "for research/demo purposes only and "
                    "is not a medical diagnosis."
                )

        }

    except HTTPException:

        raise

    except Exception as error:

        print(
            "Prediction error:",
            str(error)
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Unable to process the MRI image."
            )

        )


# ============================================================
# BLOCKCHAIN VERIFICATION ENDPOINT
# ============================================================

@app.get("/blockchain/verify/{record_id}")
def verify_blockchain_record(
    record_id: str
):

    try:

        record = (
            blockchain_service.get_analysis(
                record_id
            )
        )

        if not record.get("record_id"):

            raise HTTPException(

                status_code=404,

                detail="Blockchain record not found."

            )

        return {

            "success":
                True,

            "record":
                record

        }

    except HTTPException:

        raise

    except Exception as error:

        print(
            "Blockchain verification error:",
            str(error)
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Unable to retrieve blockchain record."
            )

        )


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        app,

        host="127.0.0.1",

        port=8000

    )