import json
import hashlib
from pathlib import Path
from web3 import Web3


# ============================================================
# BLOCKCHAIN CONFIGURATION
# ============================================================

RPC_URL = "http://127.0.0.1:8545"

CONTRACT_ADDRESS = Web3.to_checksum_address(
    "0x5FbDB2315678afecb367f032d93F642f64180aa3"
)

# Hardhat local development account #0
PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"


# ============================================================
# CONNECT TO LOCAL BLOCKCHAIN
# ============================================================

w3 = Web3(Web3.HTTPProvider(RPC_URL))

if not w3.is_connected():
    raise RuntimeError(
        "Could not connect to Hardhat blockchain at "
        "http://127.0.0.1:8545"
    )


# ============================================================
# LOAD SMART CONTRACT ABI
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ABI_PATH = (
    PROJECT_ROOT
    / "blockchain"
    / "artifacts"
    / "contracts"
    / "AlzoraRegistry.sol"
    / "AlzoraRegistry.json"
)


if not ABI_PATH.exists():
    raise FileNotFoundError(
        f"Contract ABI not found at: {ABI_PATH}"
    )


with open(ABI_PATH, "r", encoding="utf-8") as file:
    contract_artifact = json.load(file)

CONTRACT_ABI = contract_artifact["abi"]


contract = w3.eth.contract(
    address=CONTRACT_ADDRESS,
    abi=CONTRACT_ABI
)


# ============================================================
# WALLET
# ============================================================

ACCOUNT = w3.eth.account.from_key(PRIVATE_KEY)

ACCOUNT_ADDRESS = ACCOUNT.address


# ============================================================
# MRI HASH
# ============================================================

def generate_mri_hash(file_bytes: bytes) -> str:
    """
    Generate SHA-256 hash for the uploaded MRI file.
    The actual MRI image is NOT stored on blockchain.
    Only its hash is recorded.
    """

    return hashlib.sha256(file_bytes).hexdigest()


# ============================================================
# REGISTER ANALYSIS ON BLOCKCHAIN
# ============================================================

def register_analysis(
    record_id: str,
    mri_hash: str,
    prediction: str,
    confidence: float,
    model_version: str = "MobileNetV2"
):
    """
    Register an MRI analysis result on the local blockchain.
    """

    # Convert confidence to integer percentage
    confidence_int = int(round(confidence * 100))

    nonce = w3.eth.get_transaction_count(
        ACCOUNT_ADDRESS,
        "pending"
    )

    transaction = contract.functions.registerAnalysis(
        record_id,
        mri_hash,
        prediction,
        confidence_int,
        model_version
    ).build_transaction({
        "from": ACCOUNT_ADDRESS,
        "nonce": nonce,
        "gas": 500000,
        "gasPrice": w3.eth.gas_price,
    })

    signed_transaction = w3.eth.account.sign_transaction(
        transaction,
        private_key=PRIVATE_KEY
    )

    tx_hash = w3.eth.send_raw_transaction(
        signed_transaction.raw_transaction
    )

    receipt = w3.eth.wait_for_transaction_receipt(
        tx_hash
    )

    return {
        "record_id": record_id,
        "mri_hash": mri_hash,
        "prediction": prediction,
        "confidence": confidence,
        "model_version": model_version,
        "transaction_hash": tx_hash.hex(),
        "block_number": receipt["blockNumber"],
        "contract_address": CONTRACT_ADDRESS,
        "submitted_by": ACCOUNT_ADDRESS,
        "status": "confirmed"
    }


# ============================================================
# VERIFY MRI INTEGRITY
# ============================================================

def verify_mri(record_id: str, mri_hash: str) -> bool:
    """
    Verify whether the supplied MRI hash matches
    the hash stored on blockchain.
    """

    return contract.functions.verifyMRI(
        record_id,
        mri_hash
    ).call()


# ============================================================
# GET ANALYSIS RECORD
# ============================================================

def get_analysis(record_id: str):
    """
    Retrieve an analysis record from blockchain.
    """

    result = contract.functions.getAnalysis(
        record_id
    ).call()

    return {
        "record_id": result[0],
        "mri_hash": result[1],
        "prediction": result[2],
        "confidence": result[3] / 100,
        "model_version": result[4],
        "timestamp": result[5],
        "submitted_by": result[6],
    }