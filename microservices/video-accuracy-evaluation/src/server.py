# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import uvicorn
from http import HTTPStatus
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Annotated
from .config import config
from .document import validate_files, save_files_to_tmp
from .evaluate import Evaluator
from .logger import logger


app = FastAPI(title=config.APP_DISPLAY_NAME, root_path="/v1/eval")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=os.getenv("CORS_ALLOW_METHODS", "*").split(","),
    allow_headers=os.getenv("CORS_ALLOW_HEADERS", "*").split(","),
)

evaluator = Evaluator(
    bert_scorer_model_name=config.BERT_SCORER_MODEL_ID,
    sbert_model_name=config.SBERT_MODEL_ID,
    nli_model_name=config.NLI_MODEL_ID
)

class EvaluateData(BaseModel):
    generated: str
    reference: str
    question: str = ""
    metrics: Optional[List[str]] = None

@app.get(
    "/health",
    tags=["Status APIs"],
    summary="Check the health of the API service"
)
async def check_health():
    """
    Checks the health status of the application.
    This asynchronous function is used to verify that the application is running
    and healthy by returning a simple status message.

    Returns:
        dict: A dictionary containing the health status of the application.
    """

    return {"status": "Success", "message": "Service is up and running."}


@app.post(
    "/semantic-score",
    tags=["Evaluation APIs"],
    summary="Get semantic score from the datasets",
)
def get_semantic_score(input_data: EvaluateData):
    try:
        return evaluator._calculate_semantic_score(input_data.generated, input_data.reference)

    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/bert-score",
    tags=["Evaluation APIs"],
    summary="Get BERT score from the datasets",
)
def get_bert_score(input_data: EvaluateData):
    try:
        return evaluator._calculate_bert_score(input_data.generated, input_data.reference)

    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/rouge-score",
    tags=["Evaluation APIs"],
    summary="Get ROUGE score from the datasets",
)
def get_rouge_score(input_data: EvaluateData):
    try:
        return evaluator._calculate_rouge_score(input_data.generated, input_data.reference)

    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/average-score",
    tags=["Evaluation APIs"],
    summary="Get average score from the datasets for different metrics",
)
def get_average_score(input_data: EvaluateData):
    try:
        return evaluator._calculate_average_scores([(input_data.generated, input_data.reference)])

    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/factual-entailment",
    tags=["Evaluation APIs"],
    summary="Get factual entailment label from the datasets",
)
def get_factual_entailment(input_data: EvaluateData):
    try:
        return evaluator._evaluate_factual_consistency(input_data.generated, input_data.reference)

    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/evaluate",
    tags=["Evaluation APIs"],
    summary="Evaluate datasets for accuracy",
)
async def evaluate_video_accuracy(
    file: Annotated[
        UploadFile,
        File(description="Upload one file containing generated and reference data.")
    ],
):
    try:
        status = validate_files([file])
        if status is False:
            logger.exception("Unsupported file format.")
            raise HTTPException(
                status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported file format. Please upload a .tsv file."
            )

        # Save the file in /tmp/documents to load it later
        tmp_files = await save_files_to_tmp([file])
        if tmp_files is None or len(tmp_files) == 0:
            logger.exception(f"Error saving file.")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="Error saving file."
            )

        result = evaluator.run_evaluation_from_file(tmp_files[0])

        return result

    except HTTPException:
        # Re-raise HTTPException without modification
        raise

    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


if __name__ == "__main__":
    uvicorn.run("app", host="0.0.0.0", port=9000)
