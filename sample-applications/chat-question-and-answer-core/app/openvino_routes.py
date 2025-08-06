from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from .openvino_backend import get_available_devices, get_device_property
from .logger import logger

router = APIRouter()

@router.get("/devices", tags=["Device API"], summary="Get available devices list")
async def get_devices():
    try:
        devices = get_available_devices()
        return {"devices": devices}
    except Exception as e:
        logger.exception("Error getting devices list.", error=e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/devices/{device}", tags=["Device API"], summary="Get device property")
async def get_device_info(device: str = ""):
    try:
        available_devices = get_available_devices()
        if device not in available_devices:
            raise HTTPException(
                status_code=404,
                detail=f"Device {device} not found. Available devices: {available_devices}",
            )
        device_props = get_device_property(device)
        return JSONResponse(content=device_props)
    except Exception as e:
        logger.exception("Error getting properties for device.", error=e)
        raise HTTPException(status_code=500, detail=str(e))
