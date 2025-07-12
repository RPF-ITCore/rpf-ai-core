from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from core.constants import FILE_MAX_SIZE
data_router = APIRouter(prefix="/data", tags=["Data"])


@data_router.post("/upload", summary="Upload a file", status_code=status.HTTP_200_OK)
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file uploaded. Please upload a valid file."
            )
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )
        if len(contents) > FILE_MAX_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Maximum allowed size is {FILE_MAX_SIZE // (1024*1024)}MB."
            )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "File uploaded successfully!",
                "filename": file.filename,
                "size_in_bytes": len(contents)
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while uploading the file: {str(e)}"
        )
