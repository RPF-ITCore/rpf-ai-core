from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse
import logging

from services.auth_service import AuthService
from dtos.auth import (
    RegisterRequest, LoginRequest, 
    TokenResponse, UserResponse, RegisterResponse, UserListResponse
)
from dependencies.auth import get_auth_service, require_admin, get_current_user
from controllers.BaseController import BaseController

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])
base = BaseController()
logger = logging.getLogger(__name__)


@auth_router.post("/register", summary="Register a new user", status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Register a new user account
    
    - **email**: User's email address (must be unique)
    - **password**: User's password (minimum 8 characters, will be hashed)
    - **password_confirm**: Password confirmation (must match password)
    - **full_name**: User's full name (optional)
    
    Password requirements:
    - Must be at least 8 characters long
    - Must be maximum 72 characters long (bcrypt limitation)
    - password and password_confirm must match
    
    Returns the created user information.
    New users are automatically assigned the 'user' role.
    """
    try:
        # Debug: Log incoming request details
        logger.info(f"Received registration request for: {request.email}")
        logger.info(f"Password validation passed: length={len(request.password)}")
        
        # Register the user
        user = await auth_service.register_user(request)
        
        return base.ok(data={"user": user.model_dump(mode='json')}, message="User registered successfully", status_code=status.HTTP_201_CREATED)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in register endpoint: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during registration: {str(e)}"
        )


@auth_router.post("/login", summary="Login and get access token", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Login with email and password to get a JWT access token
    
    Request body (JSON):
    - **email**: Your email address
    - **password**: Your password
    
    Returns a JWT access token that should be included in the Authorization header
    for protected endpoints: `Authorization: Bearer <token>`
    
    Example request body:
    {
        "email": "user@example.com",
        "password": "yourpassword"
    }
    """
    try:
        # Authenticate user with the request directly
        token = await auth_service.authenticate_user(request)
        
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return base.ok(data={"access_token": token, "token_type": "bearer"}, message="Login successful")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in login endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during login: {str(e)}"
        )


@auth_router.get("/me", summary="Get current user information", response_model=UserResponse)
async def get_current_user_info(
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get information about the currently authenticated user
    
    Returns the user's information including email, name, and role.
    Requires authentication.
    """
    return base.ok(data=current_user.model_dump(mode='json'), message="Current user")


@auth_router.get("/users", summary="List all users (Admin only)", response_model=UserListResponse)
async def list_users(
    page: int = 1,
    page_size: int = 20,
    current_user: UserResponse = Depends(require_admin),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    List all users in the system (Admin only)
    
    - **page**: Page number for pagination (default: 1)
    - **page_size**: Number of users per page (default: 20)
    
    Returns a paginated list of all users.
    Requires admin role.
    """
    try:
        users, total = await auth_service.list_users(page=page, page_size=page_size)
        
        return base.ok(data={
            "users": [u.model_dump(mode='json') for u in users],
            "total": total
        }, message="Users listed")
        
    except Exception as e:
        logger.error(f"Error in list_users endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while listing users: {str(e)}"
        )
