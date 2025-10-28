from fastapi import Depends, HTTPException, status, Request
from jose import JWTError, jwt
from typing import Optional
import logging

from schemas.auth import User, UserRole
from services.auth_service import AuthService
from dtos.auth import UserResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


def get_auth_service(request: Request) -> AuthService:
    """Dependency to get auth service with database connection"""
    if not hasattr(request.app, 'db_client'):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not initialized"
        )
    return AuthService(request.app.db_client)


async def get_current_user(
    token: str = None,
    request: Request = None,
    auth_service: AuthService = Depends(get_auth_service)
) -> UserResponse:
    """
    Dependency to get the current authenticated user from JWT token
    
    Extracts JWT token from Authorization header, verifies it,
    and returns the user information.
    
    Example usage in routes:
        current_user: UserResponse = Depends(get_current_user)
    """
    # Extract token from Authorization header
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = auth_header.split(" ")[1]
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # Verify and decode token
        from core.config import get_settings
        settings = get_settings()
        
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from database
        user = await auth_service.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Inactive user account"
            )
        
        return user
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_current_user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error verifying authentication"
        )


def require_user(
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """
    Dependency to ensure user is authenticated
    
    This is a simple wrapper around get_current_user for semantic clarity.
    
    Example usage:
        current_user: UserResponse = Depends(require_user)
    """
    return current_user


def require_admin(
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """
    Dependency to ensure user has admin role
    
    Checks if the current user's role is 'admin'.
    Raises 403 Forbidden if user doesn't have admin role.
    
    Example usage:
        admin_user: UserResponse = Depends(require_admin)
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions. Admin role required."
        )
    return current_user
