from datetime import datetime, timedelta
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
import logging

from schemas.auth import User, UserRole
from dtos.auth import RegisterRequest, LoginRequest, UserResponse
from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Password hashing context
try:
    # Test bcrypt functionality during initialization
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    # Attempt a test hash to ensure bcrypt is working
    try:
        test_hash = pwd_context.hash("test")
        logger.info("bcrypt initialized successfully")
    except Exception as e:
        logger.error(f"bcrypt test failed during initialization: {str(e)}")
        raise
except Exception as e:
    logger.error(f"Failed to initialize bcrypt: {str(e)}")
    logger.error("Please ensure bcrypt is properly installed: pip install bcrypt==4.0.1 passlib==1.7.4")
    raise


class AuthService:
    """Service for handling authentication and user management"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.users_collection = db.users
        
    async def create_indexes(self):
        """Create necessary database indexes for performance"""
        try:
            await self.users_collection.create_index("email", unique=True)
            await self.users_collection.create_index("role")
            await self.users_collection.create_index("is_active")
            logger.info("Auth service indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating auth service indexes: {str(e)}")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a plain password against a hashed password"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """
        Hash a password using bcrypt
        
        Note: bcrypt has a 72-byte limit. Passwords longer than 72 characters
        are now validated at the DTO level to prevent this error.
        """
        # Log password details
        password_bytes = password.encode('utf-8')
        password_byte_length = len(password_bytes)
        
        logger.info(f"Hashing password: length={len(password)} chars, bytes={password_byte_length}")
        
        # Ensure password is not longer than 72 bytes
        if password_byte_length > 72:
            logger.warning(f"Password exceeds 72 bytes (has {password_byte_length} bytes), truncating to first 72 bytes")
            # Truncate to first 72 bytes
            password_bytes = password_bytes[:72]
            password = password_bytes.decode('utf-8', errors='ignore')
            logger.info(f"Truncated password to {len(password)} chars, {len(password.encode('utf-8'))} bytes")
        
        # Final check before hashing
        final_bytes = password.encode('utf-8')
        if len(final_bytes) > 72:
            logger.error(f"Password still exceeds 72 bytes after truncation: {len(final_bytes)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Password too long: {password_byte_length} bytes exceeds 72 byte limit"
            )
        
        logger.info(f"Attempting bcrypt hash with {len(final_bytes)} bytes")
        try:
            result = pwd_context.hash(password)
            logger.info("Password successfully hashed with bcrypt")
            return result
        except Exception as e:
            logger.error(f"Error hashing password with bcrypt: {str(e)}")
            logger.error(f"Password details - chars: {len(password)}, bytes: {len(final_bytes)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error hashing password: {str(e)}"
            )
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(
            to_encode, 
            settings.JWT_SECRET_KEY, 
            algorithm=settings.JWT_ALGORITHM
        )
        return encoded_jwt
    
    def verify_token(self, token: str) -> dict:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(
                token, 
                settings.JWT_SECRET_KEY, 
                algorithms=[settings.JWT_ALGORITHM]
            )
            return payload
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    async def register_user(self, request: RegisterRequest) -> UserResponse:
        """Register a new user"""
        try:
            # Debug logging
            logger.info(f"Registering user: {request.email}")
            logger.info(f"Password length: {len(request.password)} characters")
            logger.info(f"Password bytes length: {len(request.password.encode('utf-8'))} bytes")
            
            # Check if user already exists
            existing_user = await self.users_collection.find_one({"email": request.email})
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            
            # Hash the password
            logger.info("Attempting to hash password...")
            hashed_password = self.get_password_hash(request.password)
            logger.info("Password hashed successfully")
            
            # Create user document
            user = User(
                email=request.email,
                hashed_password=hashed_password,
                full_name=request.full_name,
                role=UserRole.USER,  # New registrations are always 'user' role
                is_active=True
            )
            
            # Insert into database
            result = await self.users_collection.insert_one(user.model_dump(by_alias=True, exclude={"id"}))
            
            # Fetch the created user
            created_user = await self.users_collection.find_one({"_id": result.inserted_id})
            
            # Convert to response format
            user_data = {
                "id": str(created_user["_id"]),
                **{k: v for k, v in created_user.items() if k != "_id"}
            }
            return UserResponse(**user_data)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error registering user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating user: {str(e)}"
            )
    
    async def authenticate_user(self, request: LoginRequest) -> Optional[str]:
        """Authenticate a user and return JWT token"""
        try:
            # Find user by email
            user_doc = await self.users_collection.find_one({"email": request.email})
            
            if not user_doc:
                return None
            
            user = User(**user_doc)
            
            # Check if user is active
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User account is deactivated"
                )
            
            # Verify password
            if not self.verify_password(request.password, user.hashed_password):
                return None
            
            # Create access token
            access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = self.create_access_token(
                data={"sub": str(user_doc["_id"]), "role": user.role.value},
                expires_delta=access_token_expires
            )
            
            return access_token
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error during authentication: {str(e)}"
            )
    
    async def get_user_by_id(self, user_id: str) -> Optional[UserResponse]:
        """Get a user by ID"""
        try:
            user_doc = await self.users_collection.find_one({"_id": ObjectId(user_id)})
            if user_doc:
                user_data = {
                    "id": str(user_doc["_id"]),
                    **{k: v for k, v in user_doc.items() if k != "_id"}
                }
                return UserResponse(**user_data)
            return None
        except Exception as e:
            logger.error(f"Error getting user by ID {user_id}: {str(e)}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[UserResponse]:
        """Get a user by email"""
        try:
            user_doc = await self.users_collection.find_one({"email": email})
            if user_doc:
                user_data = {
                    "id": str(user_doc["_id"]),
                    **{k: v for k, v in user_doc.items() if k != "_id"}
                }
                return UserResponse(**user_data)
            return None
        except Exception as e:
            logger.error(f"Error getting user by email {email}: {str(e)}")
            return None
    
    async def list_users(self, page: int = 1, page_size: int = 20) -> tuple[list[UserResponse], int]:
        """List all users with pagination"""
        try:
            # Count total documents
            total = await self.users_collection.count_documents({})
            
            # Get paginated results
            skip = (page - 1) * page_size
            cursor = self.users_collection.find({}).skip(skip).limit(page_size)
            
            users = []
            async for user_doc in cursor:
                user_data = {
                    "id": str(user_doc["_id"]),
                    **{k: v for k, v in user_doc.items() if k != "_id"}
                }
                users.append(UserResponse(**user_data))
            
            return users, total
            
        except Exception as e:
            logger.error(f"Error listing users: {str(e)}")
            return [], 0
