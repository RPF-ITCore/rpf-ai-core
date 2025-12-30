from fastapi import APIRouter, Query, HTTPException, status
import json
import os
import logging
from controllers.BaseController import BaseController

stats_router = APIRouter(prefix="/stats", tags=["Stats"])
base = BaseController()
logger = logging.getLogger(__name__)


def load_stats_data():
    """
    Load statistics data from stats/stats_data.json file
    Returns the parsed JSON data
    """
    try:
        # Get the base directory (project root)
        base_dir = os.path.dirname(os.path.dirname(__file__))
        stats_file_path = os.path.join(base_dir, "stats", "stats_data.json")
        
        if not os.path.exists(stats_file_path):
            raise FileNotFoundError(f"Stats data file not found at: {stats_file_path}")
        
        with open(stats_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error parsing stats data file: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error loading stats data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading stats data: {str(e)}"
        )


def load_available_cities():
    """
    Load available cities list from stats/available_cities.json file
    Returns the parsed JSON data
    """
    try:
        # Get the base directory (project root)
        base_dir = os.path.dirname(os.path.dirname(__file__))
        cities_file_path = os.path.join(base_dir, "stats", "available_cities.json")
        
        if not os.path.exists(cities_file_path):
            raise FileNotFoundError(f"Available cities file not found at: {cities_file_path}")
        
        with open(cities_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error parsing available cities file: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error loading available cities: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading available cities: {str(e)}"
        )


def load_cities_coordinates():
    """
    Load cities coordinates data from stats/cities_coordinates.json file
    Returns the parsed JSON data
    """
    try:
        # Get the base directory (project root)
        base_dir = os.path.dirname(os.path.dirname(__file__))
        coordinates_file_path = os.path.join(base_dir, "stats", "cities_coordinates.json")
        
        if not os.path.exists(coordinates_file_path):
            raise FileNotFoundError(f"Cities coordinates file not found at: {coordinates_file_path}")
        
        with open(coordinates_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error parsing cities coordinates file: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error loading cities coordinates: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading cities coordinates: {str(e)}"
        )


@stats_router.get("", summary="Get city statistics", status_code=status.HTTP_200_OK)
async def get_city_stats(
    city: str = Query(None, description="Filter by city name (case-insensitive)")
):
    """
    Get city statistics data from stats_data.json
    
    - **city** (optional): Filter by city name. If provided, returns only the matching city.
      If not provided, returns all cities.
      Case-insensitive matching is performed.
    
    Returns city statistics including scores for various aspects like Public Health,
    Economic Factor, Resource Management, Urban Planning, etc.
    """
    try:
        # Load the stats data
        stats_data = load_stats_data()
        cities = stats_data.get("cities", [])
        
        # If city parameter is provided, filter by city name (case-insensitive)
        if city:
            city_lower = city.lower().strip()
            matching_city = None
            
            for c in cities:
                if c.get("name", "").lower() == city_lower:
                    matching_city = c
                    break
            
            if not matching_city:
                return base.fail(
                    message=f"City '{city}' not found",
                    errors=[f"No statistics data found for city: {city}"],
                    status_code=status.HTTP_404_NOT_FOUND
                )
            
            return base.ok(
                data=matching_city,
                message=f"Statistics retrieved successfully for {matching_city['name']}"
            )
        
        # If no city parameter, return all cities
        return base.ok(
            data={"cities": cities},
            message=f"Statistics retrieved successfully for {len(cities)} cities"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_city_stats endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while retrieving statistics: {str(e)}"
        )


@stats_router.get("/cities", summary="Get available cities list", status_code=status.HTTP_200_OK)
async def get_available_cities():
    """
    Get the list of available cities for which statistics data is available.
    
    Returns a list of city names that can be used to filter statistics data.
    """
    try:
        # Load the available cities data
        cities_data = load_available_cities()
        cities = cities_data.get("cities", [])
        
        return base.ok(
            data={"cities": cities},
            message=f"Retrieved {len(cities)} available cities"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_available_cities endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while retrieving available cities: {str(e)}"
        )


@stats_router.get("/location", summary="Get city coordinates", status_code=status.HTTP_200_OK)
async def get_city_location(
    city: str = Query(..., description="City name (case-insensitive)")
):
    """
    Get the geographic coordinates (latitude and longitude) for a specific city.
    
    - **city** (required): City name to get coordinates for.
      Case-insensitive matching is performed.
    
    Returns the latitude and longitude coordinates for the specified city.
    """
    try:
        # Load the cities coordinates data
        coordinates_data = load_cities_coordinates()
        cities = coordinates_data.get("cities", [])
        
        if not city:
            return base.fail(
                message="City parameter is required",
                errors=["City name must be provided"],
                status_code=status.HTTP_400_BAD_REQUEST
            )
        
        # Search for the city (case-insensitive)
        city_lower = city.lower().strip()
        matching_city = None
        
        for c in cities:
            if c.get("name", "").lower() == city_lower:
                matching_city = c
                break
        
        if not matching_city:
            return base.fail(
                message=f"City '{city}' not found",
                errors=[f"No coordinates found for city: {city}"],
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        return base.ok(
            data={
                "city": matching_city["name"],
                "lat": matching_city["lat"],
                "lon": matching_city["lon"]
            },
            message=f"Coordinates retrieved successfully for {matching_city['name']}"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_city_location endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while retrieving city location: {str(e)}"
        )

