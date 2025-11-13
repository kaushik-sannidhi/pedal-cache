"""
FastAPI Backend for Eco-Friendly Navigation System

This is the main FastAPI application providing routing endpoints for pedestrian
and bike-friendly navigation. It serves 4 different routing algorithms optimized
for safety and bike infrastructure.

Endpoints:
- POST /route: Calculate a single route
- POST /route/compare: Compare multiple route types
- GET /health: Health check and system status
- GET /stats: Graph and system statistics
- GET /modes: Available routing modes
"""

import os
import sys
import time
import asyncio
import logging
import gc  # For memory management during initialization
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import uvicorn

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(backend_dir))

from crime_analyzer import CrimeDataAnalyzer
from osm_analyzer import OSMAnalyzer
from weight_calibrator import WeightCalibrator
from graph_builder import WeightedGraphBuilder
from route_calculator import RouteCalculator, RouteRequest, RouteResult
from cache_manager import CacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global application state
app_state = {
    'initialized': False,
    'crime_analyzer': None,
    'osm_analyzer': None,
    'graph_builder': None,
    'route_calculator': None,
    'calibrated_weights': None,
    'initialization_time': 0,
    'startup_errors': [],
    'cache_used': False,
    'cache_manager': None
}


# Pydantic models for API requests/responses
class RouteRequestModel(BaseModel):
    """Route calculation request model"""
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    route_type: str
    algorithm: str = 'dijkstra'
    
    @validator('start_lat', 'end_lat')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v
    
    @validator('start_lon', 'end_lon')  
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v
    
    @validator('route_type')
    def validate_route_type(cls, v):
        valid_types = ['fastest', 'safe', 'bike', 'safe_bike']
        if v not in valid_types:
            raise ValueError(f'Route type must be one of: {valid_types}')
        return v
    
    @validator('algorithm')
    def validate_algorithm(cls, v):
        valid_algorithms = ['dijkstra', 'astar']
        if v not in valid_algorithms:
            raise ValueError(f'Algorithm must be one of: {valid_algorithms}')
        return v


class RouteCompareRequestModel(BaseModel):
    """Route comparison request model"""
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    route_types: Optional[List[str]] = None
    algorithm: str = 'dijkstra'
    
    @validator('start_lat', 'end_lat')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v
    
    @validator('start_lon', 'end_lon')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v


class RouteResponseModel(BaseModel):
    """Route calculation response model"""
    route: List[List[float]]
    distance_meters: float
    estimated_time_minutes: float
    safety_score: float
    bike_coverage_percent: float
    route_type: str
    algorithm_used: str
    calculation_time_ms: float
    node_count: int
    error_message: Optional[str] = None


class HealthResponseModel(BaseModel):
    """Health check response model"""
    status: str
    initialized: bool
    system_info: Dict
    graph_stats: Optional[Dict] = None
    errors: List[str] = []


class RouteModesResponseModel(BaseModel):
    """Available routing modes response"""
    modes: Dict[str, Dict[str, str]]


async def initialize_system():
    """Initialize all system components during startup with caching support"""
    startup_start = time.time()
    
    try:
        logger.info("Starting system initialization...")
        logger.info("=" * 60)

        # Set up file paths - use DATA_DIR env var if available (Docker), else use parent dir
        data_dir_env = os.getenv('DATA_DIR')
        if data_dir_env:
            data_dir = Path(data_dir_env)
            logger.info(f"Using DATA_DIR from environment: {data_dir}")
        else:
            data_dir = backend_dir.parent  # Root directory where CSV/JSON files are located
            logger.info(f"Using default data directory: {data_dir}")

        indy_csv_path = data_dir / "IndianapolisCrime.csv"
        wl_json_path = data_dir / "WLCrime.json"
        
        # Verify data files exist
        if not indy_csv_path.exists():
            raise FileNotFoundError(f"Indianapolis crime data not found: {indy_csv_path}")
        
        if not wl_json_path.exists():
            raise FileNotFoundError(f"West Lafayette crime data not found: {wl_json_path}")
        
        logger.info(f"✓ Found crime data files")

        # Initialize cache manager
        cache_manager = CacheManager()
        app_state['cache_manager'] = cache_manager

        # Source files to track for cache invalidation
        source_files = {indy_csv_path, wl_json_path}

        # Try to load from cache
        if cache_manager.is_cache_valid(source_files):
            logger.info("✓ Valid cache found! Loading from cache...")
            cached_data = cache_manager.load_cache()

            if cached_data is not None:
                crime_analyzer, osm_analyzer, calibrated_weights, graph_builder, route_calculator = cached_data

                app_state['crime_analyzer'] = crime_analyzer
                app_state['osm_analyzer'] = osm_analyzer
                app_state['calibrated_weights'] = calibrated_weights
                app_state['graph_builder'] = graph_builder
                app_state['route_calculator'] = route_calculator
                app_state['cache_used'] = True

                logger.info("✓ Successfully loaded all data from cache!")
                logger.info(f"  - Crime analyzer: {len(crime_analyzer.combined_crimes_df) if crime_analyzer.combined_crimes_df is not None else 0} records")
                logger.info(f"  - OSM networks: {len(osm_analyzer.city_networks)} cities")
                logger.info(f"  - Weighted graphs: {len(graph_builder.weighted_graphs)} route types")
                logger.info(f"  - Route calculator: Ready with spatial indices")
            else:
                logger.warning("Cache load failed, will perform full initialization")
                app_state['cache_used'] = False
        else:
            logger.info("No valid cache found - performing full initialization...")
            logger.info("(This will take a few minutes, but will be cached for next time)")
            app_state['cache_used'] = False

        # If cache wasn't loaded, do full initialization
        if not app_state['cache_used']:
            logger.info("-" * 60)
            logger.info("FULL INITIALIZATION (First Run - Memory Optimized)")
            logger.info("-" * 60)

            # Initialize crime analyzer
            logger.info("→ Initializing crime data analyzer...")
            crime_analyzer = CrimeDataAnalyzer()
            crime_results = crime_analyzer.analyze_crime_data(str(indy_csv_path), str(wl_json_path))
            app_state['crime_analyzer'] = crime_analyzer
            logger.info(f"  ✓ Crime analysis complete: {crime_results.get('total_crime_count', 0)} records processed")

            # Memory cleanup after crime analysis
            gc.collect()
            logger.info(f"  ✓ Memory cleanup: {gc.collect()} objects collected")

            # Initialize OSM analyzer
            logger.info("→ Initializing OSM network analyzer...")
            osm_analyzer = OSMAnalyzer()
            osm_results = osm_analyzer.analyze_osm_networks()
            app_state['osm_analyzer'] = osm_analyzer
            logger.info(f"  ✓ OSM analysis complete: {osm_results.get('total_cities', 0)} cities loaded")

            # Memory cleanup after OSM analysis
            gc.collect()
            logger.info(f"  ✓ Memory cleanup: {gc.collect()} objects collected")

            # Initialize weight calibrator and run calibration
            # Pass the already-initialized analyzers to avoid re-downloading networks
            logger.info("→ Running weight calibration...")
            calibrator = WeightCalibrator()
            calibrated_weights = calibrator.master_calibration(
                str(indy_csv_path),
                str(wl_json_path),
                crime_analyzer=crime_analyzer,  # Reuse already-initialized analyzer
                osm_analyzer=osm_analyzer       # Reuse already-initialized analyzer
            )
            app_state['calibrated_weights'] = calibrated_weights
            logger.info("  ✓ Weight calibration complete")

            # Memory cleanup after calibration
            gc.collect()
            logger.info(f"  ✓ Memory cleanup: {gc.collect()} objects collected")

            # Initialize graph builder
            logger.info("→ Building weighted graphs...")
            graph_builder = WeightedGraphBuilder()

            # Load base graph from OSM analyzer
            combined_graph = osm_analyzer.export_network_for_routing()
            if combined_graph is None or combined_graph.number_of_nodes() == 0:
                raise ValueError("No valid road network available from OSM analysis")

            graph_builder.load_base_graph(combined_graph)
            graph_builder.load_calibrated_weights(calibrated_weights)
            graph_builder.set_crime_analyzer(crime_analyzer)

            # Create all weighted graphs
            weighted_graphs = graph_builder.create_weighted_graphs()
            app_state['graph_builder'] = graph_builder
            logger.info(f"  ✓ Created {len(weighted_graphs)} weighted graphs")

            # Memory cleanup after graph building
            gc.collect()
            logger.info(f"  ✓ Memory cleanup: {gc.collect()} objects collected")

            # Validate graphs
            validation_results = graph_builder.validate_graphs()
            invalid_graphs = [name for name, result in validation_results.items() if not result['valid']]
            if invalid_graphs:
                logger.warning(f"  ⚠ Some graphs failed validation: {invalid_graphs}")

            # Initialize route calculator
            logger.info("→ Initializing route calculator...")
            route_calculator = RouteCalculator()
            if not route_calculator.initialize(graph_builder, crime_analyzer):
                raise RuntimeError("Failed to initialize route calculator")
            app_state['route_calculator'] = route_calculator
            logger.info("  ✓ Route calculator initialized")

            # Final memory cleanup before caching
            gc.collect()
            logger.info(f"  ✓ Final memory cleanup: {gc.collect()} objects collected")

            # Save to cache for next time
            logger.info("→ Saving processed data to cache...")
            cache_saved = cache_manager.save_cache(
                crime_analyzer,
                osm_analyzer,
                calibrated_weights,
                graph_builder,
                route_calculator,  # Now caching route calculator too!
                source_files
            )

            if cache_saved:
                cache_info = cache_manager.get_cache_info()
                logger.info(f"  ✓ Cache saved successfully ({cache_info['total_size_mb']:.1f} MB)")
                logger.info(f"  ✓ Next startup will be much faster!")
            else:
                logger.warning("  ⚠ Cache save failed - next startup will be slow")

        # Route calculator is either loaded from cache or initialized above
        # Only initialize if not already set (i.e., cache wasn't used)
        if not app_state.get('route_calculator'):
            logger.info("→ Initializing route calculator...")
            route_calculator = RouteCalculator()
            if not route_calculator.initialize(app_state['graph_builder'], app_state['crime_analyzer']):
                raise RuntimeError("Failed to initialize route calculator")
            app_state['route_calculator'] = route_calculator
            logger.info("  ✓ Route calculator ready")

        # Mark initialization as complete
        app_state['initialized'] = True
        app_state['initialization_time'] = time.time() - startup_start
        
        logger.info("=" * 60)
        logger.info(f"✓ SYSTEM READY in {app_state['initialization_time']:.2f} seconds")
        if app_state['cache_used']:
            logger.info("  (Loaded from cache)")
        else:
            logger.info("  (Full initialization - cached for next time)")
        logger.info("=" * 60)

    except Exception as e:
        error_msg = f"System initialization failed: {str(e)}"
        logger.error(error_msg)
        app_state['startup_errors'].append(error_msg)
        app_state['initialized'] = False
        
        # Don't raise exception - let the server start but mark as not initialized
        # This allows health checks to report the error


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown"""
    # Startup
    await initialize_system()
    yield
    # Shutdown
    logger.info("System shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Eco-Friendly Navigation API",
    description="FastAPI backend for pedestrian and bike-friendly routing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints

@app.get("/health", response_model=HealthResponseModel)
async def health_check():
    """Health check endpoint with system status"""
    
    system_info = {
        'initialization_time_seconds': app_state.get('initialization_time', 0),
        'cache_used': app_state.get('cache_used', False),
        'python_version': sys.version,
        'backend_directory': str(backend_dir),
    }
    
    # Add cache information if available
    if app_state.get('cache_manager'):
        try:
            cache_info = app_state['cache_manager'].get_cache_info()
            system_info['cache_info'] = {
                'cache_exists': cache_info['cache_exists'],
                'cache_size_mb': cache_info['total_size_mb'],
                'cache_directory': cache_info['cache_directory']
            }
        except Exception as e:
            system_info['cache_error'] = str(e)

    # Add graph statistics if available
    graph_stats = None
    if app_state['initialized'] and app_state['route_calculator']:
        try:
            graph_stats = app_state['route_calculator'].get_route_statistics()
        except Exception as e:
            system_info['stats_error'] = str(e)
    
    status = "healthy" if app_state['initialized'] and not app_state['startup_errors'] else "unhealthy"
    
    return HealthResponseModel(
        status=status,
        initialized=app_state['initialized'],
        system_info=system_info,
        graph_stats=graph_stats,
        errors=app_state['startup_errors']
    )


@app.get("/modes", response_model=RouteModesResponseModel)
async def get_available_modes():
    """Get available routing modes with descriptions"""
    
    modes = {
        "fastest": {
            "name": "Fastest Route",
            "description": "Shortest distance route without penalties"
        },
        "safe": {
            "name": "Safe Route", 
            "description": "Avoids high-crime areas using crime data analysis"
        },
        "bike": {
            "name": "Bike Route",
            "description": "Prioritizes bike lanes and cycling infrastructure"
        },
        "safe_bike": {
            "name": "Safe + Bike Route",
            "description": "Balances safety and bike infrastructure preferences"
        }
    }
    
    return RouteModesResponseModel(modes=modes)


@app.get("/stats")
async def get_system_stats():
    """Get detailed system and graph statistics"""
    
    if not app_state['initialized']:
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Check /health for details."
        )
    
    try:
        stats = app_state['route_calculator'].get_route_statistics()
        
        # Add additional system information
        stats.update({
            'initialization_time_seconds': app_state['initialization_time'],
            'initialized': app_state['initialized'],
            'startup_errors': app_state['startup_errors']
        })
        
        # Add crime analysis stats if available
        if app_state['calibrated_weights']:
            crime_stats = app_state['calibrated_weights'].calibration_stats.get('crime_analysis', {})
            stats['crime_analysis'] = {
                'total_crimes': crime_stats.get('total_crime_count', 0),
                'indy_crimes': crime_stats.get('indy_crime_count', 0),
                'wl_crimes': crime_stats.get('wl_crime_count', 0),
                'crime_types_mapped': len(crime_stats.get('crime_type_weights', {}))
            }
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")


@app.post("/route", response_model=RouteResponseModel)
async def calculate_route(request: RouteRequestModel):
    """Calculate a single route between two points"""
    
    if not app_state['initialized']:
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Check /health for details."
        )
    
    try:
        # Convert to internal request format
        internal_request = RouteRequest(
            start_lat=request.start_lat,
            start_lon=request.start_lon,
            end_lat=request.end_lat,
            end_lon=request.end_lon,
            route_type=request.route_type,
            algorithm=request.algorithm
        )
        
        # Calculate route
        result = app_state['route_calculator'].calculate_route(internal_request)
        
        # Convert to response format
        response = RouteResponseModel(
            route=result.route,
            distance_meters=result.distance_meters,
            estimated_time_minutes=result.estimated_time_minutes,
            safety_score=result.safety_score,
            bike_coverage_percent=result.bike_coverage_percent,
            route_type=result.route_type,
            algorithm_used=result.algorithm_used,
            calculation_time_ms=result.calculation_time_ms,
            node_count=result.node_count,
            error_message=result.error_message
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route calculation failed: {str(e)}")


@app.get("/cache/info")
async def get_cache_info():
    """Get information about the cache status"""
    if app_state.get('cache_manager'):
        try:
            cache_info = app_state['cache_manager'].get_cache_info()
            cache_info['cache_used_at_startup'] = app_state.get('cache_used', False)
            return cache_info
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving cache info: {str(e)}")
    else:
        raise HTTPException(status_code=503, detail="Cache manager not initialized")


@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cached data (requires server restart to rebuild cache)"""
    if app_state.get('cache_manager'):
        try:
            app_state['cache_manager'].clear_cache()
            return {
                "status": "success",
                "message": "Cache cleared successfully. Restart the server to rebuild the cache."
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")
    else:
        raise HTTPException(status_code=503, detail="Cache manager not initialized")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )


# Development server startup
if __name__ == "__main__":
    # Configure for development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disabled due to lifespan initialization
        log_level="info"
    )
