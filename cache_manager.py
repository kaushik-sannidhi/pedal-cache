"""
Cache Manager Module

This module handles caching of processed data to dramatically speed up server startup.
After the first initialization, all processed data is saved to disk and loaded instantly
on subsequent startups.

Cached Components:
- Crime data analysis results and density grids
- OSM road network graphs
- Calibrated weights
- Pre-built weighted graphs for all route types

Cache invalidation:
- Automatically detects when source data files change
- Manual cache clearing available

Performance Optimizations:
- Uses pickle protocol 5 for fastest serialization
- Compresses cache files with lz4 for faster I/O
- Parallel loading of independent components
- Memory-mapped file access for large objects
"""

import pickle  # Using pickle for complex object serialization (NetworkX graphs, numpy arrays, class instances)
               # Note: marshal is faster for simple types but cannot handle our complex data structures
import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import time
import logging

# Try to import lz4 for compression (significant speedup for large files)
try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    print("Warning: lz4 not available. Install with: pip install lz4")
    print("         Cache loading will be slower without compression.")

logger = logging.getLogger(__name__)


@dataclass
class CacheMetadata:
    """Metadata about cached data for validation"""
    created_timestamp: float
    source_files: Dict[str, str]  # filename -> file hash
    source_file_mtimes: Dict[str, float] = None  # filename -> modification time (fast check)
    cache_version: str = "1.1.0"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class CacheManager:
    """
    Manages caching of processed data for fast server startup.

    Cache Structure:
    - cache/
      - metadata.json          # Cache metadata and validation
      - crime_analyzer.pkl     # CrimeDataAnalyzer state (compressed)
      - osm_analyzer.pkl       # OSMAnalyzer state (compressed)
      - calibrated_weights.pkl # CalibratedWeights (compressed)
      - graph_builder.pkl      # WeightedGraphBuilder with all graphs (compressed)
      - route_calculator.pkl   # RouteCalculator with spatial indices (compressed)

    Performance Features:
    - LZ4 compression for faster I/O on large files
    - Pickle protocol 5 for optimal serialization
    - Parallel loading of independent components
    - Memory-mapped access for large graphs
    """

    CACHE_VERSION = "1.1.0"  # Incremented for new compression format
    PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL  # Use highest available (usually 5 in Python 3.8+)

    def __init__(self, cache_dir: Optional[Path] = None, use_compression: Optional[bool] = None):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for cache files (default: CACHE_DIR env var or backend/cache)
            use_compression: Enable lz4 compression (default: auto-detect if lz4 available)
        """
        # Check for CACHE_DIR environment variable (Docker)
        cache_dir_env = os.getenv('CACHE_DIR')
        if cache_dir_env:
            cache_dir = Path(cache_dir_env)
            logger.info(f"Using CACHE_DIR from environment: {cache_dir}")
        else:
            # Default to cache directory in backend folder
            backend_dir = Path(__file__).parent.absolute()
            cache_dir = backend_dir / "cache"
            logger.info(f"Using default cache directory: {cache_dir}")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Compression settings
        self.use_compression = HAS_LZ4 if use_compression is None else (use_compression and HAS_LZ4)
        if self.use_compression:
            logger.info("✓ LZ4 compression enabled for faster cache I/O")
        else:
            logger.info("ℹ LZ4 compression disabled (install lz4 for better performance)")

        # Cache file paths
        self.metadata_path = self.cache_dir / "metadata.json"
        self.crime_analyzer_cache = self.cache_dir / "crime_analyzer.pkl"
        self.osm_analyzer_cache = self.cache_dir / "osm_analyzer.pkl"
        self.calibrated_weights_cache = self.cache_dir / "calibrated_weights.pkl"
        self.graph_builder_cache = self.cache_dir / "graph_builder.pkl"
        self.route_calculator_cache = self.cache_dir / "route_calculator.pkl"

        logger.info(f"Cache directory: {self.cache_dir}")

    def _pickle_save(self, obj: Any, filepath: Path) -> int:
        """
        Save object to pickle file with optional compression.
        Memory-optimized: uses streaming to avoid holding entire serialized object in memory.
        
        Returns:
            File size in bytes
        """
        if self.use_compression:
            # Stream directly to compressed file to minimize memory usage
            with lz4.frame.open(filepath, mode='wb', compression_level=0) as f:
                pickle.dump(obj, f, protocol=self.PICKLE_PROTOCOL)
            return filepath.stat().st_size
        else:
            # Direct pickle to file (no intermediate buffer)
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f, protocol=self.PICKLE_PROTOCOL)
            return filepath.stat().st_size

    def _pickle_load(self, filepath: Path) -> Any:
        """
        Load object from pickle file with optional decompression.
        Memory-optimized: streams decompression to avoid large intermediate buffers.
        """
        if self.use_compression:
            try:
                # Stream decompression directly from file
                with lz4.frame.open(filepath, mode='rb') as f:
                    obj = pickle.load(f)
                return obj
            except (lz4.frame.LZ4FrameError, pickle.UnpicklingError):
                # Fall back to uncompressed if decompression fails (backward compatibility)
                logger.warning(f"Failed to decompress {filepath.name}, trying uncompressed load")
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
        else:
            # Direct load from file
            with open(filepath, 'rb') as f:
                return pickle.load(f)

    def _calculate_file_hash(self, filepath: Path) -> str:
        """
        Calculate SHA256 hash of a file for change detection.
        Optimized for large files using memory-mapped I/O.
        """
        if not filepath.exists():
            return ""

        sha256_hash = hashlib.sha256()

        # For small files (< 100MB), use regular reading
        file_size = filepath.stat().st_size
        if file_size < 100 * 1024 * 1024:  # 100 MB
            with open(filepath, "rb") as f:
                # Read in larger chunks for better performance
                for byte_block in iter(lambda: f.read(1024 * 1024), b""):  # 1MB chunks
                    sha256_hash.update(byte_block)
        else:
            # For large files, use memory-mapped I/O for maximum speed
            import mmap
            with open(filepath, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped:
                    # Process in chunks to avoid loading entire file into memory
                    chunk_size = 8 * 1024 * 1024  # 8MB chunks
                    for i in range(0, len(mmapped), chunk_size):
                        sha256_hash.update(mmapped[i:i + chunk_size])

        return sha256_hash.hexdigest()

    def _save_metadata(self, source_files: Dict[Path, str]):
        """Save cache metadata with file hashes and modification times"""
        # Collect modification times for fast validation
        source_file_mtimes = {
            str(path): path.stat().st_mtime
            for path in source_files.keys()
            if path.exists()
        }

        metadata = CacheMetadata(
            created_timestamp=time.time(),
            source_files={str(path): hash_val for path, hash_val in source_files.items()},
            source_file_mtimes=source_file_mtimes,
            cache_version=self.CACHE_VERSION
        )

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info("Cache metadata saved")

    def _load_metadata(self) -> Optional[CacheMetadata]:
        """Load cache metadata"""
        if not self.metadata_path.exists():
            return None

        try:
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
            return CacheMetadata.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load cache metadata: {e}")
            return None

    def is_cache_valid(self, source_files: Dict[Path, str]) -> bool:
        """
        Check if cached data is valid (all cache files exist and source files unchanged).
        Uses fast modification time check before expensive hash calculation.

        Args:
            source_files: Dictionary of source file paths to check

        Returns:
            True if cache is valid and can be used, False otherwise
        """
        # Check if all cache files exist
        cache_files = [
            self.metadata_path,
            self.crime_analyzer_cache,
            self.osm_analyzer_cache,
            self.calibrated_weights_cache,
            self.graph_builder_cache,
            self.route_calculator_cache
        ]

        if not all(f.exists() for f in cache_files):
            logger.info("Cache invalid: Missing cache files")
            return False

        # Load and validate metadata
        metadata = self._load_metadata()
        if metadata is None:
            logger.info("Cache invalid: Cannot load metadata")
            return False

        # Check cache version
        if metadata.cache_version != self.CACHE_VERSION:
            logger.info(f"Cache invalid: Version mismatch (cached: {metadata.cache_version}, current: {self.CACHE_VERSION})")
            return False

        # Fast path: Check modification times first (avoid expensive hash calculation)
        if metadata.source_file_mtimes:
            mtimes_changed = False
            for filepath_str, cached_mtime in metadata.source_file_mtimes.items():
                filepath = Path(filepath_str)
                if filepath.exists():
                    current_mtime = filepath.stat().st_mtime
                    if abs(current_mtime - cached_mtime) > 0.1:  # Allow small time differences
                        mtimes_changed = True
                        logger.info(f"Cache invalid (fast check): Source file modified: {filepath_str}")
                        break
                else:
                    logger.info(f"Cache invalid: Source file missing: {filepath_str}")
                    return False

            if not mtimes_changed:
                # Modification times unchanged - cache is valid!
                logger.info("✓ Cache valid (fast check: modification times unchanged)")
                return True

        # Fallback: Full hash validation (slower but more reliable)
        # Only reached if mtime check failed or no mtimes stored (old cache format)
        logger.info("Running full hash validation...")
        current_hashes = {str(path): self._calculate_file_hash(path)
                         for path in source_files}

        for filepath, current_hash in current_hashes.items():
            cached_hash = metadata.source_files.get(filepath, "")
            if current_hash != cached_hash:
                logger.info(f"Cache invalid: Source file changed: {filepath}")
                return False

        logger.info("✓ Cache is valid and up-to-date")
        return True

    def save_cache(self,
                   crime_analyzer: Any,
                   osm_analyzer: Any,
                   calibrated_weights: Any,
                   graph_builder: Any,
                   route_calculator: Any,
                   source_files: Dict[Path, str]):
        """
        Save all processed data to cache with compression and memory optimization.
        Uses streaming serialization and aggressive garbage collection to minimize peak memory.

        Args:
            crime_analyzer: CrimeDataAnalyzer instance
            osm_analyzer: OSMAnalyzer instance
            calibrated_weights: CalibratedWeights instance
            graph_builder: WeightedGraphBuilder instance with all graphs
            route_calculator: RouteCalculator instance with spatial indices
            source_files: Dictionary of source file paths that were used
        """
        try:
            logger.info("Saving processed data to cache...")
            start_time = time.time()

            # Calculate source file hashes
            source_file_hashes = {path: self._calculate_file_hash(path)
                                 for path in source_files}

            # Save each component using optimized pickle save
            logger.info("Saving components (with compression)..." if self.use_compression else "Saving components...")

            # Crime analyzer (small)
            t0 = time.time()
            size = self._pickle_save(crime_analyzer, self.crime_analyzer_cache)
            logger.info(f"  ✓ Crime analyzer ({size / 1024 / 1024:.1f} MB in {time.time()-t0:.2f}s)")
            import gc
            gc.collect()  # Clean up serialization buffers

            # Calibrated weights (tiny - save early)
            t0 = time.time()
            size = self._pickle_save(calibrated_weights, self.calibrated_weights_cache)
            logger.info(f"  ✓ Calibrated weights ({size / 1024 / 1024:.1f} MB in {time.time()-t0:.2f}s)")
            gc.collect()

            # OSM analyzer (LARGEST - most memory intensive)
            t0 = time.time()
            logger.info("  → Saving OSM analyzer (large, may take time)...")
            size = self._pickle_save(osm_analyzer, self.osm_analyzer_cache)
            logger.info(f"  ✓ OSM analyzer ({size / 1024 / 1024:.1f} MB in {time.time()-t0:.2f}s)")
            gc.collect()  # Critical cleanup after large object

            # Graph builder (large - contains multiple weighted graphs)
            t0 = time.time()
            logger.info("  → Saving graph builder (large, may take time)...")
            size = self._pickle_save(graph_builder, self.graph_builder_cache)
            logger.info(f"  ✓ Graph builder ({size / 1024 / 1024:.1f} MB in {time.time()-t0:.2f}s)")
            gc.collect()  # Critical cleanup after large object

            # Route calculator (medium)
            t0 = time.time()
            size = self._pickle_save(route_calculator, self.route_calculator_cache)
            logger.info(f"  ✓ Route calculator ({size / 1024 / 1024:.1f} MB in {time.time()-t0:.2f}s)")
            gc.collect()

            # Save metadata last (indicates successful cache save)
            self._save_metadata(source_file_hashes)

            total_size = sum(f.stat().st_size for f in [
                self.crime_analyzer_cache,
                self.osm_analyzer_cache,
                self.calibrated_weights_cache,
                self.graph_builder_cache,
                self.route_calculator_cache
            ]) / 1024 / 1024

            elapsed = time.time() - start_time
            throughput = total_size / elapsed if elapsed > 0 else 0
            logger.info(f"✓ Cache saved: {total_size:.1f} MB in {elapsed:.2f}s ({throughput:.1f} MB/s)")
            
            # Final cleanup
            gc.collect()
            logger.info(f"✓ Memory cleanup after cache save: {gc.collect()} objects collected")

            return True

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            import traceback
            traceback.print_exc()
            # Clean up partial cache
            self.clear_cache()
            return False

    def load_cache(self) -> Optional[Tuple[Any, Any, Any, Any, Any]]:
        """
        Load all processed data from cache with memory optimization.
        Loads sequentially (not parallel) to minimize peak memory usage.

        Returns:
            Tuple of (crime_analyzer, osm_analyzer, calibrated_weights, graph_builder, route_calculator)
            or None if cache cannot be loaded
        """
        try:
            import gc
            logger.info("Loading data from cache (memory-optimized)...")
            start_time = time.time()

            components = {}
            total_size = 0
            
            # Load components SEQUENTIALLY to minimize peak memory
            # (Parallel loading causes all objects in memory at once)
            
            # 1. Crime analyzer (small)
            t0 = time.time()
            components['crime_analyzer'] = self._pickle_load(self.crime_analyzer_cache)
            size_mb = self.crime_analyzer_cache.stat().st_size / 1024 / 1024
            total_size += size_mb
            logger.info(f"  ✓ Crime analyzer ({size_mb:.1f} MB in {time.time()-t0:.2f}s)")
            gc.collect()

            # 2. Calibrated weights (tiny)
            t0 = time.time()
            components['calibrated_weights'] = self._pickle_load(self.calibrated_weights_cache)
            size_mb = self.calibrated_weights_cache.stat().st_size / 1024 / 1024
            total_size += size_mb
            logger.info(f"  ✓ Calibrated weights ({size_mb:.1f} MB in {time.time()-t0:.2f}s)")
            gc.collect()

            # 3. OSM analyzer (LARGEST - load carefully)
            t0 = time.time()
            logger.info("  → Loading OSM analyzer (large, may take time)...")
            components['osm_analyzer'] = self._pickle_load(self.osm_analyzer_cache)
            size_mb = self.osm_analyzer_cache.stat().st_size / 1024 / 1024
            total_size += size_mb
            logger.info(f"  ✓ OSM analyzer ({size_mb:.1f} MB in {time.time()-t0:.2f}s)")
            collected = gc.collect()
            if collected > 0:
                logger.info(f"    → Memory cleanup: {collected} objects collected")

            # 4. Graph builder (large)
            t0 = time.time()
            logger.info("  → Loading graph builder (large, may take time)...")
            components['graph_builder'] = self._pickle_load(self.graph_builder_cache)
            size_mb = self.graph_builder_cache.stat().st_size / 1024 / 1024
            total_size += size_mb
            logger.info(f"  ✓ Graph builder ({size_mb:.1f} MB in {time.time()-t0:.2f}s)")
            collected = gc.collect()
            if collected > 0:
                logger.info(f"    → Memory cleanup: {collected} objects collected")

            # 5. Route calculator (medium)
            t0 = time.time()
            components['route_calculator'] = self._pickle_load(self.route_calculator_cache)
            size_mb = self.route_calculator_cache.stat().st_size / 1024 / 1024
            total_size += size_mb
            logger.info(f"  ✓ Route calculator ({size_mb:.1f} MB in {time.time()-t0:.2f}s)")
            gc.collect()

            # Verify all components loaded
            required_components = ['crime_analyzer', 'osm_analyzer', 'calibrated_weights', 
                                  'graph_builder', 'route_calculator']
            if not all(comp in components for comp in required_components):
                logger.error("Failed to load all required components")
                return None

            total_elapsed = time.time() - start_time
            throughput = total_size / total_elapsed if total_elapsed > 0 else 0
            logger.info(f"✓ Cache loaded: {total_size:.1f} MB in {total_elapsed:.2f}s ({throughput:.1f} MB/s)")
            logger.info(f"  Sequential loading minimized peak memory usage")
            
            # Final cleanup
            collected = gc.collect()
            if collected > 0:
                logger.info(f"✓ Final memory cleanup: {collected} objects collected")

            return (components['crime_analyzer'], 
                   components['osm_analyzer'],
                   components['calibrated_weights'],
                   components['graph_builder'],
                   components['route_calculator'])

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            import traceback
            traceback.print_exc()
            return None

    def clear_cache(self):
        """Clear all cached data"""
        logger.info("Clearing cache...")

        cache_files = [
            self.metadata_path,
            self.crime_analyzer_cache,
            self.osm_analyzer_cache,
            self.calibrated_weights_cache,
            self.graph_builder_cache,
            self.route_calculator_cache
        ]

        for cache_file in cache_files:
            if cache_file.exists():
                try:
                    cache_file.unlink()
                    logger.info(f"Deleted {cache_file.name}")
                except Exception as e:
                    logger.error(f"Failed to delete {cache_file.name}: {e}")

        logger.info("Cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache status"""
        metadata = self._load_metadata()

        cache_files = [
            self.crime_analyzer_cache,
            self.osm_analyzer_cache,
            self.calibrated_weights_cache,
            self.graph_builder_cache
        ]

        total_size = sum(f.stat().st_size for f in cache_files if f.exists())

        info = {
            'cache_exists': all(f.exists() for f in cache_files),
            'cache_directory': str(self.cache_dir),
            'total_size_mb': total_size / 1024 / 1024,
            'file_count': sum(1 for f in cache_files if f.exists()),
            'metadata': metadata.to_dict() if metadata else None
        }

        return info

