# Milestone 2: Async ETL Pipeline & Data Infrastructure

## Overview
Build a robust, scalable data pipeline for ingesting, processing, and storing trajectory data with support for the NGSIM dataset and extensibility for future data sources.

## Duration: 5-6 days

## Objectives
- Implement async ETL pipeline using asyncio and aiohttp
- Set up columnar storage with Apache Parquet and partitioning
- Create NGSIM dataset integration with generic schema
- Implement data quality validation and cleaning
- Add incremental loading and change data capture

## Dependencies
- **M1**: Project foundation and architecture setup

## Acceptance Criteria

### 1. Async ETL Framework ✅
- **AsyncDataSource** abstract base class
- **AsyncETLPipeline** orchestrating data ingestion
- **Concurrent processing** using asyncio for I/O-bound operations
- **Error handling** with retry logic and graceful degradation
- **Progress tracking** with async observers

### 2. NGSIM Dataset Integration ✅
- **NGSIM data loader** supporting:
  - US-101 highway dataset
  - I-80 highway dataset
  - Peachtree Street dataset
- **Automatic download** from official NGSIM sources
- **Schema validation** ensuring data consistency
- **Coordinate transformation** to standardized format

### 3. Columnar Storage System ✅
- **Parquet storage** with optimal compression (snappy/gzip)
- **Partitioning strategy** by date/highway/vehicle_type
- **Schema evolution** support for different dataset versions
- **Metadata catalog** tracking dataset lineage
- **Query optimization** with predicate pushdown

### 4. Data Quality Pipeline ✅
- **Pydantic models** for trajectory validation:
  - Spatial constraints (coordinate bounds)
  - Temporal constraints (timestamp ordering)
  - Physical constraints (speed/acceleration limits)
- **Data cleaning** operations:
  - Outlier detection and removal
  - Missing value imputation
  - Trajectory smoothing
  - Duplicate removal

### 5. Incremental Loading ✅
- **Change data capture** detecting new/modified data
- **Checkpointing** for resumable data ingestion
- **Delta processing** avoiding full reprocessing
- **Data versioning** with timestamp-based partitions

## Technical Deliverables

1. **AsyncETLPipeline** class with configuration support
2. **NGSIMDataSource** implementing async data loading
3. **ParquetStorage** backend with partitioning
4. **TrajectoryValidator** using Pydantic models
5. **DataQualityReport** generating quality metrics
6. **IncrementalLoader** supporting delta updates

## Data Schema Design

```python
class TrajectoryPoint(BaseModel):
    vehicle_id: int
    timestamp: datetime
    x_position: float  # meters
    y_position: float  # meters
    velocity_x: float  # m/s
    velocity_y: float  # m/s
    acceleration_x: Optional[float] = None  # m/s²
    acceleration_y: Optional[float] = None  # m/s²
    lane_id: Optional[int] = None
    highway_id: str
    
class TrajectorySegment(BaseModel):
    trajectory_id: UUID
    vehicle_id: int
    start_time: datetime
    end_time: datetime
    points: List[TrajectoryPoint]
    metadata: Dict[str, Any]
```

## Performance Requirements
- **Ingestion Rate**: >10,000 trajectory points/second
- **Storage Efficiency**: <50% of raw CSV size with Parquet
- **Query Performance**: <1 second for single vehicle trajectory
- **Memory Usage**: <2GB for processing 1M trajectory points

## Dependencies & Integration Points
- **Apache Arrow/Parquet** for columnar storage
- **Polars** for high-performance data processing
- **DuckDB** for analytical queries
- **aiohttp/httpx** for async data fetching

## Risks & Mitigations
- **Risk**: NGSIM dataset availability or format changes
- **Mitigation**: Cache datasets locally, version schema definitions
- **Risk**: Memory issues with large trajectory files
- **Mitigation**: Streaming processing, chunked operations

## Success Criteria
- [ ] NGSIM datasets successfully downloaded and ingested
- [ ] Parquet storage with <50% size of original CSV
- [ ] Data quality validation catching >95% of anomalies
- [ ] Incremental loading working with sample updates
- [ ] Pipeline processing 100k+ trajectory points without errors
- [ ] All async operations properly tested with mocks

## Notes
- Prioritize data integrity over processing speed
- Design for horizontal scaling (multiple workers)
- Ensure all operations are idempotent
