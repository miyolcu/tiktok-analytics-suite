# Data Directory

## Structure

- `sample/` - Small sample datasets for testing
- `outputs/` - Generated datasets (gitignored, large files) 
- `schemas/` - JSON schemas describing data structure

## Usage

Generated data files will be saved to `outputs/` directory:
- `users_TIMESTAMP.csv` - User profiles and metadata
- `videos_TIMESTAMP.csv` - Video content and metrics  
- `interactions_TIMESTAMP.csv` - User interaction data
- `summary_TIMESTAMP.json` - Dataset summary and statistics

## Data Size Expectations

- 10K users, 100K videos: ~50MB
- 80K users, 1M videos: ~500MB  
- 200K users, 5M videos: ~2GB

Large datasets are automatically excluded from git commits.
