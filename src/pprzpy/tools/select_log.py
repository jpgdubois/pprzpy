from pathlib import Path
from typing import Tuple
import logging
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_latest_log_paths(root_dir: Path, recursive: bool = False) -> Tuple[Path, Path]:
    """
    Find most recent *.data file (recursive optional) and its matching .log.
    
    Args:
        root_dir: Search starting directory
        recursive: If True, search all subdirectories
        
    Returns:
        (data_path, log_path) of most recent pair
    """
    logger.info(f"Searching for *.data files in {root_dir}{' (recursive)' if recursive else ''}")
    
    if recursive:
        data_files = list(root_dir.rglob("*.data"))
    else:
        data_files = list(root_dir.glob("*.data"))
    
    if not data_files:
        logger.error(f"No *.data files found in {root_dir} {'(recursive)' if recursive else ''}")
        raise FileNotFoundError(f"No *.data files found in {root_dir} {'(recursive)' if recursive else ''}")
    
    logger.info(f"Found {len(data_files)} *.data files")
    
    # Find most recent by modification time
    latest_data = max(data_files, key=lambda p: p.stat().st_mtime)
    logger.debug(f"Most recent data file: {latest_data} (mtime: {latest_data.stat().st_mtime})")
    
    protocol_log = latest_data.with_suffix('.log')
    
    if not protocol_log.exists():
        logger.error(f"No matching .log found for {latest_data}")
        raise FileNotFoundError(f"No matching .log for {latest_data}")
    
    logger.info(f"Selected data:     {latest_data}")
    logger.info(f"Selected protocol: {protocol_log}")
    
    return latest_data, protocol_log


# Usage
if __name__ == "__main__":
    data_dir = Path("data")
    try:
        file_path, protocol_path = get_latest_log_paths(data_dir, recursive=True)
        print(f"Data: {file_path}")
        print(f"Log:  {protocol_path}")
    except (FileNotFoundError, IndexError) as e:
        logger.error(f"Failed to find log paths: {e}")
        exit(1)
