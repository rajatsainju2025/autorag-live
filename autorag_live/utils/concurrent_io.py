"""Concurrent I/O for efficient document loading."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List


def load_documents_concurrent(
    file_paths: List[str],
    max_workers: int = 4,
) -> List[str]:
    """
    Load documents concurrently from multiple files.

    Args:
        file_paths: List of file paths
        max_workers: Number of concurrent workers

    Returns:
        List of loaded documents
    """
    documents = []

    def load_file(path: str) -> str:
        """Load single file."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(load_file, path) for path in file_paths]
        for future in futures:
            try:
                documents.append(future.result())
            except Exception:
                # Skip failed loads
                pass

    return documents


async def load_documents_async(
    file_paths: List[str],
    max_concurrent: int = 4,
) -> List[str]:
    """
    Load documents asynchronously using asyncio.

    Args:
        file_paths: List of file paths
        max_concurrent: Maximum concurrent tasks

    Returns:
        List of loaded documents
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def load_file_async(path: str) -> str:
        async with semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, open(path).read)

    tasks = [load_file_async(path) for path in file_paths]
    return await asyncio.gather(*tasks, return_exceptions=False)


def load_documents_from_directory(
    directory: str,
    pattern: str = "*.txt",
    max_workers: int = 4,
) -> List[str]:
    """
    Load all documents from a directory.

    Args:
        directory: Directory path
        pattern: File pattern (e.g., "*.txt")
        max_workers: Number of concurrent workers

    Returns:
        List of loaded documents
    """
    dir_path = Path(directory)
    file_paths = [str(p) for p in dir_path.glob(pattern)]

    return load_documents_concurrent(file_paths, max_workers)
