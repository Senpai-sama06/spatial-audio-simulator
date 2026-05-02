import psutil

class MemoryManager:
    MAX_UTILIZATION_THRESHOLD = 0.80

    @classmethod
    def get_available_memory(cls) -> float:
        """
        Returns the available system memory in bytes.
        """
        mem = psutil.virtual_memory()
        return mem.available

    @classmethod
    def get_max_allowed_memory(cls) -> float:
        """
        Returns the maximum allowed memory utilization (80% of total system memory).
        """
        mem = psutil.virtual_memory()
        return mem.total * cls.MAX_UTILIZATION_THRESHOLD

    @classmethod
    def should_shrink_chunk(cls) -> bool:
        """
        Checks if the current memory utilization exceeds the threshold.
        """
        mem = psutil.virtual_memory()
        utilization = mem.used / mem.total
        return utilization >= cls.MAX_UTILIZATION_THRESHOLD

    @classmethod
    def calculate_chunk_size(cls, current_chunk_size: int, complexity_factor: float) -> int:
        """
        Dynamically calculates chunk size based on memory threshold and complexity.
        """
        if cls.should_shrink_chunk():
            # Shrink drastically if we are over the limit
            return max(1024, int(current_chunk_size * 0.5))
        
        # Grow chunk size if there's plenty of headroom, adjusted by complexity
        # Keep maximum chunk size reasonable to allow periodic checks
        max_chunk = 48000 * 60 # max 1 minute at 48kHz
        return min(max_chunk, int(current_chunk_size * (1.1 / complexity_factor)))
