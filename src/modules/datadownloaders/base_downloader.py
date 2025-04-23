# Standard library imports
from abc import ABC, abstractmethod
import sys
import os
import torch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm


class BaseDownloader(ABC):
    """
    Base class to build a Downloader class
    To build a child of this class and inherit the methods, need to implement the download method.
    """

    def __init__(
        self,
        cuda_devices: list = [0],
        use_multi_cpus: bool = False,
        use_multi_threads: bool = False,
        max_num_cpus: int = 1,
        max_num_threads: int = 1,
    ):
        self.cuda_devices: list = cuda_devices
        self.device = None

        # Multi-CPU settings
        self.use_multi_cpus: bool = use_multi_cpus  # Use parallel processing to speed up the pre-processing
        self.max_num_cpus = max_num_cpus if max_num_cpus is not None else multiprocessing.cpu_count()

        # Multi-thread settings
        self.use_multi_threads: bool = (
            use_multi_threads  # Use parallel processing to speed up the pre-processing
        )
        self.max_num_threads = max_num_threads if max_num_threads is not None else 1

        self.setup_cuda_environment()

    def setup_cuda_environment(self):
        """Configures environment to use specific GPU and manage memory."""
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.cuda_devices))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set the start method to 'spawn' to avoid issues with multiprocessing with CUDA when using the default 'fork' method rather than 'spawn'
        multiprocessing.set_start_method("spawn")

    def execute_tasks_multiprocesses(self, tasks):
        """
        Executes a list of tasks in parallel or sequentially, based on the use_multi_cpus flag,
        optimized for CPU-bound tasks.

        IMPORTANT: CUDA cannot be used in multithreading setting + needs to be used in a spawn mp_context in multiprocessing setting.

        Args:
            tasks (list): A list of tuples, each tuple containing a callable and its arguments.

        Yields:
            Results from executing the tasks as they are completed.
        """
        results = []
        if self.use_multi_cpus:
            with ProcessPoolExecutor(max_workers=self.max_num_cpus) as executor:
                print(
                    f"Using {executor._max_workers}/{multiprocessing.cpu_count()} of available CPUs for execution."
                )
                future_to_task = {executor.submit(task[0], *task[1]): task for task in tasks}
                for future in tqdm(
                    as_completed(future_to_task),
                    total=len(tasks),
                    desc="Processing in PARALLEL (multiprocessing)",
                    file=sys.stdout,
                ):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        task = future_to_task[future]
                        print(f"Task {task} failed: {e}")
        else:
            for task in tqdm(tasks, desc="Processing SEQUENTIALLY", file=self.tqdm_out, mininterval=5):
                try:
                    results.append(task[0](*task[1]))
                except Exception as e:
                    print(f"Task {task} failed: {e}")
        return results

    def execute_tasks_multithreads(self, tasks):
        """
        Executes a list of tasks in parallel or sequentially, based on the use_multi_threads flag,
        optimized for memory efficiency and I/O tasks

        IMPORTANT NOTE: This method is designed to be used with tasks that are IO-bound, not CPU-bound.
        Three reasons for using ThreadPoolExecutor rather than ProcessPoolExecutor:
        - Using ProcessPoolExecutor would require pickling the data to pass it to the subprocesses, which can be slow and MEMORY-INTENSIVE
        - ThreadPoolExecutor is more memory-efficient because it shares memory between threads
        - Since our operations involve reading and writing files (I/O tasks), the GIL won't be a bottleneck

        Args:
            tasks (list): A list of tuples, each tuple containing a callable and its arguments.

        Yields:
            Results from executing the tasks as they are completed.
        """
        results = []
        if self.use_multi_threads:
            with ThreadPoolExecutor(max_workers=self.max_num_threads) as executor:
                print(
                    f"Using {executor._max_workers}/{multiprocessing.cpu_count()} of available CPUs for execution."
                )
                future_to_task = {executor.submit(task[0], *task[1]): task for task in tasks}
                for future in tqdm(
                    as_completed(future_to_task),
                    total=len(tasks),
                    desc="Processing in PARALLEL (multithreading)",
                    file=self.tqdm_out,
                ):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        task = future_to_task[future]
                        print(f"Task {task} failed: {e}")
        else:
            for task in tqdm(tasks, desc="Processing SEQUENTIALLY", file=self.tqdm_out, mininterval=5):
                try:
                    results.append(task[0](*task[1]))
                except Exception as e:
                    print(f"Task {task} failed: {e}")
        return results

    @abstractmethod
    def download(self, *args, **kwargs):
        """
        Method to downloader the data.

        Raises
        ------
        NotImplementedError
            If the method is not implement
        """
        raise NotImplementedError
