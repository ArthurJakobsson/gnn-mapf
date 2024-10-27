import ray
import time

@ray.remote
def run_task(s):
    time.sleep(2) # simulate work
    
    if len(s) < 2:
        futures = [run_task.remote(s+"a"), run_task.remote(s+"b")] # child tasks
    else:
        futures = []
    worker_id = ray.get_runtime_context().get_worker_id()
    return (s, worker_id, futures)

num_workers = 3
ray.init(num_cpus=num_workers) # num_workers

# run actual tasks
t0 = time.time()
futures = [run_task.remote(s) for s in ["1", "2"]]

# wait for all tasks and child tasks to finish
while len(futures):
    ready, futures = ray.wait(futures)
    for finished_task in ready:
        s, worker_id, child_futures = ray.get(finished_task)
        print(f"task {s} finished at {int(time.time() - t0)}s by worker {worker_id}")
        futures += child_futures

ray.shutdown()
