import torch
import multiprocessing as mp

from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper

def _worker(remote, parent_remote, manager_fn_wrapper):
    parent_remote.close()

    torch.set_num_threads(1)
    manager = manager_fn_wrapper.var()

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "gather_rollouts":
                rollouts = manager.gather_rollouts()
                remote.send(rollouts)
            elif cmd == "update_policy":
                policy_dict, policy_id = data
                manager._update_policy(policy_dict, policy_id)
                remote.send(True)
            elif cmd == "update_annealing_factor":
                factor = data
                manager._update_annealing_factor(factor)
                remote.send(True)
            elif cmd == "close":
                remote.close()
                break
        except EOFError:
            break

class SubProcGameManager(object):
    def __init__(self, manager_fns, start_method=None):
        self.waiting = False
        self.closed = False
        n_processes = len(manager_fns)

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_processes)])
        self.processes = []

        for work_remote, remote, manager_fn in zip(self.work_remotes, self.remotes, manager_fns):
            args = (work_remote, remote, CloudpickleWrapper(manager_fn))
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

    def gather_rollouts_async(self):
        for remote in self.remotes:
            remote.send(("gather_rollouts", None))
        self.waiting = True

    def gather_rollouts_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return results

    def gather_rollouts(self):
        self.gather_rollouts_async()
        return self.gather_rollouts_wait()

    def update_policy(self, policy_dict, policy_id=0):
        for remote in self.remotes:
            remote.send(("update_policy", (policy_dict, policy_id)))
        results = [remote.recv() for remote in self.remotes]
        return results

    def update_annealing_factor(self, annealing_factor):
        for remote in self.remotes:
            remote.send(("update_annealing_factor", annealing_factor))
        results = [remote.recv() for remote in self.remotes]
        return results

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True
