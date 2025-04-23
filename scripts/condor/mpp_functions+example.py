#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Patch

# patch allowing tdqm to be used with starmap
# obtained from:
# https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)

mpp.Pool.istarmap = istarmap


#%% Wrapper for istarmap

def istarmap_with_kwargs(pool, fn, args_iter, kwargs_iter=repeat({})):
    
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
        
    return pool.istarmap(apply_args_and_kwargs, args_for_starmap)


def apply_args_and_kwargs(fn, args, kwargs=None):
    
    if kwargs is None:
        return fn(*args)
    else:
        return fn(*args, **kwargs)

#%% Example

with mpp.Pool(process_nb) as pool:
    results = list(tqdm(
        istarmap_with_kwargs(
            pool,
            function,
            args_iter,
            repeat(kwargs_dict)
            ),
        total=task_nb,
        desc=description,
        file=sys.stdout,
        smoothing=0,
        disable=disable
        ))