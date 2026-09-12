"""Short Linux CPU check: a timed-out nested runner must stop its worker."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from mechanism_candidates import require, save, sha
from run_mechanism_screen import run_child


def state(pid):
    path = Path('/proc') / str(pid) / 'stat'
    try:
        # Names can contain spaces; the state follows the closing parenthesis.
        return path.read_text().rsplit(')', 1)[1].split()[0]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fixture', choices=['runner', 'worker'])
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args()
    require(sys.platform == 'linux', 'Linux process semantics required; no installation needed')
    if args.fixture:
        if args.fixture == 'worker':
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        save(args.directory / (args.fixture + '.json'), dict(pid=os.getpid(), pgid=os.getpgrp()))
        if args.fixture == 'worker':
            while True:
                signal.pause()
        run_child([sys.executable, str(Path(__file__).resolve()), '--fixture', 'worker',
                   '--directory', str(args.directory)], args.directory / 'worker.log',
                  dict(os.environ), 60, grace_seconds=.1)
        return
    require(args.output and not args.output.exists(), 'Fresh native process receipt required')
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='easysteer_owned_cleanup_') as directory:
        folder = Path(directory)
        try:
            run_child([sys.executable, str(Path(__file__).resolve()), '--fixture', 'runner',
                       '--directory', directory], folder / 'runner.log', dict(os.environ),
                      3, grace_seconds=1)
        except subprocess.TimeoutExpired:
            pass
        else:
            raise ValueError('Fixture did not reach its deliberate timeout')
        from mechanism_candidates import read
        processes = {name: read(folder / (name + '.json')) for name in ['runner', 'worker']}
        require(all(p['pid'] == p['pgid'] for p in processes.values()), 'Fixture did not create separate groups')
        observed = {name: state(p['pid']) for name, p in processes.items()}
        # A regression must not leave the deliberately stubborn CPU fixture alive.
        for name, value in observed.items():
            if value not in (None, 'Z'):
                try:
                    os.killpg(processes[name]['pgid'], signal.SIGKILL)
                except ProcessLookupError:
                    pass
        require(all(value in (None, 'Z') for value in observed.values()), 'Owned descendant is still alive')
        result = dict(status='nested_timeout_stopped_owned_separate_groups', process_states=observed,
            processes=processes, elapsed_seconds=time.perf_counter()-started,
            checker_sha256=sha(Path(__file__), source=True), model_loads=0, GPU_calls=0,
            scope='Two fresh CPU fixtures only; worker ignores TERM. No foreign process is signalled. Z means exited awaiting OS reaping, not running.')
        save(args.output, result)
        print(result)


if __name__ == '__main__':
    main()
