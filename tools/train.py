import argparse
import os
import sys
from six.moves import shlex_quote


def new_cmd(session, name, cmd, mode, logdir, shell, redirect_file=None):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(shlex_quote(str(v)) for v in cmd)
    if mode == 'tmux':
        if redirect_file:
            return name, "tmux send-keys -t {}:{} {} Enter".format(session, name, shlex_quote(cmd+" 2>&1 | tee -a {}".format(redirect_file)))
        else:
            return name, "tmux send-keys -t {}:{} {} Enter".format(session, name, shlex_quote(cmd))
    elif mode == 'child':
        return name, "{} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh".format(cmd, logdir, session, name, logdir)
    elif mode == 'nohup':
        return name, "nohup {} -c {} >{}/{}.{}.out 2>&1 & echo kill $! >>{}/kill.sh".format(shell, shlex_quote(cmd),
                                                                                            logdir, session, name,
                                                                                            logdir)


def create_commands(session, num_epochs, lr, data_dir, log_dir, batch_size, num_workers, shell='bash', mode='tmux', sleep_worker=0.0):
    # for launching the TF workers and for launching tensorboard
    base_cmd = [
        'CUDA_VISIBLE_DEVICES=',
        sys.executable, 'worker.py',
        '--num_epochs', num_epochs,
        '--lr', lr,
        '--data_dir', data_dir,
        '--log_dir', log_dir,
        '--batch_size', batch_size,
        '--num_workers', num_workers,
    ]

    # ps
    cmds_map = [new_cmd(session, "ps", base_cmd + ["--job_name", "ps"], mode, log_dir, shell, 'logs/ps.log')]

    # workers for training
    for i in range(num_workers):
        cmds_map += [new_cmd(session, "w-%d" % i,
                             base_cmd + ["--job_name", "worker", "--task_index", str(i), ],
                             mode, log_dir, shell, 'logs/w-{}.log'.format(i))]

    # tensorboard
    cmds_map += [new_cmd(session, "tb", ["tensorboard", "--logdir", log_dir, "--port", "12345"], mode, log_dir, shell)]

    # htop watcher
    if mode == 'tmux':
        cmds_map += [new_cmd(session, "htop", ["htop"], mode, log_dir, shell)]

    windows = [v[0] for v in cmds_map]

    notes = []
    cmds = [
        "mkdir -p {}".format(log_dir),
        "echo {} {} > {}/cmd.sh".format(sys.executable, ' '.join([shlex_quote(arg) for arg in sys.argv if arg != '-n']),
                                        log_dir),
    ]
    if mode == 'nohup' or mode == 'child':
        cmds += ["echo '#!/bin/sh' >{}/kill.sh".format(log_dir)]
        notes += ["Run `source {}/kill.sh` to kill the job".format(log_dir)]
    if mode == 'tmux':
        notes += ["Use `tmux attach -t {}` to watch process output".format(session)]
        notes += ["Use `tmux kill-session -t {}` to kill the job".format(session)]
    else:
        notes += ["Use `tail -f {}/*.out` to watch process output".format(log_dir)]
    notes += ["Point your browser to http://localhost:12345 to see Tensorboard"]

    if mode == 'tmux':
        cmds += [
            "kill $( lsof -i:12345 -t ) > /dev/null 2>&1",  # kill any process using tensorboard's port
            "kill $( lsof -i:12222-{} -t ) > /dev/null 2>&1".format(num_workers + 12222),  # kill
            "tmux kill-session -t {}".format(session),
            "tmux new-session -s {} -n {} -d {}".format(session, windows[0], shell)
        ]
        for w in windows[1:]:
            cmds += ["tmux new-window -t {} -n {} {}".format(session, w, shell)]
        cmds += ["sleep 1"]
    for window, cmd in cmds_map:
        cmds += [cmd]
        if window[0] == 'w':
            cmds += ["sleep {}".format(sleep_worker)]

    return cmds, notes


def run():
    parser = argparse.ArgumentParser(description="Run commands")
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--lr', type=float, default=0.01,
                      help='Initial learning rate')
    parser.add_argument('--data_dir', type=str, default='/home/wing/DataSet/criteo/pre/deepffm/downSample',
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='/home/wing/Project/DeepFFM/logs',
                      help='Summaries log directory')
    parser.add_argument('--batch_size', default=1000, type=int, help='Batch size')
    parser.add_argument('--num_workers', default=3, type=int, help='Number of workers')

    parser.add_argument('--sleep_worker', default=0, type=float,
                        help='sleeping time after starting a worker (before starting the next worker)')
    parser.add_argument('-m', '--mode', type=str, default='tmux',
                        help="tmux: run workers in a tmux session. nohup: run workers with nohup. child: run workers as child processes")
    parser.add_argument('--sess_name', type=str, default='deepffm',
                        help="tmux: run workers in a tmux session. nohup: run workers with nohup. child: run workers as child processes")
    args = parser.parse_args()
    cmds, notes = create_commands(args.sess_name, args.num_epochs, args.lr, args.data_dir, args.log_dir, args.batch_size, args.num_workers, sleep_worker=args.sleep_worker)

    print("\n".join(cmds))
    print("")
    if args.mode == "tmux":
        os.environ["TMUX"] = ""
    os.system("\n".join(cmds))
    print('\n'.join(notes))


if __name__ == "__main__":
    run()
