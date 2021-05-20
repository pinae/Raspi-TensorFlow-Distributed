from paramiko.client import SSHClient
from scp import SCPClient
from select import select

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


worker_list = {'raspi1': {'text_color': Colors.OKBLUE, 'warn_color': Colors.WARNING},
               'raspi2': {'text_color': Colors.OKGREEN, 'warn_color': Colors.FAIL}}
USERNAME = 'pi'
RECV_BUFFER_SIZE = 4096


def place_code_on_workers():
    client = SSHClient()
    client.load_system_host_keys()
    for worker in worker_list.keys():
        client.connect(worker, username=USERNAME)
        scp = SCPClient(client.get_transport())
        scp.put(['worker.py'], remote_path='/home/{}/'.format(USERNAME))
        print("{color}[{}] {}{end_color}".format(
            worker, "Copied worker.py",
            color=worker_list[worker]['text_color'],
            end_color=Colors.ENDC))
        client.close()


def start_training():
    client_list = []
    for worker_name in worker_list.keys():
        client = SSHClient()
        client.load_system_host_keys()
        client.connect(worker_name, username=USERNAME)
        channel = client.get_transport().open_session()
        client_list.append((worker_name, client, channel, {'stdout': "", 'stderr': ""}))
    for worker_name, client, channel, outs in client_list:
        channel.exec_command('python3 worker.py')
    at_least_one_worker_still_running = True
    while at_least_one_worker_still_running:
        at_least_one_worker_still_running = False
        for worker_name, client, channel, outs in client_list:
            rl, wl, xl = select([channel], [], [], 0.0)
            if not channel.exit_status_ready():
                at_least_one_worker_still_running = True
            if len(rl) > 0:
                outs['stdout'] += channel.recv(RECV_BUFFER_SIZE).decode('utf-8')
                outs['stderr'] += channel.recv_stderr(RECV_BUFFER_SIZE).decode('utf-8')
                if channel.exit_status_ready():
                    if len(outs['stdout']) > 0:
                        print("{color}[{}] {}{end_color}".format(
                            worker_name, outs['stdout'],
                            color=worker_list[worker_name]['text_color'],
                            end_color=Colors.ENDC))
                        outs['stdout'] = ""
                    if len(outs['stderr']) > 0:
                        print("{color}[{}] {}{end_color}".format(
                            worker_name, outs['stderr'],
                            color=worker_list[worker_name]['text_color'],
                            end_color=Colors.ENDC))
                        outs['stderr'] = ""
                lines_stdout = outs['stdout'].split("\n")
                outs['stdout'] = lines_stdout[-1]
                lines_stderr = outs['stderr'].split("\n")
                outs['stderr'] = lines_stderr[-1]
                for line in lines_stdout[:-1]:
                    print("{color}[{}] {}{end_color}".format(
                        worker_name, line,
                        color=worker_list[worker_name]['text_color'],
                        end_color=Colors.ENDC))
                for line in lines_stderr[:-1]:
                    print("{color}[{}] {}{end_color}".format(
                        worker_name, line,
                        color=worker_list[worker_name]['warn_color'],
                        end_color=Colors.ENDC))
    for worker_name, client, channel, outs in client_list:
        client.close()


if __name__ == "__main__":
    place_code_on_workers()
    start_training()
