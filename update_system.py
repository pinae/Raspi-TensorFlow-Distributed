from paramiko.client import SSHClient

client = SSHClient()
client.load_system_host_keys()

server_list = ['raspi1', 'raspi2']


def for_all(command):
    for i in server_list:
        print(i)
        client.connect(i, username='pi')
        stdin, stdout, stderr = client.exec_command(command)
        for line in stdout:
            print('... ' + line.strip('\n'))
        for line in stderr:
            print('... ' + line.strip('\n'))
        client.close()


if __name__ == "__main__":
    for_all('sudo apt-get update')
    for_all('sudo apt-get -y upgrade')
