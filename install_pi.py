from paramiko.client import SSHClient

SERVER = 'raspi1'
USER = 'pi'

client = SSHClient()
client.load_system_host_keys()
client.connect(SERVER, username=USER)

install_pre = """
sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev gcc gfortran python-dev libgfortran5
sudo apt-get install -y libatlas3-base libatlas-base-dev libopenblas-dev libopenblas-base libblas-dev
sudo apt-get install -y liblapack-dev cython libatlas-base-dev openmpi-bin libopenmpi-dev python3-dev
sudo apt-get install -y python3-pip
sudo pip3 install keras_applications==1.0.8 --no-deps
sudo pip3 install keras_preprocessing==1.1.2 --no-deps
sudo pip3 install h5py==2.10.0
sudo pip3 install pybind11
sudo pip3 install 'setuptools>=41.0.0'
pip3 install -U --user six wheel mock
"""

tensorflow_version = 'tensorflow-2.4.0-cp37-none-linux_armv7l'

install_tf = f"""
wget "https://raw.githubusercontent.com/PINTO0309/Tensorflow-bin/master/{tensorflow_version}_download.sh"
sudo bash ./{tensorflow_version}_download.sh
sudo pip3 uninstall -y tensorflow
sudo -H pip3 install {tensorflow_version}.whl
"""

commands = install_pre + install_tf

print(SERVER)

for i in commands.splitlines():
    print(i)
    stdin, stdout, stderr = client.exec_command(i)
    for line in stdout:
        print('... ' + line.strip('\n'))
    for line in stderr:
        print('... ' + line.strip('\n'))
client.close()

print('ready')
