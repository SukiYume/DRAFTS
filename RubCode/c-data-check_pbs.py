import os, time, subprocess

if __name__ == '__main__':

    root_path      = '/home/ykzhang/DataCheck/'
    script_path    = '{}pbsspt/'.format(root_path)
    os.makedirs(script_path, exist_ok=True)

    script_name    = 'c-data-check.py'

    node_list = [i for i in range(3, 20)] + [3, 4]
    for i in range(19):
        node   = node_list[i // 1]
        node       = '{:0>2d}'.format(node)
        section    = '{:0>2d}'.format(i)
        pbs_string = '''
#PBS -N anaconda-{0}
#PBS -o {1}pbsspt/m{0}-output.log
#PBS -e {1}pbsspt/m{0}-output.err
#PBS -q gpu
#PBS -l nodes=gpu{2}
source /home/ykzhang/.bashrc
cd {1}
python {3} {0}
        '''.format(section, root_path, node, script_name)
        pbs_file_name = 'node-{}-section-{}.pbs'.format(node, section)

        with open(script_path + pbs_file_name, 'w') as f:
            f.write(pbs_string)
        subprocess.Popen('qsub {}{}'.format(script_path, pbs_file_name), shell=True)
        time.sleep(5)

