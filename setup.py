from distutils.command.build_py import build_py as BuildPyCommand
from distutils.spawn import find_executable
import os.path as osp
from setuptools import find_packages
from setuptools import setup
import shlex
import subprocess
import sys


version = '2.2.1'


if sys.argv[1] == 'release':
    for cmd in [
            'python setup.py sdist upload',
            'git tag v{0}'.format(version),
            'git push origin master --tag']:
        subprocess.call(shlex.split(cmd))
    sys.exit(0)


class LabelmeBuildPyCommand(BuildPyCommand):

    def run(self):
        if find_executable('pyrcc4') is None:
            sys.stderr.write('Please install pyrcc4 command.\n')
            sys.stderr.write('(See https://github.com/wkentaro/labelme.git)\n')
            sys.exit(1)
        this_dir = osp.dirname(osp.abspath(__file__))
        package_dir = osp.join(this_dir, 'labelme')
        src = 'resources.qrc'
        dst = 'resources.py'
        print('converting {0} -> {1}'
              .format(osp.join(package_dir, src), osp.join(package_dir, dst)))
        cmd = 'pyrcc4 -o {1} {0}'.format(src, dst)
        subprocess.call(shlex.split(cmd), cwd=package_dir)
        BuildPyCommand.run(self)


try:
    import PyQt4
except ImportError:
    sys.stderr.write('Please install PyQt4.\n')
    sys.exit(1)


setup(
    name='labelme',
    version=version,
    packages=find_packages(),
    cmdclass={'build_py': LabelmeBuildPyCommand},
    description='Simple Image Annotation Tool.',
    long_description=open('README.md').read(),
    author='Kentaro Wada',
    author_email='www.kentaro.wada@gmail.com',
    url='https://github.com/mpitid/pylabelme',
    install_requires=[],
    license='GPLv3',
    keywords='Image Annotation, Machine Learning',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX',
        'Topic :: Internet :: WWW/HTTP',
    ],
    package_data={'labelme': ['icons/*', 'resources.qrc']},
    entry_points={'console_scripts': ['labelme=labelme.app:main']},
    scripts=['scripts/labelme_draw_json', 'scripts/labelme_json_to_dataset'],
)
