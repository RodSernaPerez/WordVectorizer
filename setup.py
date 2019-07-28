#!/usr/bin/python

# Copyright 2018 Altran
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# THIS FILE IS MANAGED BY THE GLOBAL REQUIREMENTS REPO - DO NOT EDIT
# Copyright 2011 OpenStack, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements_file_content = f.read().splitlines()
requirements = []
links = []
for r in requirements_file_content:
    if 'git+' in r[:4]:
        os.system("pip install " + r)
    else:
        requirements.append(r)

with open('README.md') as f:
    long_description = f.read()

setup(
    name='word_vectorizer',
    version='1.1',
    author='Rodrigo Serna',
    author_email='rodrigo.sernaperez@altran.com',
    description='Gets the word embedding of a word.',
    long_description=long_description,
    install_requires=requirements,
    dependency_links=links,
    # packages=find_packages(exclude=['tests', '*.tests', '*.tests.*']),
    packages=find_packages(exclude=[]),
    package_data={'': ['*.json']},
    license='Apache License 2.0',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Topic :: System"
        "Framework :: Flask"
    ],

)
