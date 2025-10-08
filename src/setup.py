
from setuptools import setup, find_packages

setup(
    name='hulu_med',  # 给你的包起一个名字
    version='0.1.0',      # 版本号
    packages=find_packages(), # 自动查找项目中的所有包（即包含__init__.py的目录）
)
