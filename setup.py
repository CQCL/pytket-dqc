import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytket-dqc",
    version="0.0.1",
    author="Dan Mills",
    author_email="daniel.mills@cambridgequantum.com",
    description="Package for the distribution of quantum circuits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CQCL/pytket-dqc",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=['numpy>=1.11', 'pytket>=1.7.3', 'hypernetx>=1.2', 'celluloid>=0.2', 'igraph>=0.9.8', 'decorator>=5.1.0', 'pygraphviz>=1.7', 'importlib_resources>=5.7.1', 'pyzx>=0.7.0', 'pytket-pyzx>=0.24.0'],
    include_package_data=True,
)